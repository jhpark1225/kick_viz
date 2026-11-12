"""킥 템플릿(JSON)으로 가짜 랜드마크를 합성한다 — 실제 영상 없이 파이프라인 검증·체험용."""

import numpy as np

from biomech.body3d import foot_yaw_series, to_3d
from biomech.kinematics import compute_motion
from video.angles_from_pose import LANDMARKS, landmarks_from_fk
from video.pose_series import PoseSeries
from video.preprocess import swap_sides

IMG_W, IMG_H = 1920, 1080
PX_PER_M = 380.0
GROUND_PX = 1000.0
REAR_NOSE_VIS = 0.2         # 뒤에서 찍으면 얼굴이 안 보여 코 visibility가 낮다
REAR_CAMERA_DIST_M = 3.5    # 후면 카메라 ~ 디딤발(공 옆) 앞뒤 거리. 도움닫기 시작(약 0.8 m 뒤)·백스윙 발(약 1.3 m 뒤)까지 화면에 들어옴
REAR_CAMERA_HEIGHT_RATIO = 0.5  # 카메라 높이 ≈ 허리 (× 키, 촬영 가이드 '카메라 높이는 허리 정도')
REAR_MIN_DEPTH_M = 0.5      # 원근 나눗셈 보호 (템플릿 관절은 카메라에서 2 m 이상 떨어져 있음)
REAR_BOTTOM_MARGIN_PX = 30.0  # 후면 합성 영상에서 가장 낮은 점과 화면 아래 끝 사이 여백


def template_frames(kick: dict, H: float, fps: float, time_scale: float = 1.0):
    """템플릿 동작을 fps 간격으로 샘플링한다. time_scale > 1이면 동작이 느려진다(시간 늘이기).

    반환: (영상 시각 배열, Motion에서 고른 인덱스, Motion)
    """
    kfs = kick["keyframes"]
    t0, t1 = kfs[0]["t"], kfs[-1]["t"]
    n = int(np.floor((t1 - t0) * time_scale * fps + 1e-9)) + 1
    t_video = np.arange(n) / fps
    t_tpl = np.round(t0 + t_video / time_scale, 9)
    m = compute_motion(kick, H, n=2, extra_times=list(t_tpl))
    idx = np.searchsorted(m.t, t_tpl)
    assert np.allclose(m.t[idx], t_tpl, atol=1e-8)
    return t_video, idx, m


def rear_camera_x(kick: dict, m) -> float:
    """후면 카메라 기준 위치: 임팩트 순간 디딤발 발목의 x (공 옆). 카메라는 여기서 REAR_CAMERA_DIST_M 뒤에 둔다."""
    return float(np.interp(kick["impact"]["time"], m.t, m.points["l_ankle"][:, 0]))


def rear_landmarks(pts: dict[str, np.ndarray], H: float, ref_x: float | None = None,
                   foot_yaw_deg: float | np.ndarray = 0.0) -> dict[str, np.ndarray]:
    """3D 모델을 뒤에서 본 랜드마크 (x = 선수의 오른쪽 +, y = 위). 뒤에서 보면 선수의 왼쪽이 화면 왼쪽.

    ref_x가 있으면 실제 카메라처럼 **원근**으로 본다: 카메라는 ref_x(디딤발)에서 REAR_CAMERA_DIST_M 뒤, 허리 높이,
    몸 중심선 위. 거리 d인 점은 REAR_CAMERA_DIST_M / d배로 보이므로(핀홀 카메라), 디딤발 깊이에서는 실제 크기(m)이고
    도움닫기 초반·백스윙 때 카메라 쪽으로 온 발은 크고 낮게, 팔로스루 때 멀어진 발은 작고 높게 보인다.
    정사영(ref_x=None)으로 보면 앞뒤(시상면) 움직임이 모두 사라져 다리가 위아래로만 움직인다.
    foot_yaw_deg: 차는 발 방향(발끝 바깥 +, `foot_yaw_series`) — 인사이드 킥처럼 발을 돌리면 뒤에서 발이 옆으로 보인다.
    """
    p3 = to_3d(pts, H, foot_yaw_deg=foot_yaw_deg)
    names = {"nose": "head"}
    for s in "rl":
        names.update({f"{s}_shoulder": f"{s}_shoulder", f"{s}_elbow": f"{s}_elbow", f"{s}_wrist": f"{s}_hand",
                      f"{s}_hip": f"{s}_hip", f"{s}_knee": f"{s}_knee", f"{s}_ankle": f"{s}_ankle",
                      f"{s}_heel": f"{s}_heel", f"{s}_toe": f"{s}_toe"})
    if ref_x is None:
        return {k: np.stack([-p3[v][:, 2], p3[v][:, 1]], axis=1) for k, v in names.items()}
    xc, yc = ref_x - REAR_CAMERA_DIST_M, REAR_CAMERA_HEIGHT_RATIO * H
    out = {}
    for k, v in names.items():
        X, Y, Z = p3[v][:, 0], p3[v][:, 1], p3[v][:, 2]
        s = REAR_CAMERA_DIST_M / np.maximum(X - xc, REAR_MIN_DEPTH_M)
        out[k] = np.stack([-Z * s, yc + (Y - yc) * s], axis=1)
    return out


def synthesize(kick: dict, H: float = 1.75, fps: float = 120.0, *, noise_frac: float = 0.0,
               time_scale: float = 1.0, mirror: bool = False, left_footed: bool = False,
               px_per_m: float = PX_PER_M, view: str = "side", seed: int = 0) -> PoseSeries:
    """템플릿 → 픽셀 좌표 랜드마크(y 아래로 증가).

    noise_frac: 가우시안 노이즈 σ (키 대비 비율, 예: 0.005 = 키의 0.5%).
    mirror: 좌우 반전 영상(측면: 왼쪽으로 차는 영상, 후면: 정면 촬영처럼 좌우가 바뀜).
    left_footed: 왼발 킥(랜드마크 좌우 이름 교환). view: "side"(측면) 또는 "rear"(후면).
    """
    _, idx, m = template_frames(kick, H, fps, time_scale)
    pts = {k: v[idx] for k, v in m.points.items()}
    if view == "rear":
        lm = rear_landmarks(pts, H, rear_camera_x(kick, m), foot_yaw_series(kick, m.t[idx]))
    else:
        lm = landmarks_from_fk(pts)
    rng = np.random.default_rng(seed)
    if noise_frac > 0:
        lm = {k: v + rng.normal(0, noise_frac * H, v.shape) for k, v in lm.items()}
    if left_footed:
        lm = swap_sides(lm)
        if view == "rear":      # 왼발잡이는 거울상: 좌우 위치도 바뀐다
            lm = {k: v * np.array([-1.0, 1.0]) for k, v in lm.items()}
    allx = np.concatenate([v[:, 0] for v in lm.values()])
    cx = (allx.min() + allx.max()) / 2
    sx = -1.0 if mirror else 1.0
    ground_px = GROUND_PX
    if view == "rear":      # 원근: 도움닫기 때 카메라에 가까운 발이 지면선보다 아래로 보이므로 화면 안에 들어오게 올린다
        ymin = float(min(v[:, 1].min() for v in lm.values()))
        ground_px = min(GROUND_PX, IMG_H - REAR_BOTTOM_MARGIN_PX + ymin * px_per_m)
    xy = {k: np.stack([IMG_W / 2 + sx * (v[:, 0] - cx) * px_per_m, ground_px - v[:, 1] * px_per_m], axis=1)
          for k, v in lm.items()}
    vis = {k: np.ones(len(idx)) for k in LANDMARKS}
    if view == "rear":
        vis["nose"][:] = REAR_NOSE_VIS
    return PoseSeries(xy=xy, vis=vis, width=IMG_W, height=IMG_H, meta_fps=fps,
                      source=f"synthetic:{kick['id']}:{view}")
