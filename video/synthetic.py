"""킥 템플릿(JSON)으로 가짜 랜드마크를 합성한다 — 실제 영상 없이 파이프라인 검증·체험용."""

import numpy as np

from biomech.body3d import to_3d
from biomech.kinematics import compute_motion
from video.angles_from_pose import LANDMARKS, landmarks_from_fk
from video.pose_series import PoseSeries
from video.preprocess import swap_sides

IMG_W, IMG_H = 1920, 1080
PX_PER_M = 380.0
GROUND_PX = 1000.0
REAR_NOSE_VIS = 0.2         # 뒤에서 찍으면 얼굴이 안 보여 코 visibility가 낮다


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


def rear_landmarks(pts: dict[str, np.ndarray], H: float) -> dict[str, np.ndarray]:
    """3D 모델을 뒤에서 본 랜드마크 (x = 선수의 오른쪽 +, y = 위). 뒤에서 보면 선수의 왼쪽이 화면 왼쪽."""
    p3 = to_3d(pts, H)
    names = {"nose": "head"}
    for s in "rl":
        names.update({f"{s}_shoulder": f"{s}_shoulder", f"{s}_elbow": f"{s}_elbow", f"{s}_wrist": f"{s}_hand",
                      f"{s}_hip": f"{s}_hip", f"{s}_knee": f"{s}_knee", f"{s}_ankle": f"{s}_ankle",
                      f"{s}_heel": f"{s}_heel", f"{s}_toe": f"{s}_toe"})
    return {k: np.stack([-p3[v][:, 2], p3[v][:, 1]], axis=1) for k, v in names.items()}


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
    lm = rear_landmarks(pts, H) if view == "rear" else landmarks_from_fk(pts)
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
    xy = {k: np.stack([IMG_W / 2 + sx * (v[:, 0] - cx) * px_per_m, GROUND_PX - v[:, 1] * px_per_m], axis=1)
          for k, v in lm.items()}
    vis = {k: np.ones(len(idx)) for k in LANDMARKS}
    if view == "rear":
        vis["nose"][:] = REAR_NOSE_VIS
    return PoseSeries(xy=xy, vis=vis, width=IMG_W, height=IMG_H, meta_fps=fps,
                      source=f"synthetic:{kick['id']}:{view}")
