"""랜드마크 전처리: 결측 보간, 저역통과 필터, 차는 다리 판별, 방향 정규화, 스케일, 품질 검사."""

import numpy as np
from scipy.signal import butter, filtfilt

from biomech.body import SEGMENT_RATIOS
from video.angles_from_pose import angles_from_pose, frontal_angles
from video.pose_series import PoseSeries, Processed

VIS_MIN = 0.5               # 이보다 낮은 visibility는 결측으로 본다
MAX_GAP_FRAMES = 5          # 선형 보간 신뢰 한계 (프레임)
DEFAULT_CUTOFF_HZ = 8.0     # 4차 Butterworth 저역통과 차단 주파수 (생체역학 표준 처리)
CUTOFF_RANGE_HZ = (6.0, 12.0)
FILTER_ORDER = 4
LOW_FPS = 60.0
SIDE_VIEW_MAX_RATIO = 0.35  # (양 어깨 x 간격 ÷ 몸통 길이) 중앙값 상한 — 넘으면 후면·정면 촬영
REAR_VIEW_MIN_RATIO = 0.45  # 후면 분석일 때 이보다 작으면 사선·측면 촬영 경고
OFFSCREEN_MAX_FRAC = 0.20
LEG_RATIO = SEGMENT_RATIOS["thigh"] + SEGMENT_RATIOS["shank"]   # 0.491 H
SCALE_EXCLUDE_S = 0.15      # 스케일 계산에서 제외할 킥 직전 구간 (s)

LOW_FPS_MSG = ("킥 임팩트는 약 10 ms로 매우 짧아 60fps 미만에서는 각속도·임팩트 검출이 부정확합니다. "
               "슬로모션 촬영을 권장합니다.")


def swap_sides(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """l_ ↔ r_ 이름을 바꾼다 (왼발 킥을 내부적으로 '오른발 = 차는 다리'로 맞춤)."""
    def sw(k: str) -> str:
        if k.startswith("l_"):
            return "r_" + k[2:]
        if k.startswith("r_"):
            return "l_" + k[2:]
        return k
    return {sw(k): v for k, v in d.items()}


def to_body_frame(xy_px: dict[str, np.ndarray], flip_x: bool, px_per_m: float = 1.0
                  ) -> dict[str, np.ndarray]:
    """이미지 좌표(y 아래) → 몸 기준 좌표(위 = +y, 차는 방향 = +x), 단위 변환 포함."""
    sx = -1.0 if flip_x else 1.0
    out = {}
    for k, v in xy_px.items():
        v = np.asarray(v, dtype=float)
        out[k] = np.stack([sx * v[..., 0], -v[..., 1]], axis=-1) / px_per_m
    return out


def fill_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """결측(valid=False) 프레임을 선형 보간한다. 앞뒤 끝은 가장 가까운 값으로 채운다."""
    idx = np.arange(len(x))
    if valid.sum() < 2:
        raise ValueError("랜드마크가 거의 검출되지 않았습니다")
    return np.stack([np.interp(idx, idx[valid], x[valid, j]) for j in range(x.shape[1])], axis=1)


def long_gap_mask(valid: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    """max_gap 프레임보다 긴 결측 구간이면 True (보간을 신뢰할 수 없는 프레임)."""
    out = np.zeros(len(valid), dtype=bool)
    i = 0
    while i < len(valid):
        if not valid[i]:
            j = i
            while j < len(valid) and not valid[j]:
                j += 1
            if j - i > max_gap:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def lowpass(x: np.ndarray, fps: float, cutoff_hz: float) -> tuple[np.ndarray, bool]:
    """4차 Butterworth 저역통과를 양방향(filtfilt)으로 적용한다. 적용 못 하면 원본을 돌려준다."""
    nyq = fps / 2
    b, a = butter(FILTER_ORDER, min(cutoff_hz, 0.95 * nyq) / nyq)
    if len(x) <= 3 * max(len(a), len(b)) or cutoff_hz >= 0.95 * nyq:
        return x, False
    return filtfilt(b, a, x, axis=0), True


def speed(p: np.ndarray, fps: float) -> np.ndarray:
    """좌표 시계열의 속력 (단위/s)."""
    return np.linalg.norm(np.gradient(p, 1 / fps, axis=0), axis=1)


def kick_foot(xy: dict[str, np.ndarray], fps: float) -> tuple[str, int]:
    """발끝 속도 최대값이 큰 쪽 = 차는 다리. (side, 최고 속도 프레임)을 반환한다."""
    sr, sl = speed(xy["r_toe"], fps), speed(xy["l_toe"], fps)
    side = "R" if sr.max() >= sl.max() else "L"
    s = sr if side == "R" else sl
    return side, int(np.argmax(s))


def kick_direction_flip(xy_px: dict[str, np.ndarray], i_peak: int, approach: slice) -> bool:
    """공 차는 방향이 이미지 왼쪽이면 True (x 반전 필요).

    접근 구간 디딤발(l_)의 뒤꿈치→발끝 방향(몸이 향하는 방향)으로 판단한다. 지면 킥은 이 방향이
    임팩트 시 차는 발의 이동 방향과 같고, 공을 몸 뒤쪽으로 차는 동작에서도 템플릿 좌표계와
    일치한다. 발 방향이 애매하면 임팩트 전후 차는 발의 x 이동 방향을 쓴다.
    """
    facing = np.median(xy_px["l_toe"][approach, 0] - xy_px["l_heel"][approach, 0])
    foot_len = np.median(np.linalg.norm(xy_px["l_toe"][approach] - xy_px["l_heel"][approach], axis=1))
    if abs(facing) > 0.3 * foot_len:
        return bool(facing < 0)
    foot = (xy_px["r_ankle"] + xy_px["r_toe"]) / 2
    a, b = max(i_peak - 3, 0), min(i_peak + 3, len(foot) - 1)
    return bool(foot[b, 0] - foot[a, 0] < 0)


def ground_level(xy: dict[str, np.ndarray]) -> float:
    """지면 기준선 = 디딤발 뒤꿈치/발끝 y의 하위 5% 값.

    공중 동작처럼 디딤발이 지면에 닿지 않는 영상을 위해 차는 발 쪽 값이 더 낮으면 그 값을 쓴다.
    """
    q = lambda s: np.quantile(np.concatenate([xy[f"{s}_heel"][:, 1], xy[f"{s}_toe"][:, 1]]), 0.05)
    return float(min(q("l"), q("r")))


def shoulder_ratio(raw: PoseSeries) -> float:
    """(양 어깨 x 간격 ÷ 몸통 길이)의 중앙값. 측면이면 작고, 후면·정면이면 크다."""
    xy = raw.xy
    ok = np.all(np.isfinite(xy["l_shoulder"]) & np.isfinite(xy["r_shoulder"]), axis=1)
    if ok.sum() == 0:
        return 0.0
    gap = np.abs(xy["l_shoulder"][ok, 0] - xy["r_shoulder"][ok, 0])
    torso = np.linalg.norm((xy["l_shoulder"][ok] + xy["r_shoulder"][ok]) / 2
                           - (xy["l_hip"][ok] + xy["r_hip"][ok]) / 2, axis=1)
    return float(np.median(gap / np.maximum(torso, 1e-9)))


def detect_view(raw: PoseSeries) -> str:
    """촬영 방향 자동 감지: 'side'(측면) 또는 'rear'(후면·정면 = 관상면)."""
    return "rear" if shoulder_ratio(raw) > SIDE_VIEW_MAX_RATIO else "side"


def preprocess(raw: PoseSeries, H: float, fps: float | None = None,
               cutoff_hz: float = DEFAULT_CUTOFF_HZ, view: str = "side") -> Processed:
    """원시 랜드마크를 몸 기준 좌표(m)·관절 각도 시계열로 바꾼다. fps = 실제 촬영 fps.

    view = "side": x = 공 차는 방향, 각도 = 4.3 시상면 각도.
    view = "rear": x = 차는 다리 쪽(좌우), 각도 = 관상면 각도(`frontal_angles`). "auto"면 자동 감지.
    """
    fps = float(fps or raw.meta_fps)
    if view == "auto":
        view = detect_view(raw)
    warnings: list[tuple[str, str]] = []
    if fps < LOW_FPS:
        warnings.append(("warning", LOW_FPS_MSG))

    # 1) 결측·저신뢰 보간 + 2) 저역통과 필터 (픽셀 좌표에서 수행, 선형이므로 순서 무관)
    xy_px, xy_filled, vis, applied = {}, {}, {}, True
    for k, v in raw.xy.items():
        finite = np.all(np.isfinite(v), axis=1)
        valid = (raw.vis[k] >= VIS_MIN) & finite
        if valid.sum() < 2:     # 계속 저신뢰면 좌표는 그대로 쓰고 visibility로 불확실 표시
            valid = finite
        xy_filled[k] = fill_gaps(v, valid)
        xy_px[k], ok = lowpass(xy_filled[k], fps, cutoff_hz)
        applied &= ok
        uncertain = long_gap_mask(valid)
        vis[k] = np.where(uncertain, np.minimum(raw.vis[k], VIS_MIN - 1e-3), raw.vis[k])

    # 3) 차는 다리 판별 → 내부적으로 r_ = 차는 다리
    side, i_peak = kick_foot(xy_px, fps)
    if side == "L":
        xy_px, xy_filled, vis = swap_sides(xy_px), swap_sides(xy_filled), swap_sides(vis)
        warnings.append(("info", "왼발 킥으로 인식했습니다. 내부적으로 좌우를 바꿔 '차는 다리'로 분석합니다."))

    # 접근 구간 = 킥 직전 SCALE_EXCLUDE_S 이전 프레임들
    end = max(int(i_peak - SCALE_EXCLUDE_S * fps), 0)
    sl = slice(0, end) if end >= 5 else slice(None)

    leg_len = lambda s, idx: (np.linalg.norm(xy_px[f"{s}_knee"][idx] - xy_px[f"{s}_hip"][idx], axis=1)
                              + np.linalg.norm(xy_px[f"{s}_ankle"][idx] - xy_px[f"{s}_knee"][idx], axis=1))
    if view == "rear":
        # 4) 방향 정규화: 차는 다리 쪽 = +x (뒤에서 찍든 앞에서 찍든 같은 부호), 위 = +y
        flip = bool(np.median(xy_px["r_hip"][:, 0] - xy_px["l_hip"][:, 0]) < 0)
        # 5) 스케일: 뒤에서 보면 무릎을 굽힐 때 다리가 짧아 보이므로 다리가 가장 펴진 쪽(상위 10%)을 쓴다
        px_per_m = float(np.quantile(np.concatenate([leg_len(s, slice(None)) for s in "rl"]), 0.9)) \
            / (LEG_RATIO * H)
    else:
        # 4) 방향 정규화 (공 차는 방향 = +x, 위 = +y)
        flip = kick_direction_flip(xy_px, i_peak, sl)
        # 5) 스케일: 접근 초반 (대퇴+하퇴) 길이 중앙값 ÷ 0.491 H
        px_per_m = float(np.median(np.concatenate([leg_len(s, sl) for s in "rl"]))) / (LEG_RATIO * H)

    xy = to_body_frame(xy_px, flip, px_per_m)
    g = ground_level(xy)
    x0 = float(((xy["l_hip"][0] + xy["r_hip"][0]) / 2)[0])
    xy = {k: v - np.array([x0, g]) for k, v in xy.items()}
    xy_unf = {k: v - np.array([x0, g]) for k, v in to_body_frame(xy_filled, flip, px_per_m).items()}

    # 6) 촬영 각도 품질 검사
    ratio = shoulder_ratio(raw)
    if view == "side" and ratio > SIDE_VIEW_MAX_RATIO:
        warnings.append(("warning", "정면/사선 촬영으로 보입니다. 측면에서 찍어야 정확합니다. "
                                    "후면 영상이라면 촬영 방향을 '후면'으로 선택하세요."))
    if view == "rear" and ratio < REAR_VIEW_MIN_RATIO:
        warnings.append(("warning", "양 어깨가 거의 겹쳐 보여 측면(또는 사선) 촬영으로 보입니다. "
                                    "후면 분석은 선수 바로 뒤에서 찍어야 정확합니다."))
    out = np.zeros(raw.n, dtype=bool)
    for v in raw.xy.values():
        out |= (v[:, 0] < 0) | (v[:, 0] > raw.width) | (v[:, 1] < 0) | (v[:, 1] > raw.height)
    if out.mean() > OFFSCREEN_MAX_FRAC:
        warnings.append(("warning", f"몸이 화면 밖으로 나간 프레임이 {out.mean():.0%}입니다. "
                                    "전신과 공이 모두 보이게 촬영하세요."))
    if not applied:
        warnings.append(("info", "프레임 수가 너무 적거나 fps가 낮아 저역통과 필터를 생략했습니다."))

    t = np.arange(raw.n) / fps
    angles = frontal_angles(xy) if view == "rear" else angles_from_pose(xy)
    return Processed(t=t, xy=xy, vis=vis, angles=angles, fps=fps, H=H,
                     kick_side=side, flipped_x=flip, px_per_m=px_per_m,
                     cutoff_hz=cutoff_hz if applied else None, xy_px=xy_px, xy_unfiltered=xy_unf,
                     warnings=warnings, view=view)
