"""원시 랜드마크 정리: 좌우 다리 이름 뒤바뀜 보정, 순간 튐(스파이크) 제거, 분절 길이 점검.

측면 영상에서 MediaPipe는 가까운 다리와 먼 다리가 겹칠 때 좌우 이름을 몇 프레임씩 바꿔 붙이는 일이 잦다.
이 상태로 저역통과 필터를 걸면 두 다리 좌표가 섞여 각도가 크게 틀어지고, 발끝 속도가 가짜로 커져
차는 다리·임팩트 검출까지 틀린다. 그래서 필터 전에 원시 좌표에서 먼저 바로잡는다.
"""

from dataclasses import dataclass, replace

import numpy as np

from video.pose_series import PoseSeries

# 좌우가 함께 바뀌는 묶음 (같은 묶음 안의 점은 같이 바꾼다).
# 양쪽 고관절은 측면에서 거의 같은 위치에 겹쳐 보여, 잡음만으로도 바뀜 판정이 오락가락하므로 넣지 않는다
# (바뀌어도 위치가 거의 같아 각도 영향이 작다).
SWAP_GROUPS = {
    "다리": ["knee", "ankle", "heel", "toe"],
    "팔": ["shoulder", "elbow", "wrist"],
}
SWAP_PRIOR = 0.08           # MediaPipe 원래 이름을 유지하는 쪽에 주는 비용 (몸통 길이 비율, 점 하나당)
SPIKE_MIN = 0.10            # 스파이크 판정 최소 이탈 거리 (몸통 길이 비율)
SPIKE_RETURN = 0.5          # 튄 뒤 제자리로 돌아오는 정도 (앞뒤 프레임 간격 < 이 비율 × 튄 거리)
SEG_DEV_MAX = 0.40          # 대퇴·하퇴 겉보기 길이가 중앙값에서 이 비율 넘게 벗어나면 비정상 (품질 지표로만 보고)


@dataclass
class CleanReport:
    """정리 결과 요약 (화면 안내·성능 평가용)."""
    swapped_frames: dict[str, int]      # 묶음별 좌우를 바로잡은 프레임 수
    spikes: int                         # 제거한 스파이크 점 개수
    seg_outlier_frac: float | None      # 분절 길이가 비정상인 프레임 비율 (측면만, 후면은 None)

    @property
    def total_swapped(self) -> int:
        return int(sum(self.swapped_frames.values()))


def torso_length(xy: dict[str, np.ndarray]) -> float:
    """몸통 길이(어깨 중점↔고관절 중점) 중앙값 (픽셀). 거리 기준값으로 쓴다."""
    sh = (xy["l_shoulder"] + xy["r_shoulder"]) / 2
    hp = (xy["l_hip"] + xy["r_hip"]) / 2
    v = np.linalg.norm(sh - hp, axis=1)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else 1.0


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """NaN 프레임을 앞뒤 값으로 선형 보간 (보정 판단용 임시 좌표)."""
    out = a.copy()
    idx = np.arange(len(a))
    ok = np.all(np.isfinite(a.reshape(len(a), -1)), axis=1)
    if ok.sum() < 2:
        return np.nan_to_num(out)
    flat = out.reshape(len(a), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(idx, idx[ok], flat[ok, j])
    return flat.reshape(a.shape)


def swap_states(L: np.ndarray, R: np.ndarray, prior: float) -> np.ndarray:
    """프레임별 좌우 바뀜 여부(True = 바꿔야 함)를 2차 비터비로 찾는다 (선운동학: 등속 예측).

    L, R: (n, k, 2). 상태 = (이전 프레임 바뀜, 지금 프레임 바뀜). 비용 = 각 점이
    등속 예측 위치(2·p[t−1] − p[t−2])에서 벗어난 거리 + 원래 이름을 바꾸는 비용(prior).
    두 다리가 교차하는 순간 1차(직전 위치) 비용은 '튕겨 나가는' 잘못된 바꿈을 고르므로 속도까지 본다.
    """
    n = len(L)
    if n < 3:
        return np.zeros(n, dtype=bool)
    P = np.stack([np.stack([L, R], 1), np.stack([R, L], 1)])     # (2 상태, n, 2 측, k, 2)
    k = L.shape[1]
    INF = np.inf
    # cost[a, b]: 프레임 t−1 상태 a, t 상태 b까지의 최소 비용 (t = 1에서 시작)
    cost = np.array([[prior * k * (a + b) for b in (0, 1)] for a in (0, 1)], dtype=float)
    back = np.zeros((n, 2, 2), dtype=int)                       # back[t, b, c] = t−2 상태... 저장용
    for t in range(2, n):
        new = np.full((2, 2), INF)
        for b in (0, 1):            # t−1 상태
            for c in (0, 1):        # t 상태
                best, arg = INF, 0
                for a in (0, 1):    # t−2 상태
                    pred = 2 * P[b, t - 1] - P[a, t - 2]
                    d = float(np.linalg.norm(P[c, t] - pred, axis=-1).sum())
                    v = cost[a, b] + d
                    if v < best:
                        best, arg = v, a
                new[b, c] = best + prior * k * c
                back[t, b, c] = arg
        cost = new
    b, c = np.unravel_index(int(np.argmin(cost)), cost.shape)
    states = np.zeros(n, dtype=int)
    states[n - 1], states[n - 2] = c, b
    for t in range(n - 1, 1, -1):
        a = back[t, states[t - 1], states[t]]
        states[t - 2] = a
    return states.astype(bool)


def fix_swaps(raw: PoseSeries, prior: float = SWAP_PRIOR) -> tuple[PoseSeries, dict[str, int]]:
    """묶음(다리·팔)별로 좌우 이름이 뒤바뀐 프레임을 바로잡는다."""
    scale = torso_length(raw.xy)
    xy = {k: v.copy() for k, v in raw.xy.items()}
    vis = {k: v.copy() for k, v in raw.vis.items()}
    counts = {}
    for group, names in SWAP_GROUPS.items():
        names = [m for m in names if f"l_{m}" in xy and f"r_{m}" in xy]
        if not names:
            continue
        L = _fill_nan(np.stack([xy[f"l_{m}"] for m in names], 1)) / scale
        R = _fill_nan(np.stack([xy[f"r_{m}"] for m in names], 1)) / scale
        s = swap_states(L, R, prior)
        if s.mean() > 0.5:          # 전체를 통째로 바꾸는 해는 원래 이름을 기준으로 되돌린다
            s = ~s
        counts[group] = int(s.sum())
        for m in names:
            for d in (xy, vis):
                lv, rv = d[f"l_{m}"].copy(), d[f"r_{m}"].copy()
                d[f"l_{m}"][s], d[f"r_{m}"][s] = rv[s], lv[s]
    return replace(raw, xy=xy, vis=vis), counts


def spike_mask(p: np.ndarray, min_dev: float) -> np.ndarray:
    """1~2프레임 동안 튀었다가 제자리로 돌아오는 점(True)을 찾는다.

    빠른 킥 동작(발이 크게 이동)은 앞뒤 프레임 간격도 크므로 걸리지 않는다:
    '튄 거리'는 크고 '앞뒤 프레임 사이 간격'은 작을 때만 스파이크로 본다.
    """
    n = len(p)
    out = np.zeros(n, dtype=bool)
    for w in (1, 2):                # 1프레임·2프레임 스파이크
        for i in range(1, n - w):
            a, b = p[i - 1], p[i + w]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            mid = p[i:i + w]
            if not np.all(np.isfinite(mid)):
                continue
            dev = np.linalg.norm(mid - (a + b) / 2, axis=1).min()
            gap = np.linalg.norm(b - a)
            if dev > min_dev and gap < SPIKE_RETURN * dev:
                out[i:i + w] = True
    # 영상 처음·끝 프레임: 이웃이 한쪽뿐이므로 다음 두 프레임이 서로 가까운데 첫 프레임만 멀면 스파이크
    for i, j, k in ((0, 1, 2), (n - 1, n - 2, n - 3)):
        if n >= 3 and np.all(np.isfinite(p[[i, j, k]])):
            if np.linalg.norm(p[i] - p[j]) > min_dev and \
                    np.linalg.norm(p[j] - p[k]) < SPIKE_RETURN * np.linalg.norm(p[i] - p[j]):
                out[i] = True
    return out


def remove_spikes(raw: PoseSeries) -> tuple[PoseSeries, int]:
    """스파이크 점의 visibility를 0으로 내려 이후 단계에서 결측으로 보간되게 한다."""
    min_dev = SPIKE_MIN * torso_length(raw.xy)
    vis = {k: v.copy() for k, v in raw.vis.items()}
    total = 0
    for k, v in raw.xy.items():
        m = spike_mask(v, min_dev)
        vis[k][m] = 0.0
        total += int(m.sum())
    return replace(raw, vis=vis), total


def segment_outliers(raw: PoseSeries) -> np.ndarray:
    """대퇴·하퇴 겉보기 길이가 중앙값에서 SEG_DEV_MAX 넘게 벗어난 프레임 (True)."""
    bad = np.zeros(raw.n, dtype=bool)
    for s in "lr":
        for a, b in (("hip", "knee"), ("knee", "ankle")):
            ln = np.linalg.norm(raw.xy[f"{s}_{b}"] - raw.xy[f"{s}_{a}"], axis=1)
            med = np.nanmedian(ln)
            if not np.isfinite(med) or med <= 0:
                continue
            with np.errstate(invalid="ignore"):
                bad |= np.abs(ln / med - 1) > SEG_DEV_MAX
    return bad


def clean_pose(raw: PoseSeries, view: str = "side") -> tuple[PoseSeries, CleanReport]:
    """좌우 뒤바뀜 보정 → 스파이크 제거 → (측면) 분절 길이 점검.

    분절 길이 점검은 좌표를 바꾸지 않고 품질 지표로만 보고한다. 인사이드 킥처럼 다리가 시상면 밖으로
    돌면 측면에서도 겉보기 길이가 실제로 짧아지고, 후면에서는 무릎을 굽힐 때마다 짧아 보이기 때문이다.
    """
    raw2, counts = fix_swaps(raw)
    raw3, spikes = remove_spikes(raw2)
    seg = float(segment_outliers(raw3).mean()) if view == "side" else None
    return raw3, CleanReport(swapped_frames=counts, spikes=spikes, seg_outlier_frac=seg)
