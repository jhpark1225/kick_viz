"""킥 자동 분류 (참고용): ① 규칙 게이트로 후보를 좁히고 ② 템플릿 DTW 유사도로 순위를 매긴다.

앱에서 피드백 기준 킥은 항상 사용자가 고른다. 분류 결과는 '동작 모양 비교'로만 보여 주고,
선택한 킥과 뚜렷이 다를 때 다시 확인하라는 안내(mismatch_note)에만 쓴다.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from biomech.kick_loader import available_kicks, load_kick
from video.events import Events, detect_events, kick_foot_point
from video.pose_series import Processed
from video.preprocess import DEFAULT_CUTOFF_HZ, preprocess
from video.reference import MIN_REFERENCE, Example, load_reference
from video.synthetic import synthesize

WINDOW = (-0.25, 0.20)      # 임팩트 기준 비교 구간 (s)
RESAMPLE_HZ = 100.0
# 발목(r_ankle)은 뺀다: 임팩트 순간 발이 블러로 뭉개져 MediaPipe 뒤꿈치·발끝이 부정확하고, 실제 영상에서
# 모든 킥이 템플릿보다 20~50° 작게 측정돼(인스텝 5~11° vs 31°) 발목을 세운 킥(인사이드·로빙)으로 쏠렸다.
FEATURES = ["r_hip", "r_knee", "l_knee", "trunk", "height"]
TEMPLATE_FPS = 120.0
TEMPERATURE = 0.50          # softmax 온도 (DTW 거리 단위). 0.10이면 실제 영상에서 틀린 답에도 90~100%가 붙었다
                            # (2026-11-12, 킥모음 측면 12개 중 1위 정답 1개). 0.50에서도 합성 템플릿은 모두 1위로 맞힌다.
GATE_PENALTY = 1.0          # 게이트 밖 킥에 더하는 거리 (사실상 순위 뒤로)
GROUP_GAP = 0.15            # 같은 그룹 상위 후보의 신뢰도 차이가 이보다 작으면 그룹으로 표시
LOW_CONFIDENCE = 0.5        # 1위 신뢰도가 이보다 낮으면 단정하지 않음
INSIDE_FOOT_RATIO = 0.6     # 뒤꿈치→발끝 겉보기 길이 ÷ 0.15 H 가 이보다 작으면 인사이드 가중
INSIDE_BONUS = 0.85         # 인사이드 거리 배율

GROUND_KICKS = ["instep", "inside", "outside", "curve", "chip", "toe", "half_volley"]
INSTEP_FAMILY = ["instep", "curve"]
GROUPS: dict[str, list[str]] = {
    "인스텝 계열(인스텝/감아차기)": INSTEP_FAMILY,
    "사이드 계열(인사이드/아웃사이드)": ["inside", "outside"],
}
GROUP_NOTE = "공의 회전·발 안쪽/바깥쪽 사용 여부는 측면 영상 포즈만으로 구분하기 어렵습니다."


@dataclass
class Classification:
    """분류 결과. ranking은 (킥 id, 신뢰도 0~1) 내림차순."""
    ranking: list[tuple[str, float]]
    gate: str
    candidates: list[str]
    evidence: list[str]
    group: str | None = None
    group_members: list[str] = field(default_factory=list)
    distances: dict[str, float] = field(default_factory=dict)
    method: str = "내장 애니메이션과 비교"

    @property
    def top3(self) -> list[tuple[str, float]]:
        return self.ranking[:3]

    @property
    def best(self) -> str:
        return self.ranking[0][0]

    @property
    def uncertain(self) -> bool:
        """하나로 단정하기 어려운지 (그룹 표시이거나 1위 신뢰도가 낮음)."""
        return self.group is not None or self.ranking[0][1] < LOW_CONFIDENCE


# ---------------------------------------------------------------------------
# 특징 시계열
# ---------------------------------------------------------------------------
def feature_matrix(p: Processed, t_impact: float) -> np.ndarray:
    """임팩트 기준 WINDOW 구간을 RESAMPLE_HZ로 다시 샘플링한 특징 행렬 (프레임 × 특징)."""
    t_rel = p.t - t_impact
    lo, hi = max(WINDOW[0], t_rel[0]), min(WINDOW[1], t_rel[-1])
    grid = np.arange(np.ceil(lo * RESAMPLE_HZ), np.floor(hi * RESAMPLE_HZ) + 1) / RESAMPLE_HZ
    cols = [p.angles[k] for k in FEATURES[:-1]] + [kick_foot_point(p.xy)[:, 1]]
    m = np.stack([np.interp(grid, t_rel, c) for c in cols], axis=1)
    # 각도는 구간 평균을 빼 '시간에 따른 모양'만 비교한다. 카메라 각도·MediaPipe 편향으로 실제 영상의 각도가
    # 킥 종류와 상관없이 일정하게 어긋나기 때문이다 (2026-11-12 실제 측면 12개: 상위 3개 정답 3 → 7개).
    # 접촉 높이(마지막 열)는 공의 높이를 구분하는 정보라 절대값을 유지한다.
    m[:, :-1] -= m[:, :-1].mean(axis=0)
    return m


@lru_cache(maxsize=64)
def template_processed(kick_id: str, H: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ,
                       view: str = "side") -> Processed:
    """템플릿을 합성 → 같은 전처리를 거친 '영상으로 본' 템플릿 (view = 'side' 또는 'rear')."""
    kick = load_kick(kick_id)
    return preprocess(synthesize(kick, H, TEMPLATE_FPS, view=view), H, TEMPLATE_FPS, cutoff_hz, view=view)


@lru_cache(maxsize=64)
def template_features(kick_id: str, H: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ) -> np.ndarray:
    kick = load_kick(kick_id)
    t_imp = kick["impact"]["time"] - kick["keyframes"][0]["t"]
    return feature_matrix(template_processed(kick_id, H, cutoff_hz), t_imp)


@lru_cache(maxsize=16)
def feature_scales(H: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ) -> np.ndarray:
    """표준화 기준: 11개 템플릿을 모은 특징별 표준편차."""
    allf = np.concatenate([template_features(k, H, cutoff_hz) for k in available_kicks()])
    return allf.std(axis=0) + 1e-9


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """다변량 DTW 거리 (유클리드 비용, 경로 길이로 정규화)."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = ((D[i - 1, j - 1], L[i - 1, j - 1]), (D[i - 1, j], L[i - 1, j]), (D[i, j - 1], L[i, j - 1]))
            d, l = min(prev, key=lambda x: x[0])
            D[i, j] = d + cost[i - 1, j - 1]
            L[i, j] = l + 1
    return float(D[n, m] / L[n, m])


# ---------------------------------------------------------------------------
# 규칙 게이트
# ---------------------------------------------------------------------------
def gate(p: Processed, ev: Events) -> tuple[str, list[str], list[str]]:
    """규칙 게이트: (게이트 이름, 후보 킥, 근거 문장)."""
    i = ev.impact
    trunk = float(p.angles["trunk"][i])
    h = float(ev.contact_height)
    mkf = ev.frames.get("가속")          # 무릎 최대 굴곡 프레임
    if h > 0.30:
        return "공중 공", ["volley"], [f"임팩트 접촉 높이 {h:.2f} m → 공중의 공을 차는 발리 킥"]
    if 0.13 <= h <= 0.30 and trunk > 0:
        return "바운드 공", ["half_volley", *INSTEP_FAMILY], [
            f"접촉 높이 {h:.2f} m, 임팩트 시 몸통 +{trunk:.0f}° (앞으로 숙여 공을 덮음) → 하프 발리 또는 인스텝 계열"]
    if mkf is not None:
        knee_max = float(p.angles["r_knee"][mkf])
        dt = (i - mkf) / p.fps
        if knee_max < 60 and dt < 0.08:
            return "짧은 백스윙", ["toe", "inside", "chip"], [
                f"무릎 최대 굴곡 {knee_max:.0f}°, 무릎 최대 굴곡→임팩트 {dt * 1000:.0f} ms → 짧고 빠른 킥"]
    return "지면 킥", list(GROUND_KICKS), [f"임팩트 접촉 높이 {h:.2f} m → 지면에 있는 공"]


def detail_evidence(p: Processed, ev: Events) -> list[str]:
    """발목·무릎 특징으로 보충 근거를 만든다."""
    out = []
    mkf = ev.frames.get("가속")
    if mkf is not None:
        knee = float(p.angles["r_knee"][mkf])
        out.append(f"무릎 최대 굴곡 {knee:.0f}° → " + ("큰 백스윙(강한 슈팅)" if knee >= 90 else "작은 백스윙"))
    return out


def foot_length_ratio(p: Processed, impact: int) -> float:
    """임팩트 전후 차는 발 뒤꿈치→발끝 겉보기 길이 ÷ 0.15 H (고관절 외회전 시 측면에서 짧아 보임)."""
    lo, hi = max(impact - int(0.05 * p.fps), 0), impact + int(0.05 * p.fps) + 1
    L = np.linalg.norm(p.xy["r_toe"][lo:hi] - p.xy["r_heel"][lo:hi], axis=1)
    return float(np.median(L) / (0.15 * p.H))


# ---------------------------------------------------------------------------
# 분류
# ---------------------------------------------------------------------------
def softmax_conf(dist: dict[str, float], temperature: float = TEMPERATURE) -> list[tuple[str, float]]:
    ids = list(dist)
    d = np.array([dist[k] for k in ids])
    w = np.exp(-(d - d.min()) / temperature)
    conf = w / w.sum()
    return sorted(zip(ids, conf.tolist()), key=lambda x: -x[1])


def find_group(ranking: list[tuple[str, float]]) -> tuple[str | None, list[str]]:
    """상위 후보들이 같은 구분 불가 그룹이고 신뢰도 차이 < 15%p면 그 그룹."""
    (k1, c1), (k2, c2) = ranking[0], ranking[1]
    for name, members in GROUPS.items():
        if k1 in members and k2 in members and c1 - c2 < GROUP_GAP:
            return name, members
    return None, []


def classify(p: Processed, ev: Events | None = None, reference: list[Example] | None = None) -> Classification:
    """사용자 동작을 분류한다.

    참조 데이터(실제 영상 특징, video/reference.py)가 MIN_REFERENCE개 이상이면 킥마다 가장 가까운 실제 예시와의
    DTW 거리(1-최근접 이웃)를 쓴다. 참조 예시가 없는 킥과 참조가 부족할 때는 내장 애니메이션(템플릿)과 비교한다.
    reference=None이면 data/reference_kicks.json을 읽고, []이면 템플릿만 쓴다.
    """
    if p.view != "side":
        raise ValueError("킥 자동 분류는 측면 영상에서만 할 수 있습니다")
    ev = ev or detect_events(p)
    if ev.impact is None:
        raise ValueError("임팩트를 찾지 못해 분류할 수 없습니다")
    gate_name, cands, evidence = gate(p, ev)
    evidence += detail_evidence(p, ev)
    cutoff = p.cutoff_hz or DEFAULT_CUTOFF_HZ
    scale = feature_scales(p.H, cutoff)
    user = feature_matrix(p, p.t[ev.impact]) / scale
    ref = load_reference() if reference is None else reference
    by_label: dict[str, list[Example]] = {}
    for e in ref:
        by_label.setdefault(e.label, []).append(e)
    use_ref = len(ref) >= MIN_REFERENCE
    side_foot = not use_ref and foot_length_ratio(p, ev.impact) < INSIDE_FOOT_RATIO
    dist, from_ref = {}, []
    for k in available_kicks():
        if use_ref and by_label.get(k):
            d = min(dtw_distance(user, e.seq / scale) for e in by_label[k])
            from_ref.append(k)
        else:
            d = dtw_distance(user, template_features(k, p.H, cutoff) / scale)
            if side_foot and k == "inside":
                d *= INSIDE_BONUS
        dist[k] = d + (0.0 if k in cands else GATE_PENALTY)
    if side_foot:
        evidence.append("측면에서 발이 짧아 보임(고관절 외회전 추정) → 인사이드 쪽 가중")
    if use_ref:
        method = f"실제 영상 {len(ref)}개와 비교" + ("" if len(from_ref) == len(dist) else
                                                 f" (참조 영상 없는 킥 {len(dist) - len(from_ref)}종은 내장 애니메이션)")
    else:
        method = "내장 애니메이션과 비교"
    ranking = softmax_conf(dist)
    group, members = find_group(ranking)
    return Classification(ranking=ranking, gate=gate_name, candidates=cands, evidence=evidence[:3],
                          group=group, group_members=members, distances=dist, method=method)


def same_group(a: str, b: str) -> bool:
    """두 킥이 측면 2D로 구분하기 어려운 같은 그룹(GROUPS)인지."""
    return any(a in m and b in m for m in GROUPS.values())


def mismatch_note(selected: str, c: Classification) -> str | None:
    """사용자가 고른 킥과 동작 모양(자동 분류 1위)이 뚜렷이 다를 때만 다시 확인하라는 안내 문구.

    같은 킥이거나, 같은 그룹(인스텝/감아차기, 인사이드/아웃사이드 — 측면으로 원래 구분 어려움)이거나,
    분류 신뢰도가 낮으면(c.uncertain) None. 분류는 참고용이므로 선택을 바꾸지 않고 안내만 한다.
    """
    best = c.best
    if best == selected or c.uncertain or same_group(selected, best):
        return None
    return (f"선택한 킥({load_kick(selected)['name_ko']})과 동작 모양이 {load_kick(best)['name_ko']}에 더 가깝게 "
            "보입니다. 킥 종류를 다시 확인해 보세요. (측면 영상만으로는 발의 어느 부분으로 찼는지 볼 수 없어 참고용입니다)")
