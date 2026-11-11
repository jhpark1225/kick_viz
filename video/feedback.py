"""수정점 피드백: 지표 계산 → 규칙(data/feedback_rules.json) 판정 → 우선순위."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video.events import (Events, kick_foot_point, light_filtered, shank_angular_velocity,
                          thigh_angular_velocity)
from video.pose_series import Processed
from video.preprocess import LOW_FPS, VIS_MIN, speed

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback_rules.json"
FOOT_SPEED_WINDOW = 0.03    # 임팩트 직전 발 속도를 찾는 구간 (s)
SEQ_TOL_FRAMES = 1          # 정강이 최고점 ≤ 임팩트 판정 허용 (프레임, 임팩트 검출 오차)
SEQUENCE_SCORE = 100.0      # sequence_ok 거짓이면 항상 Top 3

GOOD, IMPROVE, UNSURE = "✅ 좋음", "⚠️ 개선", "❔ 측정 불확실"

# 지표 → (한국어 이름, 단위)
METRICS: dict[str, tuple[str, str]] = {
    "max_r_knee_flex": ("백스윙 무릎 최대 굴곡", "°"),
    "min_r_hip_ext": ("백스윙 고관절 최대 신전", "°"),
    "impact_r_knee_flex": ("임팩트 시 무릎 굴곡", "°"),
    "impact_r_ankle_pf": ("임팩트 시 발목 저측굴곡", "°"),
    "impact_trunk": ("임팩트 시 몸통 기울기", "°"),
    "impact_l_knee_flex": ("임팩트 시 디딤발 무릎 굴곡", "°"),
    "followthrough_r_hip_flex": ("팔로스루 고관절 최대 굴곡", "°"),
    "sequence_ok": ("근위→원위 순차 (허벅지 → 정강이)", ""),
    "peak_shank_ang_vel": ("정강이 최대 각속도", "°/s"),
    "foot_speed_impact": ("임팩트 직전 발 속도", "m/s"),
    "arm_balance": ("임팩트 시 반대팔 어깨 굴곡", "°"),
    "kick_duration": ("백스윙 최대 → 임팩트 시간", "s"),
    # 후면(관상면) 지표 — video/frontal.py
    "stance_knee_valgus": ("디딤 무릎 안쪽 무너짐(외반)", "°"),
    "trunk_lateral_lean": ("임팩트 시 몸통 옆 기울기 (디딤발 쪽 +)", "°"),
    "pelvic_obliquity": ("임팩트 시 골반 기울기 (차는 쪽 들림 +)", "°"),
    "plant_foot_distance": ("디딤발 ↔ 차는 발 좌우 거리 (공 옆 거리 근사)", "m"),
    "opposite_arm_abduction": ("임팩트 시 디딤발 쪽 팔 벌림", "°"),
}
VELOCITY_METRICS = {"sequence_ok", "peak_shank_ang_vel", "foot_speed_impact"}

SIDE_VIEW_LIMITS = [
    "무릎 안쪽 무너짐(외반)",
    "골반 회전",
    "접근 각도",
    "디딤발과 공 사이 좌우 거리",
]
SIDE_VIEW_NOTE = "측면 영상으로는 판단할 수 없습니다 → 정면/위에서 추가 촬영 필요"
SAFETY_NOTE = ("이 피드백은 교육용이며 의료적 진단이 아닙니다. 통증이 있거나 계속되면 "
               "전문가(의사·물리치료사)와 상담하세요.")


@dataclass
class MetricValue:
    value: float | bool | None
    uncertain: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
def _window_vis(p: Processed, names: list[str], lo: int, hi: int) -> float:
    """구간 [lo, hi]의 랜드마크 평균 visibility 중 최소값."""
    lo, hi = max(lo, 0), min(hi, p.n - 1)
    return min(float(np.mean(p.vis[k][lo:hi + 1])) for k in names)


def compute_metrics(p: Processed, ev: Events) -> dict[str, MetricValue]:
    """12개 지표를 계산한다. 구간 visibility < 0.5 또는 fps < 60인 각속도 지표는 불확실로 표시."""
    out: dict[str, MetricValue] = {k: MetricValue(None, True, "이벤트 검출 실패") for k in METRICS}
    imp = ev.impact
    if imp is None:
        return out
    fps, A = p.fps, p.angles
    plant = ev.frames.get("디딤")
    start = plant if plant is not None else max(imp - int(0.3 * fps), 0)
    bs, ft = ev.frames.get("백스윙"), ev.frames.get("팔로스루")
    leg = ["r_hip", "r_knee", "r_ankle"]
    trunk_lm = ["l_shoulder", "r_shoulder", "l_hip", "r_hip"]
    k2 = max(int(round(0.02 * fps)), 1)   # 임팩트 순간 지표의 visibility 확인 구간 (±20 ms)

    def put(name: str, value, names: list[str], lo: int, hi: int) -> None:
        vis = _window_vis(p, names, lo, hi)
        reasons = []
        if vis < VIS_MIN:
            reasons.append(f"해당 구간 visibility {vis:.2f} < {VIS_MIN}")
        if name in VELOCITY_METRICS and fps < LOW_FPS:
            reasons.append(f"촬영 fps {fps:.0f} < {LOW_FPS:.0f} (각속도·속도 지표)")
        out[name] = MetricValue(value, bool(reasons), "; ".join(reasons))

    seg = slice(start, imp + 1)
    put("max_r_knee_flex", float(np.max(A["r_knee"][seg])), leg, start, imp)
    put("min_r_hip_ext", float(np.min(A["r_hip"][seg])), leg + trunk_lm, start, imp)
    put("impact_r_knee_flex", float(A["r_knee"][imp]), leg, imp - k2, imp + k2)
    put("impact_r_ankle_pf", float(A["r_ankle"][imp]), ["r_knee", "r_ankle", "r_heel", "r_toe"], imp - k2, imp + k2)
    put("impact_trunk", float(A["trunk"][imp]), trunk_lm, imp - k2, imp + k2)
    put("impact_l_knee_flex", float(A["l_knee"][imp]), ["l_hip", "l_knee", "l_ankle"], imp - k2, imp + k2)
    if ft is not None:
        put("followthrough_r_hip_flex", float(A["r_hip"][ft]), leg + trunk_lm, imp, ft)

    # 순서 판정은 노이즈에 강한 8 Hz 데이터, 최고 속도 크기는 덜 깎이는 25 Hz 데이터로 계산
    hi = min(imp + SEQ_TOL_FRAMES, p.n - 1)
    w_th, w_sh = thigh_angular_velocity(p), shank_angular_velocity(p)
    i_th = start + int(np.argmax(w_th[start:hi + 1]))
    i_sh = start + int(np.argmax(w_sh[start:hi + 1]))
    put("sequence_ok", bool(i_th < i_sh <= imp + SEQ_TOL_FRAMES), leg, start, imp)
    w_sh_light = shank_angular_velocity(p, light=True)
    put("peak_shank_ang_vel", float(np.max(w_sh_light[start:hi + 1])), leg, start, imp)

    lo = max(imp - int(round(FOOT_SPEED_WINDOW * fps)), 0)
    s = speed(kick_foot_point(light_filtered(p, ["r_ankle", "r_toe"])), fps)
    put("foot_speed_impact", float(np.max(s[lo:imp + 1])), ["r_ankle", "r_toe"], lo, imp)
    put("arm_balance", float(A["l_shoulder"][imp]), ["l_shoulder", "l_elbow"], imp - k2, imp + k2)
    if bs is not None:
        put("kick_duration", float((imp - bs) / fps), leg, bs, imp)
    return out


# ---------------------------------------------------------------------------
# 규칙 판정
# ---------------------------------------------------------------------------
def load_rules(path: Path = RULES_PATH) -> dict[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Judged:
    rule: dict
    value: float | bool | None
    status: str
    score: float
    message: str
    reason: str = ""

    @property
    def name(self) -> str:
        return METRICS[self.rule["metric"]][0]

    @property
    def unit(self) -> str:
        return self.rule.get("unit", "")


def format_value(v, unit: str) -> str:
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if unit == "s":
        return f"{v:.3f} s"
    if unit == "m/s":
        return f"{v:.1f} m/s"
    if unit == "m":
        return f"{v:.2f} m"
    return f"{v:.0f}{unit}"


def format_range(rule: dict) -> str:
    if "expected" in rule:
        return "예" if rule["expected"] else "아니오"
    lo, hi = rule["good_range"]
    u = rule.get("unit", "")
    if lo is None:
        return f"{format_value(hi, u)} 이하"
    if hi is None:
        return f"{format_value(lo, u)} 이상"
    return f"{format_value(lo, u)} ~ {format_value(hi, u)}"


def judge(rule: dict, mv: MetricValue) -> Judged:
    """한 규칙을 판정한다. 점수 = 범위를 벗어난 정도(허용폭 단위) × weight."""
    v = mv.value
    if v is None or mv.uncertain:
        return Judged(rule, v, UNSURE, 0.0, "", mv.reason or "측정할 수 없음")
    if "expected" in rule:
        if bool(v) == rule["expected"]:
            return Judged(rule, v, GOOD, 0.0, "")
        return Judged(rule, v, IMPROVE, SEQUENCE_SCORE * rule["weight"], rule["fail_msg"])
    lo, hi = rule["good_range"]
    tol = rule.get("tol") or ((hi - lo) / 2 if lo is not None and hi is not None else 1.0)
    if lo is not None and v < lo:
        return Judged(rule, v, IMPROVE, (lo - v) / tol * rule["weight"], rule["low_msg"])
    if hi is not None and v > hi:
        return Judged(rule, v, IMPROVE, (v - hi) / tol * rule["weight"], rule["high_msg"])
    return Judged(rule, v, GOOD, 0.0, "")


@dataclass
class Feedback:
    kick_id: str
    items: list[Judged]

    @property
    def top3(self) -> list[Judged]:
        """우선 개선점: 점수순 상위 3개 (sequence_ok 거짓은 점수가 커서 항상 포함)."""
        bad = sorted((j for j in self.items if j.status == IMPROVE), key=lambda j: -j.score)
        return bad[:3]

    @property
    def strengths(self) -> list[Judged]:
        """잘한 점: 범위 안 + weight 높은 항목 1~2개."""
        good = sorted((j for j in self.items if j.status == GOOD), key=lambda j: -j.rule["weight"])
        return good[:2]

    @property
    def n_improve(self) -> int:
        return sum(j.status == IMPROVE for j in self.items)


def make_feedback(kick_id: str, metrics: dict[str, MetricValue], rules: dict | list | None = None) -> Feedback:
    """선택된 킥의 규칙으로 모든 지표를 판정한다. rules가 리스트면 그 규칙 목록을 그대로 쓴다."""
    items = rules if isinstance(rules, list) else (rules or load_rules())[kick_id]
    return Feedback(kick_id, [judge(r, metrics[r["metric"]]) for r in items])


def split_message(msg: str) -> tuple[str, str]:
    """'무엇을 고칠지. 왜(생체역학).' → (무엇, 왜)."""
    head, _, tail = msg.partition(". ")
    return (head + ("." if tail else ""), tail)
