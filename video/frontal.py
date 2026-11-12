"""후면(관상면) 영상 지표 — 측면에서 안 보이는 무릎 외반·골반 기울기·몸통 옆 기울기·디딤발 위치·팔 벌림."""

from pathlib import Path

import numpy as np

from video.events import Events, kick_foot_point
from video.feedback import MetricValue
from video.pose_series import Processed
from video.preprocess import VIS_MIN

RULES_REAR_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback_rules_rear.json"
VALGUS_AFTER_IMPACT = 0.05      # 디딤 무릎 외반 최대값 탐색: 디딤 ~ 임팩트 + 0.05 s

REAR_NOT_MEASURED = [
    "무릎 굽힘·펴기 각도, 발목 각도",
    "몸통 앞뒤 기울기",
    "허벅지 → 정강이 순차·각속도, 백스윙 크기",
]
REAR_NOT_MEASURED_NOTE = "후면 영상으로는 판단할 수 없습니다 → 측면 영상을 함께 올려 주세요"
REAR_LIMIT_NOTE = ("후면 분석은 다리를 앞뒤로 휘두르는 동작(시상면)이 화면에 거의 보이지 않아 동작 모양 비교(참고)를 "
                   "하지 않고, 고른 킥의 관상면 기준으로만 피드백합니다. 임팩트 시각은 '스윙 아크의 최저점'으로 찾으므로 "
                   "지면 킥에서 약 ±30 ms, 공중의 공(발리)에서는 더 큰 오차가 있을 수 있습니다.")


def compute_frontal_metrics(p: Processed, ev: Events) -> dict[str, MetricValue]:
    """후면 지표 5개. 구간 평균 visibility < 0.5면 측정 불확실."""
    names = ["stance_knee_valgus", "trunk_lateral_lean", "pelvic_obliquity", "plant_foot_distance",
             "opposite_arm_abduction"]
    out = {k: MetricValue(None, True, "이벤트 검출 실패") for k in names}
    imp = ev.impact
    if imp is None:
        return out
    A, fps = p.angles, p.fps
    plant = ev.frames.get("디딤")
    start = plant if plant is not None else max(imp - int(0.15 * fps), 0)
    stop = min(imp + int(round(VALGUS_AFTER_IMPACT * fps)), p.n - 1)
    k2 = max(int(round(0.02 * fps)), 1)

    def put(name: str, value, lms: list[str], lo: int, hi: int) -> None:
        lo, hi = max(lo, 0), min(hi, p.n - 1)
        vis = min(float(np.mean(p.vis[k][lo:hi + 1])) for k in lms)
        bad = vis < VIS_MIN
        out[name] = MetricValue(value, bad, f"해당 구간 visibility {vis:.2f} < {VIS_MIN}" if bad else "")

    put("stance_knee_valgus", float(np.max(A["stance_knee_valgus"][start:stop + 1])),
        ["l_hip", "l_knee", "l_ankle"], start, stop)
    put("trunk_lateral_lean", float(A["trunk_lateral_lean"][imp]),
        ["l_shoulder", "r_shoulder", "l_hip", "r_hip"], imp - k2, imp + k2)
    put("pelvic_obliquity", float(A["pelvic_obliquity"][imp]), ["l_hip", "r_hip"], imp - k2, imp + k2)
    dist = abs(float(p.xy["l_ankle"][imp, 0] - kick_foot_point(p.xy)[imp, 0]))
    put("plant_foot_distance", dist, ["l_ankle", "r_ankle", "r_toe"], imp - k2, imp + k2)
    put("opposite_arm_abduction", float(A["arm_abduction"][imp]), ["l_shoulder", "l_wrist"], imp - k2, imp + k2)
    return out


def rear_rules(kick_id: str) -> list[dict]:
    """킥별 후면 규칙 (없으면 공통 규칙)."""
    from video.feedback import load_rules
    rules = load_rules(RULES_REAR_PATH)
    return rules.get(kick_id, rules["default"])
