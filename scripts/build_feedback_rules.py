"""data/feedback_rules.json 생성기.

각 킥 템플릿을 합성 영상(120 fps) → 같은 전처리·이벤트·지표 계산을 거쳐 '영상으로 본 템플릿 값'을 구하고,
허용폭(각도 ±15°, 시간 ±30%, 속도 ±25%)으로 1차 기준 범위를 만든다. 인스텝은 선행연구 범위를 우선한다.
생성된 JSON은 직접 고쳐도 된다 (다시 실행하면 덮어씀). 실행: python scripts/build_feedback_rules.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biomech.kick_loader import available_kicks, load_kick  # noqa: E402
from video.events import detect_events  # noqa: E402
from video.feedback import METRICS, RULES_PATH, compute_metrics  # noqa: E402
from video.preprocess import preprocess  # noqa: E402
from video.synthetic import synthesize  # noqa: E402

H, FPS = 1.75, 120.0
TOL_ANGLE, TOL_TIME, TOL_SPEED = 15.0, 0.30, 0.25

# metric → (id, phase, weight, concept, low_msg, high_msg)
BASE: dict[str, tuple] = {
    "max_r_knee_flex": (
        "backswing_knee", "백스윙", 3, "각운동역학: 관성모멘트",
        "백스윙에서 무릎을 더 접으세요. 다리를 접으면 회전축(고관절)에 대한 관성모멘트가 줄어 허벅지를 더 빨리 휘두를 수 있습니다.",
        "무릎을 과도하게 접었습니다. 펴야 할 각도가 커져 임팩트 타이밍이 늦어질 수 있습니다."),
    "min_r_hip_ext": (
        "backswing_hip_ext", "백스윙", 2, "근골격계: 신장-단축 주기 SSC",
        "고관절을 너무 뒤로 젖혔습니다. 허리가 과도하게 젖혀지면 허리·고관절 앞쪽의 부하가 커질 수 있습니다.",
        "백스윙에서 허벅지를 더 뒤로 보내세요. 고관절 굴곡근(장요근·대퇴직근)이 먼저 늘어나야 신장-단축 주기(SSC)로 더 큰 힘을 낼 수 있습니다."),
    "impact_r_knee_flex": (
        "impact_knee", "임팩트", 2, "각운동학",
        "임팩트 때 무릎이 이미 많이 펴져 있습니다. 무릎이 빠르게 펴지는 도중에 공을 맞혀야 정강이 각속도를 공에 전달할 수 있습니다.",
        "임팩트 때 무릎이 많이 굽어 있습니다. 무릎을 펴는 동작이 덜 진행된 채 공을 맞혀 정강이 속도를 다 쓰지 못했을 수 있습니다."),
    "impact_r_ankle_pf": (
        "impact_ankle", "임팩트", 3, "선운동역학: 운동량 전달",
        "임팩트 때 발목을 더 펴서(발끝을 아래로) 단단히 고정하세요. 발목이 고정돼야 충돌 때 발이 덜 꺾여 운동량이 공에 잘 전달됩니다(공/발 속도비 증가).",
        "임팩트 때 발목을 너무 많이 폈습니다. 이 킥에 맞는 발 부위로 공을 맞히려면 발목 각도를 조금 더 세워 고정하세요."),
    "impact_trunk": (
        "impact_trunk", "임팩트", 2, "포물체 운동: 발사각",
        "임팩트 때 몸이 너무 뒤로 젖혀졌습니다. 몸통이 뒤로 기울면 공 아래쪽을 차게 되어 발사각이 커지고 공이 뜹니다.",
        "임팩트 때 몸이 너무 앞으로 숙여졌습니다. 공을 눌러 차게 되어 발사각이 너무 낮아질 수 있습니다."),
    "impact_l_knee_flex": (
        "plant_knee", "임팩트", 1, "선운동역학: 지면반력·안정",
        "디딤발 무릎을 조금 더 굽히세요. 무릎을 살짝 굽혀야 지면반력을 흡수하고 균형을 유지하기 쉽습니다.",
        "디딤발 무릎이 많이 굽었습니다. 디딤다리가 내려앉으면 차는 다리의 회전 공간과 힘 전달이 줄 수 있습니다."),
    "followthrough_r_hip_flex": (
        "followthrough_hip", "팔로스루", 2, "부상 위험: 햄스트링 감속 부하",
        "팔로스루를 더 길게 가져가세요. 다리를 급하게 멈추면 햄스트링이 짧은 시간에 큰 감속을 해야 해서 부하가 커질 수 있습니다.",
        "팔로스루에서 다리가 과도하게 올라갑니다. 햄스트링이 늘어난 상태의 감속 부하가 커질 수 있으니 상체와 함께 자연스럽게 마무리하세요."),
    "sequence_ok": (
        "sequence", "가속", 3, "근위→원위 순차", "", ""),
    "peak_shank_ang_vel": (
        "shank_speed", "가속", 2, "각운동학",
        "정강이(무릎 아래)를 더 빠르게 휘두르세요. 허벅지를 먼저 휘두른 뒤 감속하면 정강이 각속도가 커집니다(채찍 효과).",
        "정강이 각속도가 기준보다 큽니다. 모션 블러 등 측정 오차일 수 있으니 슬로모션 영상으로 다시 확인해 보세요."),
    "foot_speed_impact": (
        "foot_speed", "임팩트", 2, "선운동학: 속도",
        "임팩트 직전 발 속도를 높이세요. 공 속도 ≈ 발 속도 × 공/발 속도비이므로 발 속도가 공 속도를 결정합니다.",
        "발 속도가 기준보다 빠릅니다. 이 킥은 정확도가 중요하니 속도보다 접촉 부위와 방향을 먼저 맞추세요."),
    "arm_balance": (
        "arm_balance", "임팩트", 1, "균형 (각운동량 보존)",
        "임팩트 때 차는 다리 반대쪽 팔을 앞·옆으로 더 들어 올리세요. 팔이 다리의 회전(각운동량)을 상쇄해 몸의 균형을 잡아 줍니다.",
        "반대팔을 너무 높이 들었습니다. 상체가 뒤로 젖혀지거나 균형이 흔들릴 수 있습니다."),
    "kick_duration": (
        "kick_duration", "백스윙→임팩트", 1, "선운동학: 시간",
        "백스윙에서 임팩트까지가 너무 짧습니다. 다리를 충분히 끌어온 뒤 휘둘러야 가속할 시간이 확보됩니다.",
        "백스윙에서 임팩트까지가 깁니다. 동작이 느리면 각속도가 떨어지고 상대에게 읽히기 쉽습니다."),
}
SEQ_FAIL_MSG = ("정강이가 허벅지보다 먼저 최고 속도에 도달했습니다. 허벅지를 먼저 앞으로 휘두른 뒤 감속시키면 "
                "각운동량이 정강이로 전달됩니다(채찍 효과).")

# 킥별 예외: 제외할 지표, 메시지·범위 덮어쓰기
EXCLUDE: dict[str, set[str]] = {
    "toe": {"max_r_knee_flex"},                                   # 토 킥은 백스윙이 거의 없음
}
OVERRIDE: dict[str, dict[str, dict]] = {
    "instep": {
        "impact_r_knee_flex": {"good_range": [40, 60], "note": "CLAUDE.md 9장 선행연구 범위 (임팩트 무릎 40~60°)"},
        "peak_shank_ang_vel": {
            "good_range": [1350, 2500], "tol": 250,
            "note": "선행연구 범위 1500~2500°/s (CLAUDE.md 9장). 영상(120 fps, 25 Hz 필터) 측정은 최고점이 "
                    "약 5~10% 낮게 나오므로 하한을 10% 낮춰 1350°/s로 보정했다. 템플릿(약 1500°/s)도 측정값은 약 1460°/s."},
    },
    "inside": {
        "impact_r_ankle_pf": {
            "low_msg": "발목을 너무 당겼습니다(배측굴곡). 발 안쪽 면이 공에 수직으로 닿도록 발목을 직각 가까이 고정하세요.",
            "high_msg": "발목을 세워(발끝을 들어) 고정하세요. 발 안쪽의 넓은 면이 공에 닿아야 방향이 정확해집니다."},
    },
    "chip": {
        "followthrough_r_hip_flex": {
            "direction": "작을수록 좋음",
            "high_msg": "팔로스루를 짧게 끊으세요. 로빙은 공 아래를 찍고 다리를 멈춰야 백스핀이 걸리고 공이 높이 뜹니다.",
            "low_msg": ""},
    },
}


def tolerance(metric: str, value: float) -> float:
    unit = METRICS[metric][1]
    if unit == "s":
        return TOL_TIME * value
    if unit in ("m/s", "°/s"):
        return TOL_SPEED * value
    return TOL_ANGLE


def rnd(metric: str, v: float) -> float:
    unit = METRICS[metric][1]
    return round(v, 3) if unit == "s" else round(v, 1) if unit == "m/s" else round(v)


def build() -> dict[str, list[dict]]:
    rules: dict[str, list[dict]] = {}
    for kid in available_kicks():
        p = preprocess(synthesize(load_kick(kid), H, FPS), H, FPS)
        values = compute_metrics(p, detect_events(p))
        out = []
        for metric, (rid, phase, weight, concept, low, high) in BASE.items():
            if metric in EXCLUDE.get(kid, set()) or values[metric].value is None:
                continue
            v = values[metric].value
            rule = {"id": rid, "phase": phase, "metric": metric, "unit": METRICS[metric][1]}
            if metric == "sequence_ok":
                if not v:
                    continue                    # 템플릿 자체가 순차가 아니면 판정하지 않음
                rule.update({"expected": True, "fail_msg": SEQ_FAIL_MSG})
            else:
                tol = tolerance(metric, v)
                rule.update({"template_value": rnd(metric, v),
                             "good_range": [rnd(metric, v - tol), rnd(metric, v + tol)],
                             "tol": rnd(metric, tol), "low_msg": low, "high_msg": high})
            rule.update({"weight": weight, "concept": concept, "is_example": True})
            ov = dict(OVERRIDE.get(kid, {}).get(metric, {}))
            if ov.pop("direction", None) == "작을수록 좋음":
                rule["good_range"][0] = None
                rule["direction"] = "작을수록 좋음"
            rule.update(ov)
            out.append(rule)
        rules[kid] = out
    return rules


if __name__ == "__main__":
    rules = build()
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"{RULES_PATH} 작성: " + ", ".join(f"{k} {len(v)}개" for k, v in rules.items()))
