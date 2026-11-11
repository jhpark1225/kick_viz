import copy
import re

import pytest

from biomech.kick_loader import available_kicks, load_kick
from video.events import detect_events
from video.feedback import (GOOD, IMPROVE, METRICS, UNSURE, VELOCITY_METRICS, compute_metrics,
                            load_rules, make_feedback)
from video.preprocess import preprocess
from video.synthetic import synthesize

H = 1.75
FPS = 120


def feedback_for(kick: dict, kick_id: str = "instep", fps: float = FPS):
    p = preprocess(synthesize(kick, H, fps), H, fps)
    return make_feedback(kick_id, compute_metrics(p, detect_events(p)))


def test_instep_template_no_improvements():
    """인스텝 템플릿 그대로 → ⚠️ 개선 항목 0개."""
    fb = feedback_for(load_kick("instep"))
    assert fb.n_improve == 0, [(j.rule["id"], j.value) for j in fb.items if j.status == IMPROVE]
    assert fb.top3 == []
    assert 1 <= len(fb.strengths) <= 2


@pytest.mark.parametrize("kick_id", available_kicks())
def test_every_template_passes_own_rules(kick_id):
    """모든 킥: 자기 템플릿 그대로면 ⚠️ 0개 (규칙 범위가 템플릿과 일치)."""
    fb = feedback_for(load_kick(kick_id), kick_id)
    assert fb.n_improve == 0, [(j.rule["id"], j.value) for j in fb.items if j.status == IMPROVE]


def test_reduced_backswing_knee():
    """백스윙 무릎 굴곡을 30° 줄인 변형 → backswing_knee가 Top 3, low_msg 출력.

    백스윙 직후 '가속' 키프레임의 무릎(95°)도 백스윙 구간 최대값 후보이므로 함께 30° 줄인다.
    """
    kick = copy.deepcopy(load_kick("instep"))
    for kf in kick["keyframes"]:
        if kf["phase"] in ("백스윙", "가속"):
            kf["r_knee"] -= 30
    fb = feedback_for(kick)
    top = {j.rule["id"]: j for j in fb.top3}
    assert "backswing_knee" in top
    assert top["backswing_knee"].message == top["backswing_knee"].rule["low_msg"]
    assert "무릎을 더 접으세요" in top["backswing_knee"].message


def test_reversed_sequence_in_top3():
    """허벅지·정강이 최고점 순서를 뒤바꾼 변형(가속 단계에서 무릎을 먼저 폄) → sequence_ok 경고가 Top 3."""
    kick = copy.deepcopy(load_kick("instep"))
    for kf in kick["keyframes"]:
        if kf["phase"] == "가속":
            kf["r_hip"], kf["r_knee"] = -25, 40
    fb = feedback_for(kick)
    seq = [j for j in fb.top3 if j.rule["metric"] == "sequence_ok"]
    assert seq and seq[0].value is False
    assert "채찍 효과" in seq[0].message


def test_low_fps_velocity_metrics_uncertain():
    """fps = 30으로 합성 → 각속도(·속도) 지표는 ❔ 측정 불확실."""
    fb = feedback_for(load_kick("instep"), fps=30)
    for j in fb.items:
        if j.rule["metric"] in VELOCITY_METRICS:
            assert j.status == UNSURE, j.rule["metric"]
    assert any(j.status != UNSURE for j in fb.items if j.rule["metric"] not in VELOCITY_METRICS)
    assert all(j.status != UNSURE for j in fb.top3)


def test_low_visibility_uncertain():
    """해당 구간 visibility < 0.5 → ❔ 측정 불확실."""
    kick = load_kick("instep")
    raw = synthesize(kick, H, FPS)
    for k in ("r_ankle", "r_toe", "r_heel"):
        raw.vis[k][:] = 0.2
    p = preprocess(raw, H, FPS)
    fb = make_feedback("instep", compute_metrics(p, detect_events(p)))
    ankle = next(j for j in fb.items if j.rule["metric"] == "impact_r_ankle_pf")
    assert ankle.status == UNSURE and "visibility" in ankle.reason


def test_rules_json_format():
    rules = load_rules()
    assert set(rules) == set(available_kicks())
    for kid, items in rules.items():
        ids = [r["id"] for r in items]
        assert len(ids) == len(set(ids))
        for r in items:
            assert r["metric"] in METRICS and r["is_example"] is True
            assert r["concept"] and r["weight"] > 0 and r["phase"]
            if "expected" in r:
                assert r["fail_msg"]
            else:
                lo, hi = r["good_range"]
                assert lo is None or hi is None or lo < hi
    assert "max_r_knee_flex" not in {r["metric"] for r in rules["toe"]}
    chip_ft = next(r for r in rules["chip"] if r["metric"] == "followthrough_r_hip_flex")
    assert chip_ft["good_range"][0] is None      # 작을수록 좋음 (반대 방향 판정)
    instep = {r["metric"]: r for r in rules["instep"]}
    assert instep["impact_r_knee_flex"]["good_range"] == [40, 60]


def test_messages_not_medical():
    """의료적 진단처럼 들리는 표현을 쓰지 않는다."""
    banned = re.compile(r"진단|파열|손상됩니다|부상입니다|질환")
    for items in load_rules().values():
        for r in items:
            for key in ("low_msg", "high_msg", "fail_msg"):
                assert not banned.search(r.get(key, "")), r[key]


def test_strengths_are_good_items():
    fb = feedback_for(load_kick("instep"))
    assert all(j.status == GOOD for j in fb.strengths)
    assert fb.strengths[0].rule["weight"] == max(j.rule["weight"] for j in fb.items)
