"""후면(관상면) 영상 분석 — 3D 모델을 뒤에서 본 합성 랜드마크로 검증."""

import numpy as np
import pytest

from biomech.kick_loader import available_kicks, load_kick
from video.classify import classify
from video.events import detect_events
from video.feedback import IMPROVE, METRICS, load_rules, make_feedback
from video.frontal import RULES_REAR_PATH, compute_frontal_metrics, rear_rules
from video.preprocess import detect_view, preprocess
from video.synthetic import PX_PER_M, synthesize

H = 1.75
FPS = 120
GROUND_KICKS = [k for k in available_kicks() if not load_kick(k)["airborne"]]
AERIAL_BALL = {"volley"}            # 공중의 공: 스윙 최저점 ≠ 임팩트


def rear(kick_id: str, raw=None, **opts):
    raw = raw or synthesize(load_kick(kick_id), H, FPS, view="rear", **opts)
    p = preprocess(raw, H, FPS, view="auto")
    return p, detect_events(p)


def rear_feedback(kick_id: str, raw=None, **opts):
    p, ev = rear(kick_id, raw, **opts)
    return make_feedback(kick_id, compute_frontal_metrics(p, ev), rear_rules(kick_id))


@pytest.mark.parametrize("kick_id", available_kicks())
def test_view_detection(kick_id):
    kick = load_kick(kick_id)
    assert detect_view(synthesize(kick, H, FPS)) == "side"
    assert detect_view(synthesize(kick, H, FPS, view="rear")) == "rear"


@pytest.mark.parametrize("kick_id", GROUND_KICKS)
def test_rear_normalization_invariant(kick_id):
    """후면: 왼발 킥·좌우 반전(정면 촬영처럼)이어도 같은 관상면 각도."""
    base, _ = rear(kick_id)
    assert base.view == "rear" and base.kick_side == "R"
    for opts in (dict(mirror=True), dict(left_footed=True), dict(left_footed=True, mirror=True)):
        p, _ = rear(kick_id, **opts)
        for k, v in base.angles.items():
            assert np.max(np.abs(p.angles[k] - v)) < 1e-6, (opts, k)


def test_rear_scale_estimate():
    p, _ = rear("instep")
    assert p.px_per_m == pytest.approx(PX_PER_M, rel=0.05)


@pytest.mark.parametrize("kick_id", GROUND_KICKS)
def test_rear_impact_timing(kick_id):
    """후면 임팩트(스윙 최저점): 지면 공은 ±4프레임(120 fps, 약 33 ms), 공중의 공은 ±6프레임.

    측면(±1프레임)보다 느슨한 이유: 뒤에서는 발의 앞뒤 움직임과 충돌 감속이 보이지 않는다.
    """
    kick = load_kick(kick_id)
    tol = 6 if kick_id in AERIAL_BALL else 4
    for opts in ({}, dict(noise_frac=0.005, seed=2)):
        _, ev = rear(kick_id, **opts)
        t_true = kick["impact"]["time"] - kick["keyframes"][0]["t"]
        assert abs(ev.time("임팩트") - t_true) * FPS <= tol + 1e-9


@pytest.mark.parametrize("kick_id", GROUND_KICKS)
def test_rear_template_no_improvements(kick_id):
    """템플릿을 뒤에서 본 동작은 후면 기준을 모두 만족한다."""
    fb = rear_feedback(kick_id)
    assert fb.n_improve == 0, [(j.rule["id"], j.value) for j in fb.items if j.status == IMPROVE]


def _shift(raw, names, dx_m):
    for k in names:
        raw.xy[k] = raw.xy[k] + np.array([dx_m * PX_PER_M, 0.0])
    return raw


def test_valgus_detected():
    """디딤 무릎을 몸 안쪽으로 8 cm 밀면 → 무릎 외반 경고가 Top 3 + high_msg."""
    raw = _shift(synthesize(load_kick("instep"), H, FPS, view="rear"), ["l_knee"], 0.08)
    fb = rear_feedback("instep", raw)
    top = {j.rule["id"]: j for j in fb.top3}
    assert "plant_knee_valgus" in top
    assert "안쪽으로" in top["plant_knee_valgus"].message


def test_plant_foot_too_close():
    """디딤발을 차는 발 쪽으로 12 cm 옮기면 → 디딤발 위치 경고(low_msg)."""
    raw = _shift(synthesize(load_kick("instep"), H, FPS, view="rear"),
                 ["l_ankle", "l_heel", "l_toe"], 0.12)
    fb = rear_feedback("instep", raw)
    j = next(j for j in fb.items if j.rule["id"] == "plant_foot_side")
    assert j.status == IMPROVE and j.message == j.rule["low_msg"]


def test_trunk_lean_wrong_side():
    """몸통을 차는 다리 쪽으로 기울이면 → 몸통 옆 기울기 경고(low_msg)."""
    raw = _shift(synthesize(load_kick("instep"), H, FPS, view="rear"),
                 ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "nose"], 0.12)
    fb = rear_feedback("instep", raw)
    j = next(j for j in fb.items if j.rule["id"] == "trunk_side_lean")
    assert j.status == IMPROVE and j.message == j.rule["low_msg"]


def test_arm_not_abducted():
    """디딤발 쪽 팔을 몸에 붙이면 → 팔 벌림 경고."""
    raw = synthesize(load_kick("instep"), H, FPS, view="rear")
    for k in ("l_elbow", "l_wrist"):
        raw.xy[k][:, 0] = raw.xy["l_shoulder"][:, 0]
    fb = rear_feedback("instep", raw)
    j = next(j for j in fb.items if j.rule["id"] == "balance_arm")
    assert j.status == IMPROVE


def test_rear_rules_format():
    rules = load_rules(RULES_REAR_PATH)
    assert set(rules) == {"_note", "default"}
    for r in rules["default"]:
        assert r["metric"] in METRICS and r["is_example"] is True and r["concept"]
    assert rear_rules("instep") == rules["default"]


def test_classify_refuses_rear():
    p, ev = rear("instep")
    with pytest.raises(ValueError, match="측면"):
        classify(p, ev)


def test_side_pipeline_unchanged_by_view_param():
    """측면은 view를 지정하든 자동 감지하든 결과가 같다."""
    raw = synthesize(load_kick("instep"), H, FPS)
    a, b = preprocess(raw, H, FPS), preprocess(raw, H, FPS, view="auto")
    assert b.view == "side"
    assert all(np.allclose(a.angles[k], b.angles[k]) for k in a.angles)
