import numpy as np
import pytest

from biomech.kick_loader import PHASES, available_kicks, load_kick
from video.events import detect_events
from video.preprocess import preprocess
from video.synthetic import synthesize

H = 1.75
FPS = 120
KICKS = available_kicks()
NOISE = 0.005       # σ = 키의 0.5%
SEEDS = range(3)


def impact_error_frames(kick: dict, ev) -> float:
    t_true = kick["impact"]["time"] - kick["keyframes"][0]["t"]
    return (ev.time("임팩트") - t_true) * FPS


@pytest.fixture(scope="module", params=KICKS)
def clean(request):
    kick = load_kick(request.param)
    p = preprocess(synthesize(kick, H, FPS), H, FPS)
    return kick, p, detect_events(p)


def test_impact_clean_within_1_frame(clean):
    kick, _, ev = clean
    assert abs(impact_error_frames(kick, ev)) <= 1 + 1e-9


@pytest.mark.parametrize("kick_id", KICKS)
def test_impact_noisy_within_2_frames(kick_id):
    kick = load_kick(kick_id)
    for seed in SEEDS:
        p = preprocess(synthesize(kick, H, FPS, noise_frac=NOISE, seed=seed), H, FPS)
        ev = detect_events(p)
        assert abs(impact_error_frames(kick, ev)) <= 2 + 1e-9, f"seed {seed}"


def test_no_kick_airborne(clean):
    kick, _, ev = clean
    assert not ev.airborne          # 남은 8개 킥은 모두 디딤발이 지면에 있다


def test_phase_order_ground_kicks(clean):
    """지면 킥: 접근 ≤ 디딤 < 백스윙 ≤ 가속 ≤ 임팩트 < 팔로스루, 검출 실패 없음."""
    kick, _, ev = clean
    if kick["airborne"]:
        pytest.skip("공중 동작은 디딤이 없을 수 있음")
    f = ev.frames
    assert ev.failed == []
    assert f["접근"] <= f["디딤"] < f["백스윙"] <= f["가속"] <= f["임팩트"] < f["팔로스루"]


def test_contact_height_close_to_template(clean):
    """임팩트 시 차는 발(발목·발끝 중점) 높이가 템플릿 접촉 높이와 비슷하다 (instep 계열).

    발리는 임팩트 때 발이 약 10 m/s로 올라가므로 임팩트 ±1프레임(8 ms) 오차만으로 약 8 cm가
    달라진다. 그래서 허용 오차를 0.10 m로 둔다.
    """
    kick, _, ev = clean
    if kick["impact"]["contact_point"] != "instep":
        pytest.skip("접촉점 정의가 다름")
    assert ev.contact_height == pytest.approx(kick["impact"]["contact_height"], abs=0.10)


def test_manual_override_and_failure_does_not_raise():
    """사용자 지정 프레임을 쓰고, 검출 실패는 오류 없이 None으로 표시한다."""
    kick = load_kick("instep")
    p = preprocess(synthesize(kick, H, FPS), H, FPS)
    ev = detect_events(p, {"임팩트": 40})
    assert ev.frames["임팩트"] == 40 and "임팩트" in ev.manual
    assert ev.frames["팔로스루"] >= 40
    # 디딤발이 계속 움직이는 영상 → 디딤 검출 실패지만 예외 없이 진행
    p.xy["l_ankle"] = p.xy["l_ankle"] + np.stack([np.arange(p.n) * 0.05, np.zeros(p.n)], axis=1)
    ev = detect_events(p)
    assert ev.frames["디딤"] is None and "디딤" in ev.failed
    assert ev.frames["임팩트"] is not None and set(ev.frames) == set(PHASES)
