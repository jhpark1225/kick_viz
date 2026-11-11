"""영상 → 템플릿 왕복: 템플릿으로 합성한 측면 영상에서 다시 템플릿을 만들면 원래 값이 나와야 한다."""

import numpy as np
import pytest

from biomech.kick_loader import PHASES, available_kicks, load_kick
from biomech.template_fit import fit_hip_lift
from video.events import detect_events
from video.preprocess import preprocess
from video.synthetic import synthesize
from video.template_from_video import average_keyframes, build_template, keyframes_from_video, problems

H, FPS = 1.75, 120
GROUND_KICKS = [k for k in available_kicks() if not load_kick(k)["airborne"]]


def measured(kick_id: str, **opts):
    kick = load_kick(kick_id)
    p = preprocess(synthesize(kick, H, FPS, **opts), H, FPS)
    return kick, keyframes_from_video(p, detect_events(p))


@pytest.mark.parametrize("kick_id", GROUND_KICKS)
def test_round_trip_passes_model_checks(kick_id):
    """합성 영상 → 템플릿: 스키마·지면·접촉 높이 조건 통과."""
    kick, kfs = measured(kick_id)
    new, check = build_template(kfs, kick, "합성")
    bad = [s for s in problems(new, check) if "순차" not in s]
    assert not bad, bad
    assert [kf["phase"] for kf in new["keyframes"]] == PHASES


def test_round_trip_instep_close_to_template():
    """인스텝: 임팩트 각도는 ±10°(무릎 ±15°), 무릎 최대 굴곡→임팩트 시간은 ±2프레임 안에서 원래 템플릿과 같다.

    디딤은 8 Hz 필터가 급정지를 부드럽게 만들어 약 30 ms 늦게 잡히므로 시간 비교에 쓰지 않는다.

    합성 영상에는 공과의 충돌이 없어 임팩트(급감속)가 1프레임(8 ms) 일찍 잡힐 수 있고, 그동안
    무릎은 약 1500°/s로 펴지므로 약 12° 덜 펴진 값이 측정된다. 그래서 무릎만 ±15°로 둔다.
    """
    kick, kfs = measured("instep")
    t0 = kick["keyframes"][0]["t"]
    imp_true = next(kf for kf in kick["keyframes"] if kf["phase"] == "임팩트")
    imp = kfs[PHASES.index("임팩트")]
    for k in ("trunk", "r_hip", "r_knee", "r_ankle", "l_knee"):
        tol = 15 if k == "r_knee" else 10
        assert abs(imp[k] - imp_true[k]) <= tol, (k, imp[k], imp_true[k])
    mkf_true = next(kf["t"] for kf in kick["keyframes"] if kf["phase"] == "가속")
    dt_true = imp_true["t"] - mkf_true
    dt = imp["t"] - kfs[PHASES.index("가속")]["t"]
    assert abs(dt - dt_true) * FPS <= 2


def test_rear_view_rejected():
    kick = load_kick("instep")
    p = preprocess(synthesize(kick, H, FPS, view="rear"), H, FPS, view="rear")
    with pytest.raises(ValueError, match="측면"):
        keyframes_from_video(p, detect_events(p))


def test_average_of_identical_runs_is_same():
    _, kfs = measured("instep")
    avg = average_keyframes([kfs, kfs, kfs])
    assert avg == [{k: kf[k] for k in avg[0]} for kf in kfs]


def test_average_of_noisy_runs():
    """노이즈가 다른 영상 3개 평균 → 단계 시각이 오름차순, 템플릿 생성 가능."""
    kick = load_kick("instep")
    runs = [measured("instep", noise_frac=0.005, seed=s)[1] for s in (1, 2, 3)]
    avg = average_keyframes(runs)
    ts = [kf["t"] for kf in avg]
    assert np.all(np.diff(ts) > 0)
    new, check = build_template(avg, kick, "합성 3개 평균")
    assert not [s for s in problems(new, check) if "지면" in s or "접촉 높이" in s]


def test_fit_hip_lift_reaches_contact_height():
    """임팩트 lift를 0으로 망가뜨려도 보정 후 접촉 높이가 공 높이로 돌아온다."""
    kick = load_kick("instep")
    for kf in kick["keyframes"]:
        kf["kick_hip_lift"] = 0.0
    fixed, check = fit_hip_lift(kick)
    assert check["contact_height"] == pytest.approx(kick["impact"]["contact_height"], abs=0.005)
    assert check["kick_foot_min"] >= 0


def test_average_resists_one_bad_run():
    """영상 3개 중 1개가 크게 틀려도(무릎 +40°, 시각 +0.2 s) 중앙값이라 결과가 끌려가지 않는다."""
    _, kfs = measured("instep")
    bad = [{**kf, "r_knee": kf["r_knee"] + 40, "t": round(kf["t"] + 0.2, 3)} for kf in kfs]
    avg = average_keyframes([kfs, kfs, bad])
    for a, k in zip(avg, kfs):
        assert a["r_knee"] == k["r_knee"] and a["t"] == k["t"]
