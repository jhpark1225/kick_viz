import numpy as np
import pytest

from biomech.kick_loader import available_kicks, load_kick
from biomech.kinematics import (angular_velocity, compute_motion, contact_point, frame_points,
                                segment_abs_angles)

H = 1.75
N = 240          # 보간 프레임 수 (최소 200)
TOL = 1e-6
KICKS = available_kicks()


@pytest.fixture(scope="module", params=KICKS)
def kick_motion(request):
    kick = load_kick(request.param)       # load_kick이 스키마 검증을 수행
    return kick, compute_motion(kick, H, N)


def test_schema_and_frames(kick_motion):
    kick, m = kick_motion
    assert len(m.t) >= 200
    assert np.all(np.diff(m.t) > 0)


def test_feet_above_ground(kick_motion):
    kick, m = kick_motion
    if kick["airborne"]:
        # 공중 동작: 착지 프레임에서 손·등(골반·어깨)이 지면 위
        for name in ("r_hand", "l_hand", "hip", "shoulder"):
            assert m.points[name][-1, 1] >= -TOL, name
        return
    for name in ("r_toe", "r_heel", "l_toe", "l_heel"):
        y = m.points[name][:, 1]
        assert y.min() >= -TOL, f"{name} 최저 {y.min():.4f} m (t={m.t[y.argmin()]:.3f})"


def test_contact_height_at_impact(kick_motion):
    kick, m = kick_motion
    imp = kick["impact"]
    i = int(np.argmin(np.abs(m.t - imp["time"])))
    assert abs(m.t[i] - imp["time"]) < 1e-9
    h = contact_point(frame_points(m, i), imp["contact_point"])[1]
    assert h == pytest.approx(imp["contact_height"], abs=0.03)


def test_stance_foot_pinned(kick_motion):
    kick, m = kick_motion
    if kick["stance_pin"] is None:
        pytest.skip("공중 동작")
    t_plant = next(kf["t"] for kf in kick["keyframes"] if kf["phase"] == "디딤")
    pin = m.points[m.pin_point][m.t >= t_plant - 1e-9]
    assert np.max(np.linalg.norm(pin - pin[0], axis=1)) < 1e-3


def test_proximal_to_distal_sequence_instep():
    kick = load_kick("instep")
    m = compute_motion(kick, H, N)
    thigh, shank = segment_abs_angles(m.angles["trunk"], m.angles["r_hip"], m.angles["r_knee"])
    t_thigh = m.t[np.argmax(angular_velocity(m.t, thigh))]
    t_shank = m.t[np.argmax(angular_velocity(m.t, shank))]
    assert t_thigh < t_shank <= kick["impact"]["time"]
