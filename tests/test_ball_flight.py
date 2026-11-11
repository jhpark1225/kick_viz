import numpy as np
import pytest

from biomech.ball_flight import (STEER, ball_path_3d, foot_params, kick_flight, launch_direction, simulate,
                                 spin_description, spin_from_contact)
from biomech.body3d import foot_yaw_series, to_3d
from biomech.kick_loader import available_kicks, load_kick
from biomech.kinematics import compute_motion, frame_points
from biomech.projectile import BALL_RADIUS, flight_time, horizontal_range, max_height, trajectory_drag

V, ELEV = 25.0, 15.0


def test_no_spin_no_drag_matches_parabola():
    """회전·공기저항이 없으면 포물선 해석해와 같다."""
    f = simulate(20.0, 30.0, h0=0.3, drag=False, magnus=False, dt=0.0005)
    assert f.range_m == pytest.approx(horizontal_range(20.0, 30.0, 0.3), rel=1e-4)
    assert f.flight_time == pytest.approx(flight_time(20.0, 30.0, 0.3), rel=1e-3)
    assert f.max_height == pytest.approx(max_height(20.0, 30.0, 0.3), rel=1e-3)
    assert abs(f.lateral_m) < 1e-9


def test_drag_matches_2d_model():
    """회전 없이 공기저항만 → 기존 2D 항력 궤적과 같다."""
    _, x, _ = trajectory_drag(20.0, 30.0)
    assert simulate(20.0, 30.0, magnus=False).range_m == pytest.approx(x[-1], rel=2e-3)


def test_spin_formula_magnitude():
    """ω = 3Δv d / (2r²) (얇은 구껍질 각충격량)."""
    w = spin_from_contact(V, launch_direction(0, 0), 0.02, 0.0)
    assert np.linalg.norm(w) == pytest.approx(1.5 * V * 0.02 / BALL_RADIUS**2)


def test_below_center_backspin_lifts():
    """공 아래를 차면 백스핀 → 회전 없을 때보다 더 높이·멀리 (로빙)."""
    spin = kick_flight(V, ELEV, 0.0, 0, 0, 0.0, -0.03)
    flat = kick_flight(V, ELEV, 0.0, 0, 0, 0.0, -0.03, magnus=False)
    assert spin.omega[2] > 0                                # 왼쪽 축(+Z) 회전 = 백스핀
    assert spin.max_height > flat.max_height
    assert "백스핀" in spin_description(spin)[0]


def test_above_center_topspin_dips():
    spin = kick_flight(V, ELEV, 0.0, 0, 0, 0.0, 0.03)
    flat = kick_flight(V, ELEV, 0.0, 0, 0, 0.0, 0.03, magnus=False)
    assert spin.max_height < flat.max_height and spin.range_m < flat.range_m


@pytest.mark.parametrize("side, bends", [(0.02, "left"), (-0.02, "right")])
def test_side_contact_curves(side, bends):
    """공의 오른쪽을 차면(감아차기) 왼쪽으로, 왼쪽을 차면(아웃사이드) 오른쪽으로 휜다. 좌우 대칭."""
    f = kick_flight(V, ELEV, 0.0, 0, 0, side, 0.0)
    assert (f.lateral_m > 0.3) if bends == "left" else (f.lateral_m < -0.3)
    mirror = kick_flight(V, ELEV, 0.0, 0, 0, -side, 0.0)
    assert f.lateral_m == pytest.approx(-mirror.lateral_m, abs=1e-6)
    assert ("왼쪽" if bends == "left" else "오른쪽") in spin_description(f)[-1]


def test_curve_size_reasonable():
    """감아차기(25 m/s, 2 cm 옆): 회전 약 8~12 회/s, 휘는 거리 1~6 m (프리킥 수준)."""
    f = kick_flight(V, ELEV, 0.0, 0, 0, 0.02, 0.0)
    assert 8 <= f.spin_rps <= 12 and 1.0 <= f.lateral_m <= 6.0


def test_center_contact_no_spin():
    """정중앙 → 회전 없음, 휘지 않음."""
    f = kick_flight(V, ELEV, 0.0, 0, 0, 0.0, 0.0)
    assert f.spin_rps == 0 and abs(f.lateral_m) < 1e-9
    assert "회전 거의 없음" in spin_description(f)[0]


def test_foot_yaw_steers_ball():
    """발끝을 바깥으로 더 돌리면(Δψ > 0) 공이 오른쪽으로 간다 (방향 변화 = STEER × Δψ)."""
    f = kick_flight(V, ELEV, 0.0, 0, 10.0, 0.0, 0.0)
    az = np.degrees(np.arctan2(f.v0[2], f.v0[0]))
    assert az == pytest.approx(-STEER * 10.0)
    assert f.lateral_m < 0


def test_off_center_reduces_speed():
    f = kick_flight(V, ELEV, 0.0, 0, 0, 0.0, -0.05)
    assert np.linalg.norm(f.v0) == pytest.approx(V * np.sqrt(1 - (0.05 / BALL_RADIUS) ** 2))


@pytest.mark.parametrize("kick_id", available_kicks())
def test_kick_defaults(kick_id):
    kick = load_kick(kick_id)
    fp = foot_params(kick)
    assert {"foot_yaw_deg", "aim_deg", "contact_offset_m"} <= set(kick["impact"])
    f = kick_flight(20.0, kick["impact"]["launch_angle_deg"], 0.5, fp["aim_deg"], 0, fp["side"], fp["up"])
    assert np.isfinite(f.pos).all()
    if kick_id == "curve":
        # 오른쪽으로 조준해 출발하므로 착지점 자체보다 '회전 없을 때보다 왼쪽으로 휘었는지'를 본다
        flat = kick_flight(20.0, kick["impact"]["launch_angle_deg"], 0.5, fp["aim_deg"], 0, fp["side"], fp["up"],
                           magnus=False)
        assert fp["aim_deg"] < 0 and f.lateral_m > flat.lateral_m + 0.3
    if kick_id == "outside":
        assert f.omega[1] > 0                                # 아웃사이드 → 오른쪽으로 휘는 회전
    if kick_id == "toe":
        assert f.spin_rps < 0.3


def test_foot_yaw_in_3d_model():
    """3D 모델: 인사이드 킥 임팩트 때 차는 발끝이 바깥(오른쪽 = −Z)으로 돌아가고 발 길이는 그대로."""
    kick = load_kick("inside")
    m = compute_motion(kick, 1.75)
    i = int(np.argmin(np.abs(m.t - kick["impact"]["time"])))
    p = frame_points(m, i)
    yaw = foot_yaw_series(kick, m.t[i:i + 1])[0]
    assert yaw == pytest.approx(kick["impact"]["foot_yaw_deg"])
    a, b = to_3d(p, 1.75), to_3d(p, 1.75, foot_yaw_deg=yaw)
    assert b["r_toe"][2] < a["r_toe"][2] - 0.1
    for j in ("r_toe", "r_heel"):
        assert np.linalg.norm(b[j] - b["r_ankle"]) == pytest.approx(np.linalg.norm(a[j] - a["r_ankle"]))
    assert np.allclose(b["l_toe"], a["l_toe"])               # 디딤발은 그대로
    assert foot_yaw_series(kick, np.array([0.0]))[0] == 0.0  # 접근 때는 0


def test_ball_path_3d_continuous_at_impact():
    t = np.linspace(0, 1, 101)
    path2d = np.column_stack([np.full(101, 1.0), np.full(101, BALL_RADIUS)])
    f = kick_flight(20.0, 10.0, 0.0, 0, 0, 0.02, 0.0)
    out = ball_path_3d(t, 0.5, path2d, np.array([1.0, BALL_RADIUS]), -0.1, f)
    assert np.allclose(out[50], [1.0, BALL_RADIUS, -0.1])
    assert out[-1, 0] > 1.0 and out[-1, 2] > -0.1            # 앞으로, 왼쪽으로 휘며 날아감


# ---------------------------------------------------------------------------
# 왼발잡이 (오른발 킥의 좌우 거울상)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("side, up", [(0.02, 0.0), (-0.015, -0.02), (0.0, -0.03)])
def test_left_foot_is_mirror(side, up):
    """왼발: 실제 접촉 위치를 좌우 반대로 주면 궤적은 오른발의 정확한 거울상, 백스핀은 그대로."""
    r = kick_flight(V, ELEV, 0.0, -8.0, 5.0, side, up)
    lf = kick_flight(V, ELEV, 0.0, -8.0, 5.0, -side, up, left=True)
    assert np.allclose(lf.pos[:, :2], r.pos[:, :2]) and np.allclose(lf.pos[:, 2], -r.pos[:, 2])
    assert np.allclose(lf.omega, r.omega * np.array([-1, -1, 1]))
    assert lf.spin_rps == pytest.approx(r.spin_rps)


def test_left_foot_curve_bends_right():
    """왼발 감아차기(공의 왼쪽 옆을 감아 참) → 오른쪽으로 휜다."""
    kick = load_kick("curve")
    fp = foot_params(kick)
    args = (20.0, kick["impact"]["launch_angle_deg"], 0.5, fp["aim_deg"], 0, -fp["side"], fp["up"])
    lf = kick_flight(*args, left=True)
    flat = kick_flight(*args, magnus=False, left=True)
    assert lf.lateral_m < flat.lateral_m - 0.3
    assert "오른쪽" in spin_description(lf)[-1]
