import itertools

import numpy as np
import pytest

from biomech.body import segment_lengths
from biomech.kinematics import angular_velocity, interpolate, leg, pose

H = 1.75
L = segment_lengths(H)
HIP = np.array([0.0, 1.0])


def test_straight_leg_hangs_below_hip():
    knee, ankle, _, _ = leg(HIP, 0, 0, 0, 0, L, H)
    assert ankle[0] == pytest.approx(HIP[0], abs=1e-9)
    assert ankle[1] < HIP[1]
    assert np.linalg.norm(ankle - HIP) == pytest.approx(L["thigh"] + L["shank"])


def test_hip_flex_90_thigh_horizontal_forward():
    knee, *_ = leg(HIP, 0, 90, 0, 0, L, H)
    assert knee[1] == pytest.approx(HIP[1], abs=1e-9)
    assert knee[0] > HIP[0]


def test_trunk_lean_with_equal_hip_flex_keeps_thigh_vertical():
    knee, *_ = leg(HIP, 30, 30, 0, 0, L, H)
    assert knee[0] == pytest.approx(HIP[0], abs=1e-9)
    assert knee[1] < HIP[1]


def test_knee_flex_90_shank_horizontal_backward():
    knee, ankle, _, _ = leg(HIP, 0, 0, 90, 0, L, H)
    assert ankle[1] == pytest.approx(knee[1], abs=1e-9)
    assert ankle[0] < knee[0]


def test_ankle_pf_90_toe_points_down_along_shank():
    knee, ankle, _, toe = leg(HIP, 0, 0, 0, 90, L, H)
    shank_dir = (ankle - knee) / np.linalg.norm(ankle - knee)
    # 발끝이 정강이 연장 방향(아래)으로 향함: 발목→발끝 벡터의 주성분이 정강이 방향
    v = toe - ankle
    assert np.dot(v, shank_dir) == pytest.approx(0.12 * H, abs=1e-9)
    assert toe[1] < ankle[1]


@pytest.mark.parametrize("trunk,hip,knee,ankle", list(itertools.product(
    [-120, -20, 0, 15], [-30, 0, 70, 110], [0, 50, 105], [-15, 0, 55])))
def test_segment_lengths_preserved(trunk, hip, knee, ankle):
    angles = {"trunk": trunk, "r_hip": hip, "r_knee": knee, "r_ankle": ankle,
              "l_hip": -hip, "l_knee": knee, "l_ankle": -ankle,
              "r_shoulder": hip, "l_shoulder": -hip, "r_elbow": knee, "l_elbow": 20}
    p = pose(angles, HIP, H, kick_hip_lift=0.1)
    dist = lambda a, b: np.linalg.norm(p[a] - p[b])
    for s in "rl":
        assert dist(f"{s}_hip", f"{s}_knee") == pytest.approx(L["thigh"])
        assert dist(f"{s}_knee", f"{s}_ankle") == pytest.approx(L["shank"])
        assert dist(f"{s}_ankle", f"{s}_toe") == pytest.approx(np.hypot(0.12, 0.039) * H)
        assert dist(f"{s}_ankle", f"{s}_heel") == pytest.approx(np.hypot(0.03, 0.039) * H)
        assert dist("shoulder", f"{s}_elbow") == pytest.approx(L["upper_arm"])
        assert dist(f"{s}_elbow", f"{s}_hand") == pytest.approx(L["forearm"])
    assert dist("hip", "shoulder") == pytest.approx(L["trunk"])
    assert dist("shoulder", "head") == pytest.approx(L["neck_head"])


def test_pchip_does_not_overshoot():
    kfs = [{"t": 0.0, "v": 0}, {"t": 0.1, "v": 10}, {"t": 0.2, "v": 10}, {"t": 0.3, "v": 0}]
    _, out = interpolate(kfs, ["v"], 301)
    assert out["v"].max() <= 10 + 1e-9 and out["v"].min() >= -1e-9


def test_angular_velocity_of_linear_ramp():
    t = np.linspace(0, 1, 101)
    assert np.allclose(angular_velocity(t, 500 * t), 500)
