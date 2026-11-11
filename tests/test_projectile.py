import numpy as np
import pytest

from biomech.projectile import (G, best_angle, flight_time, horizontal_range, max_height,
                                trajectory, trajectory_drag)


@pytest.mark.parametrize("v", [10.0, 20.0, 30.0])
def test_best_angle_is_45_without_drag(v):
    assert best_angle(v) == pytest.approx(45.0, abs=1.0)


@pytest.mark.parametrize("v,theta", [(10, 30), (25, 10), (18, 50), (30, 45)])
def test_analytic_formulas(v, theta):
    s = np.sin(np.radians(theta))
    assert flight_time(v, theta) == pytest.approx(2 * v * s / G)
    assert max_height(v, theta) == pytest.approx((v * s) ** 2 / (2 * G))
    assert horizontal_range(v, theta) == pytest.approx(v**2 * np.sin(np.radians(2 * theta)) / G)


def test_trajectory_matches_formulas():
    x, y = trajectory(20, 35)
    assert y[0] == pytest.approx(0) and y[-1] == pytest.approx(0, abs=1e-9)
    assert y.max() == pytest.approx(max_height(20, 35), rel=1e-3)
    assert x[-1] == pytest.approx(horizontal_range(20, 35))


def test_drag_shortens_range():
    _, xd, _ = trajectory_drag(25, 30)
    assert xd[-1] < horizontal_range(25, 30)
