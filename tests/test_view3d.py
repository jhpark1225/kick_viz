import numpy as np
import pytest

from biomech.body3d import ARM_ABDUCTION_DEG, HIP_HALF_RATIO, to_3d
from biomech.kick_loader import available_kicks, load_kick
from biomech.kinematics import compute_motion, frame_points
from viz.body_shapes import body_meshes, capsule, side_outline
from viz.stickfigure import static_figure
from viz.stickfigure3d import MAX_FRAMES_3D, animated_figure_3d

H = 1.75


@pytest.fixture(scope="module")
def instep_pose():
    kick = load_kick("instep")
    m = compute_motion(kick, H)
    return kick, m, frame_points(m, int(np.argmin(np.abs(m.t - kick["impact"]["time"]))))


def test_to_3d_keeps_sagittal_coordinates(instep_pose):
    """3D 모델을 옆에서 보면(X, Y) 2D 측면 모델과 같다 (팔 제외). 다리는 고관절 폭만큼 좌우로 떨어진다."""
    _, _, p = instep_pose
    p3 = to_3d(p, H)
    for k in ("hip", "r_hip", "l_knee", "r_ankle", "r_toe", "head", "shoulder"):
        assert np.allclose(p3[k][:2], p[k])
    assert p3["r_knee"][2] == pytest.approx(-HIP_HALF_RATIO * H)
    assert p3["l_knee"][2] == pytest.approx(HIP_HALF_RATIO * H)


def test_arm_abduction_preserves_length(instep_pose):
    """팔 벌림을 넣어도 위팔·아래팔 길이는 그대로이고, 팔은 바깥쪽으로 벌어진다."""
    _, _, p = instep_pose
    p3 = to_3d(p, H)
    for s, sign in (("l", 1), ("r", -1)):
        sh = p3[f"{s}_shoulder"]
        assert np.linalg.norm(p3[f"{s}_elbow"] - sh) == pytest.approx(np.linalg.norm(p[f"{s}_elbow"] - p["shoulder"]))
        assert np.linalg.norm(p3[f"{s}_hand"] - p3[f"{s}_elbow"]) == pytest.approx(
            np.linalg.norm(p[f"{s}_hand"] - p[f"{s}_elbow"]))
    assert ARM_ABDUCTION_DEG["l"] > ARM_ABDUCTION_DEG["r"]


def test_outline_shapes(instep_pose):
    _, _, p = instep_pose
    c = capsule(np.zeros(2), np.array([0.0, -0.4]), 0.07, 0.05)
    assert np.allclose(c[0], c[-1])                       # 닫힌 다각형
    assert c[:, 0].max() == pytest.approx(0.07, abs=1e-6)  # 근위 반지름만큼 옆으로 나옴
    shapes = side_outline(p, H)
    assert set(shapes) == {"far", "torso", "near"}
    assert all(len(x) > 10 for x, _ in shapes.values())


def test_body_meshes_valid(instep_pose):
    _, _, p = instep_pose
    for v, f in body_meshes(to_3d(p, H), H).values():
        assert np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v)


def test_static_figure_with_outline(instep_pose):
    kick, m, _ = instep_pose
    with_o = static_figure(kick, m, 10, H, {"outline": True})
    without = static_figure(kick, m, 10, H, {"outline": False})
    assert len(with_o.data) == len(without.data)          # 트레이스 개수 고정 (애니메이션 호환)
    assert len(with_o.data[0].x) > 0 and len(without.data[0].x) == 0


@pytest.mark.parametrize("kick_id", available_kicks())
def test_3d_animation_builds(kick_id):
    kick = load_kick(kick_id)
    m = compute_motion(kick, H)
    fig = animated_figure_3d(kick, m, H, {"outline": True}, 0.25)
    assert 2 <= len(fig.frames) <= MAX_FRAMES_3D
    n = len(fig.frames[0].data)
    assert all(len(f.data) == n for f in fig.frames)
    assert len(fig.data) == n + 1                          # + 바닥
    assert fig.layout.scene.aspectmode == "data"


def test_3d_spin_rotates_camera():
    kick = load_kick("instep")
    m = compute_motion(kick, H)
    fig = animated_figure_3d(kick, m, H, {"outline": False}, 0.25, spin=True)
    eyes = [(f.layout.scene.camera.eye.x, f.layout.scene.camera.eye.y) for f in fig.frames]
    ang = np.unwrap([np.arctan2(y, x) for x, y in eyes])
    assert ang[-1] - ang[0] == pytest.approx(2 * np.pi * (len(eyes) - 1) / len(eyes), abs=1e-6)


def test_left_foot_3d_mirror():
    """왼발: 3D에서 차는 다리(r_, 진한 색)가 왼쪽(+Z)에 오고, 측면 카메라도 차는 쪽으로 옮겨진다."""
    from viz.stickfigure3d import cameras, frame_traces_3d
    kick = load_kick("instep")
    m = compute_motion(kick, H)
    p = frame_points(m, 20)
    angles = {k: float(v[20]) for k, v in m.angles.items()}
    right = frame_traces_3d(p, angles, [], H, False, None, False)
    left = frame_traces_3d(p, angles, [], H, False, None, False, foot="L")
    kick_leg_r = [v for v in right[5].y if v is not None]          # 6번째 = 차는 다리 선 (y축 = 좌우)
    kick_leg_l = [v for v in left[5].y if v is not None]
    assert max(kick_leg_r) < 0 < min(kick_leg_l)
    assert np.allclose(kick_leg_l, [-v for v in kick_leg_r])
    assert cameras("L")["측면"]["eye"]["y"] == -cameras("R")["측면"]["eye"]["y"]


def test_left_foot_labels():
    from viz.stickfigure import joint_label, swap_lr
    assert swap_lr("오른발로 차고 왼다리로 버팀") == "왼발로 차고 오른다리로 버팀"
    assert joint_label("r_shoulder", "L") == "왼 어깨" and joint_label("r_knee", "L") == "무릎"


def test_left_foot_topview_mirrored():
    from viz.ball_charts import foot_topview
    r = foot_topview(80, "inside", 0.0, 0.0)
    lf = foot_topview(80, "inside", 0.0, 0.0, left=True)
    assert np.allclose(np.array(lf.data[0].x), -np.array(r.data[0].x))
