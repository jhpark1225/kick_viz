import numpy as np
import pytest

from biomech.kinematics import ANGLE_KEYS, pose
from video.angles_from_pose import angles_from_pose, landmarks_from_fk, wrap180
from video.preprocess import to_body_frame

H = 1.75
N_POSES = 1000
TOL_DEG = 0.01

# 무작위 자세 범위 (도)
RANGES = {
    "trunk": (-120, 60), "r_hip": (-60, 130), "l_hip": (-60, 130),
    "r_knee": (0, 150), "l_knee": (0, 150), "r_ankle": (-30, 80), "l_ankle": (-30, 80),
    "r_shoulder": (-60, 175), "l_shoulder": (-60, 175), "r_elbow": (0, 150), "l_elbow": (0, 150),
}


def random_poses(n: int, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [{k: rng.uniform(*RANGES[k]) for k in ANGLE_KEYS} for _ in range(n)]


def max_err(a: dict, b: dict) -> float:
    return max(float(np.max(np.abs(wrap180(np.asarray(a[k]) - np.asarray(b[k]))))) for k in ANGLE_KEYS)


def test_roundtrip_random_poses():
    """FK → 랜드마크 → angles_from_pose → 원래 각도 (오차 < 0.01°)."""
    rng = np.random.default_rng(1)
    for ang in random_poses(N_POSES):
        hip = rng.uniform(-2, 2, size=2)
        lm = landmarks_from_fk(pose(ang, hip, H))
        assert max_err(angles_from_pose(lm), ang) < TOL_DEG


def test_roundtrip_vectorized_series():
    """시계열(n, 2) 입력도 프레임별 결과와 같다."""
    poses = random_poses(50, seed=2)
    lms = [landmarks_from_fk(pose(a, np.zeros(2), H)) for a in poses]
    series = {k: np.array([lm[k] for lm in lms]) for k in lms[0]}
    out = angles_from_pose(series)
    truth = {k: np.array([a[k] for a in poses]) for k in ANGLE_KEYS}
    assert max_err(out, truth) < TOL_DEG


def to_image(lm: dict, px_per_m: float, mirror: bool) -> dict:
    """몸 좌표(m, 위 +y) → 이미지 좌표(픽셀, y 아래로 증가), mirror면 왼쪽으로 차는 영상."""
    sx = -1.0 if mirror else 1.0
    return {k: np.array([960 + sx * v[0] * px_per_m, 1000 - v[1] * px_per_m]) for k, v in lm.items()}


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("px_per_m", [1.0, 250.0, 911.3])
def test_image_coords_flip_and_scale(mirror, px_per_m):
    """x 반전(왼쪽으로 차는 영상)·y 반전(이미지 좌표)·스케일 변경 후에도 같은 각도."""
    for ang in random_poses(200, seed=4):
        lm = landmarks_from_fk(pose(ang, np.zeros(2), H))
        body = to_body_frame(to_image(lm, px_per_m, mirror), flip_x=mirror, px_per_m=px_per_m)
        assert max_err(angles_from_pose(body), ang) < TOL_DEG


def test_without_flip_angles_differ():
    """반대로 확인: 왼쪽으로 차는 영상을 반전하지 않으면 각도가 틀어진다 (정규화가 필요함을 확인)."""
    ang = random_poses(1, seed=5)[0]
    lm = landmarks_from_fk(pose(ang, np.zeros(2), H))
    body = to_body_frame(to_image(lm, 300.0, mirror=True), flip_x=False, px_per_m=300.0)
    assert max_err(angles_from_pose(body), ang) > 1.0


@pytest.mark.parametrize("scale", [0.01, 1.0, 523.7])
def test_scale_invariance(scale):
    """좌표 스케일(픽셀/미터)이 바뀌어도 각도는 같다."""
    for ang in random_poses(100, seed=3):
        lm = {k: v * scale for k, v in landmarks_from_fk(pose(ang, np.zeros(2), H)).items()}
        assert max_err(angles_from_pose(lm), ang) < TOL_DEG
