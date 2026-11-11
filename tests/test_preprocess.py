import numpy as np
import pytest

from biomech.kick_loader import available_kicks, load_kick
from biomech.kinematics import ANGLE_KEYS
from video.angles_from_pose import wrap180
from video.preprocess import fill_gaps, long_gap_mask, preprocess
from video.synthetic import synthesize, template_frames

H = 1.75
FPS = 120
NO_FILTER = 1e6     # 차단 주파수를 나이퀴스트보다 크게 → 필터 생략


def angle_err(a: dict, b: dict, keys=ANGLE_KEYS) -> float:
    return max(float(np.max(np.abs(wrap180(a[k] - b[k])))) for k in keys)


@pytest.mark.parametrize("kick_id", available_kicks())
def test_orientation_and_side_normalization(kick_id):
    """왼쪽으로 차는 영상·왼발 킥·다른 스케일이어도 정규화 후 같은 각도가 나온다."""
    kick = load_kick(kick_id)
    base = preprocess(synthesize(kick, H, FPS), H, FPS)
    assert base.kick_side == "R" and not base.flipped_x
    for opts in (dict(mirror=True), dict(left_footed=True), dict(left_footed=True, mirror=True),
                 dict(px_per_m=900.0)):
        p = preprocess(synthesize(kick, H, FPS, **opts), H, FPS)
        assert p.kick_side == ("L" if opts.get("left_footed") else "R")
        assert p.flipped_x == bool(opts.get("mirror"))
        assert angle_err(p.angles, base.angles) < 1e-6
        if opts.get("left_footed"):
            assert any("왼발 킥으로 인식" in w for _, w in p.warnings)


@pytest.mark.parametrize("kick_id", available_kicks())
def test_unfiltered_matches_template(kick_id):
    """필터 없이 합성 → 전처리하면 다리 각도가 템플릿과 일치한다.

    trunk(와 trunk를 쓰는 고관절·어깨)는 kick_hip_lift(오른쪽 고관절만 올림) 때문에 고관절 중점이
    달라져 최대 약 2° 차이가 난다.
    """
    kick = load_kick(kick_id)
    _, idx, m = template_frames(kick, H, FPS)
    truth = {k: m.angles[k][idx] for k in ANGLE_KEYS}
    p = preprocess(synthesize(kick, H, FPS), H, FPS, cutoff_hz=NO_FILTER)
    assert p.cutoff_hz is None
    assert angle_err(p.angles, truth, ["r_knee", "l_knee", "r_ankle", "l_ankle", "r_elbow", "l_elbow"]) < 1e-6
    assert angle_err(p.angles, truth) < 2.0


@pytest.mark.parametrize("px_per_m", [200.0, 380.0, 700.0])
def test_scale_estimate(px_per_m):
    """키로부터 픽셀 → m 스케일을 1% 이내로 추정한다."""
    p = preprocess(synthesize(load_kick("instep"), H, FPS, px_per_m=px_per_m), H, FPS)
    assert p.px_per_m == pytest.approx(px_per_m, rel=0.01)


def test_ground_is_zero():
    """지면 기준선이 y ≈ 0이 되도록 이동한다."""
    p = preprocess(synthesize(load_kick("instep"), H, FPS), H, FPS, cutoff_hz=NO_FILTER)
    feet = np.concatenate([p.xy[k][:, 1] for k in ("l_heel", "l_toe")])
    assert np.quantile(feet, 0.05) == pytest.approx(0.0, abs=1e-9)


def test_filter_reduces_noise():
    """노이즈(σ = 키의 0.5%) 성분의 RMS가 Butterworth 필터 후 절반 이하로 준다.

    (필터 자체의 평활 오차를 빼기 위해 같은 필터를 거친 깨끗한 데이터와 비교한다.)
    """
    kick = load_kick("instep")
    noisy = synthesize(kick, H, FPS, noise_frac=0.005, seed=7)
    rms = lambda a, b: float(np.sqrt(np.mean(wrap180(a - b) ** 2)))
    raw_err = rms(preprocess(noisy, H, FPS, cutoff_hz=NO_FILTER).angles["r_knee"],
                  preprocess(synthesize(kick, H, FPS), H, FPS, cutoff_hz=NO_FILTER).angles["r_knee"])
    filt_err = rms(preprocess(noisy, H, FPS).angles["r_knee"],
                   preprocess(synthesize(kick, H, FPS), H, FPS).angles["r_knee"])
    assert filt_err < raw_err / 2


def test_fill_gaps_and_long_gap_mask():
    x = np.arange(20, dtype=float)[:, None] * np.array([[1.0, 2.0]])
    valid = np.ones(20, dtype=bool)
    valid[3:6] = False          # 3프레임 결측 → 보간 신뢰
    valid[10:17] = False        # 7프레임 결측 → 신뢰 불가
    filled = fill_gaps(np.where(valid[:, None], x, np.nan), valid)
    assert np.allclose(filled, x)
    mask = long_gap_mask(valid)
    assert not mask[3:6].any() and mask[10:17].all() and mask.sum() == 7


def test_low_visibility_marked():
    """긴 저신뢰 구간은 visibility가 0.5 미만으로 표시된다."""
    raw = synthesize(load_kick("instep"), H, FPS)
    raw.vis["r_ankle"][20:30] = 0.1
    p = preprocess(raw, H, FPS)
    assert (p.vis["r_ankle"][20:30] < 0.5).all() and (p.vis["r_ankle"][:20] >= 0.5).all()


def test_quality_warnings():
    kick = load_kick("instep")
    # fps 경고
    p = preprocess(synthesize(kick, H, 30), H, 30)
    assert any("60fps 미만" in w for _, w in p.warnings)
    p = preprocess(synthesize(kick, H, FPS), H, FPS)
    assert not any(lvl == "warning" for lvl, _ in p.warnings)
    # 정면 촬영: 양 어깨 x 간격이 몸통 길이의 절반
    raw = synthesize(kick, H, FPS)
    raw.xy["l_shoulder"] = raw.xy["l_shoulder"] + np.array([0.5 * 0.288 * H * 380, 0])
    assert any("정면/사선" in w for _, w in preprocess(raw, H, FPS).warnings)
    # 화면 이탈: 영상 폭을 줄여 몸 일부가 밖에 있게 함
    raw = synthesize(kick, H, FPS)
    raw.width = int(np.median(raw.xy["r_toe"][:, 0]))
    assert any("화면 밖" in w for _, w in preprocess(raw, H, FPS).warnings)
