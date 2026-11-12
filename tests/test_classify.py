import numpy as np
import pytest

from biomech.kick_loader import available_kicks, load_kick
from video.classify import (GROUPS, Classification, classify, dtw_distance, find_group, mismatch_note, softmax_conf)
from video.preprocess import preprocess
from video.synthetic import synthesize

H = 1.75
FPS = 120
KICKS = available_kicks()


def run(kick_id: str, **opts):
    kick = load_kick(kick_id)
    return classify(preprocess(synthesize(kick, H, FPS, **opts), H, FPS))


def top3_ids(c) -> list[str]:
    return [k for k, _ in c.top3]


@pytest.mark.parametrize("kick_id", KICKS)
def test_template_top1_or_group(kick_id):
    """합성 데이터: 정답이 1위이거나, 구분 불가 그룹으로 표시되면 그 그룹에 정답이 포함된다."""
    c = run(kick_id)
    assert c.best == kick_id or (c.group is not None and kick_id in c.group_members), c.top3
    assert len(c.top3) == 3
    assert sum(conf for _, conf in c.ranking) == pytest.approx(1.0)
    assert 2 <= len(c.evidence) <= 3


@pytest.mark.parametrize("kick_id", KICKS)
@pytest.mark.parametrize("seed", [0, 1])
def test_noise_top3(kick_id, seed):
    """가우시안 노이즈(σ = 키의 0.5%)를 넣어도 정답이 상위 3위 안."""
    assert kick_id in top3_ids(run(kick_id, noise_frac=0.005, seed=seed))


@pytest.mark.parametrize("kick_id", KICKS)
@pytest.mark.parametrize("time_scale", [0.8, 1.2])
def test_playback_speed_top3(kick_id, time_scale):
    """동작 시간을 ±20% 늘이거나 줄여도 정답이 상위 3위 안 (DTW 검증)."""
    assert kick_id in top3_ids(run(kick_id, time_scale=time_scale))


def test_mirror_and_left_foot_same_result():
    """왼쪽으로 찬 영상·왼발 킥도 같은 분류."""
    base = run("instep")
    assert run("instep", mirror=True, left_footed=True).best == base.best


def test_dtw_basic():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 3))
    assert dtw_distance(a, a) == pytest.approx(0.0)
    # 같은 곡선을 시간만 늘인 경우 DTW 거리가 작다 (유클리드 비교보다 훨씬 작음)
    t = np.linspace(0, 1, 40)
    f = lambda tt: np.stack([np.sin(6 * tt), np.cos(4 * tt)], axis=1)
    slow = f(np.linspace(0, 1, 52))
    shifted = f(np.clip(t + 0.1, 0, 1))
    assert dtw_distance(f(t), slow) < 0.05 < dtw_distance(f(t), shifted) * 10


def test_group_display_rule():
    """같은 그룹 상위 후보의 신뢰도 차이 < 15%p면 그룹으로 표시, 아니면 표시하지 않음."""
    name, members = find_group([("instep", 0.45), ("curve", 0.35), ("chip", 0.2)])
    assert name is not None and members == GROUPS[name]
    assert find_group([("instep", 0.70), ("curve", 0.20), ("chip", 0.1)])[0] is None
    assert find_group([("inside", 0.40), ("outside", 0.38), ("toe", 0.22)])[0] is not None
    assert find_group([("inside", 0.40), ("toe", 0.38), ("outside", 0.22)])[0] is None


def test_softmax_monotonic():
    r = softmax_conf({"a": 0.1, "b": 0.3, "c": 0.2})
    assert [k for k, _ in r] == ["a", "c", "b"]


def _fake(ranking):
    """순위만 정한 분류 결과 (그룹 판정은 classify와 같은 규칙)."""
    group, members = find_group(ranking)
    return Classification(ranking=ranking, gate="지면 킥", candidates=[], evidence=[], group=group,
                          group_members=members)


def test_mismatch_note_same_kick():
    assert mismatch_note("instep", _fake([("instep", 0.8), ("toe", 0.1), ("chip", 0.1)])) is None


def test_mismatch_note_same_group():
    """인스텝·감아차기, 인사이드·아웃사이드는 측면으로 원래 구분이 어려워 안내하지 않는다."""
    assert mismatch_note("curve", _fake([("instep", 0.8), ("toe", 0.1), ("chip", 0.1)])) is None
    assert mismatch_note("outside", _fake([("inside", 0.8), ("toe", 0.1), ("chip", 0.1)])) is None


def test_mismatch_note_low_confidence():
    """1위 신뢰도가 낮거나(< 0.5) 그룹 표시면 안내하지 않는다."""
    assert mismatch_note("inside", _fake([("instep", 0.4), ("toe", 0.35), ("chip", 0.25)])) is None
    assert mismatch_note("chip", _fake([("instep", 0.5), ("curve", 0.4), ("toe", 0.1)])) is None


def test_mismatch_note_different_group_confident():
    note = mismatch_note("inside", _fake([("instep", 0.8), ("toe", 0.1), ("chip", 0.1)]))
    assert note is not None
    assert "인사이드 킥" in note and "인스텝 킥" in note and "참고용" in note


def test_mismatch_note_on_synthetic_video():
    """합성 인스텝 영상을 로빙으로 골랐을 때: 실제 분류 결과로도 같은 규칙이 적용된다."""
    c = run("instep")
    note = mismatch_note("chip", c)
    assert (note is None) == c.uncertain
    assert mismatch_note(c.best, c) is None

