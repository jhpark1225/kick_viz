"""실측 참조 데이터로 분류: 이름 → 정답, 저장·읽기, 참조가 있으면 실제 예시와 비교."""

import numpy as np
import pytest

from biomech.kick_loader import available_kicks, load_kick
from video.classify import classify, feature_matrix
from video.events import detect_events
from video.preprocess import preprocess
from video.reference import MIN_REFERENCE, Example, label_from_name, load_reference, save_reference
from video.synthetic import synthesize

H = 1.75


def processed(kick_id: str, seed: int, time_scale: float = 1.0, fps: float = 60):
    p = preprocess(synthesize(load_kick(kick_id), H, fps, noise_frac=0.008, seed=seed, time_scale=time_scale), H, fps)
    return p, detect_events(p)


def example(kick_id: str, seed: int, time_scale: float = 1.0) -> Example:
    p, ev = processed(kick_id, seed, time_scale)
    return Example(kick_id, f"{kick_id}_{seed}", feature_matrix(p, p.t[ev.impact]), float(ev.contact_height), p.fps)


def test_label_from_name():
    assert label_from_name("인스텝_민수_3") == "instep"
    assert label_from_name("하프발리-2") == "half_volley"
    assert label_from_name("toe_1") == "toe"
    assert label_from_name("모르는킥") is None


def test_save_load_round_trip(tmp_path):
    exs = [example("instep", 1), example("chip", 2)]
    path = tmp_path / "ref.json"
    save_reference(exs, {"n": 2}, path)
    back = load_reference(path)
    assert [e.label for e in back] == ["instep", "chip"]
    assert np.allclose(back[0].seq, exs[0].seq, atol=1e-3)


def test_missing_file_means_templates():
    """참조 파일이 없으면(테스트 기본) 내장 애니메이션과 비교한다."""
    p, ev = processed("instep", 1)
    assert classify(p, ev).method == "내장 애니메이션과 비교"


@pytest.fixture(scope="module")
def reference_set():
    """킥마다 '실제 영상'처럼 노이즈·속도가 다른 예시 2개 (속도 0.9배, 1.1배)."""
    return [example(k, s, ts) for k in available_kicks() for s, ts in ((11, 0.9), (12, 1.1))]


def test_reference_used_and_accurate(reference_set):
    """참조가 충분하면 실제 예시와 비교하고, 새 예시(다른 노이즈·속도)를 대부분 맞힌다."""
    assert len(reference_set) >= MIN_REFERENCE
    hits = 0
    for k in available_kicks():
        p, ev = processed(k, seed=21, time_scale=1.0)
        c = classify(p, ev, reference=reference_set)
        assert c.method.startswith("실제 영상")
        hits += c.best == k or (c.group is not None and k in c.group_members)
    assert hits >= len(available_kicks()) - 1


def test_too_few_references_fall_back(reference_set):
    p, ev = processed("instep", 1)
    c = classify(p, ev, reference=reference_set[:MIN_REFERENCE - 1])
    assert c.method == "내장 애니메이션과 비교"


def test_kick_without_reference_uses_template(reference_set):
    """참조에 없는 킥은 내장 애니메이션 거리로 순위에 남는다."""
    ref = [e for e in reference_set if e.label != "toe"]
    p, ev = processed("instep", 1)
    c = classify(p, ev, reference=ref)
    assert "toe" in c.distances and "내장 애니메이션" in c.method
