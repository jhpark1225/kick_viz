import pytest


@pytest.fixture(autouse=True)
def no_reference_file(monkeypatch, tmp_path):
    """테스트는 data/reference_kicks.json(실측 참조) 유무와 상관없이 같은 결과가 나와야 한다.

    참조가 필요한 테스트는 classify(..., reference=[...])로 직접 넘긴다.
    """
    import video.reference as ref
    monkeypatch.setattr(ref, "REF_PATH", tmp_path / "no_reference.json")
