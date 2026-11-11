"""실측 참조 데이터: 직접 찍은 킥 영상들의 특징(관절 각도 흐름)을 모아 분류 기준으로 쓴다.

내장 애니메이션(추정값) 대신 실제 사람이 찬 영상끼리 비교하면, 측정 편향(카메라 각도·MediaPipe·30 fps)이
참조와 입력에 똑같이 들어가 서로 상쇄된다. 영상 자체는 저장하지 않고 특징 시계열만 저장한다.
만들기: scripts/build_reference_set.py (영상 폴더 → 교차검증 → data/reference_kicks.json).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REF_PATH = Path(__file__).resolve().parent.parent / "data" / "reference_kicks.json"
LABELS = {"인스텝": "instep", "인사이드": "inside", "아웃사이드": "outside", "감아차기": "curve",
          "로빙": "chip", "칩": "chip", "발리": "volley", "토킥": "toe", "토": "toe", "하프발리": "half_volley"}
MIN_REFERENCE = 8           # 참조 예시가 이보다 적으면 내장 애니메이션과 비교 (너무 적으면 우연에 좌우됨)


@dataclass
class Example:
    """참조 예시 하나 (영상 1개)."""
    label: str              # 킥 id
    name: str               # 원본 파일 이름 (확장자 제외)
    seq: np.ndarray         # classify.feature_matrix 결과 (프레임 × 특징), 각도는 평균을 뺀 모양
    contact_height: float
    fps: float


def label_from_name(stem: str) -> str | None:
    """파일 이름 첫 단어 → 킥 id (예: '인스텝_민수_3' → 'instep', 'instep_2' → 'instep')."""
    s = stem.replace("-", "_").replace(" ", "_")
    for kid in sorted(set(LABELS.values()), key=len, reverse=True):     # 영문 id ('half_volley'처럼 '_' 포함 가능)
        if s == kid or s.startswith(kid + "_"):
            return kid
    return LABELS.get(s.split("_")[0])


def save_reference(examples: list[Example], meta: dict, path: Path = REF_PATH) -> None:
    data = {"meta": meta, "examples": [
        {"label": e.label, "name": e.name, "contact_height": round(e.contact_height, 4), "fps": e.fps,
         "seq": np.round(e.seq, 4).tolist()} for e in examples]}
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")


_cache: dict[tuple, list[Example]] = {}


def load_reference(path: Path | None = None) -> list[Example]:
    """참조 데이터 읽기 (없으면 빈 목록). 파일이 바뀌면 다시 읽는다."""
    path = path or REF_PATH
    if not path.exists():
        return []
    key = (str(path), path.stat().st_mtime)
    if key not in _cache:
        data = json.loads(path.read_text(encoding="utf-8"))
        _cache[key] = [Example(d["label"], d["name"], np.asarray(d["seq"], dtype=float),
                               float(d["contact_height"]), float(d["fps"])) for d in data["examples"]]
    return _cache[key]


def counts(examples: list[Example]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in examples:
        out[e.label] = out.get(e.label, 0) + 1
    return out
