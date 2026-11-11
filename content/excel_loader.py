"""엑셀 정리 자료(축구킥_생체역학_정리.xlsx) → 킥별 생체역학 설명.

원본 엑셀은 강의 순서(주차) 기준으로 정리되어 있다. 화면에서는 순서 번호 대신 각 킥에 쓰이는
생체역학 **개념** 이름으로 보여주므로, 읽을 때 순서 번호를 개념 이름으로 바꾼다.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXCEL_NAMES = ["축구킥_생체역학_정리.xlsx", "축구_생체역학_정리.xlsx"]

# 원본 엑셀의 시트 이름 (파일 구조이므로 그대로 둔다)
SHEET_OVERVIEW = "1_킥종류_개요"
SHEET_CONCEPTS = "2_강의주차별_분석"
SHEET_PHASES = "3_킥_동작단계"
SHEET_MATRIX = "4_킥x주차_매트릭스"
HEADER_ROW = 3          # 헤더는 4행
KICK_COL = "킥 종류"
CONCEPT_COL = "개념"

# 원본의 순서 번호 → 개념 이름 (화면 표시 순서도 이 순서)
CONCEPTS: dict[str, str] = {
    "2": "용어·운동면 (관절 동작 용어)",
    "3": "근골격계 (주동근·길항근)",
    "4": "신경계 (근 활성 타이밍·SSC)",
    "5": "선운동학 기초 (변위·속도·가속도)",
    "6": "인체 선운동학 (접근·발 속도)",
    "7": "포물체 운동 (공 궤적)",
    "9": "선운동역학 (충격량·지면반력)",
    "10": "각운동학 (관절 각속도·근위→원위 순차)",
    "11": "각운동역학 (관성모멘트·각운동량)",
    "12": "부상 위험 (헬스케어)",
}
_NUM = re.compile(r"(\d+)\s*주차")


def concept_name(label: str) -> str:
    """원본의 순서 표기(예: '11주차')를 개념 이름으로 바꾼다."""
    m = _NUM.search(str(label))
    return CONCEPTS.get(m.group(1), str(label)) if m else str(label)


def strip_order(text: str) -> str:
    """'5·6주차 (선운동학)' → '선운동학': 순서 번호를 지우고 남은 괄호를 정리한다.

    번호만 있으면('3·4·11주차') 개념 이름의 앞부분으로 바꾼다 → '근골격계·신경계·각운동역학'.
    """
    text = str(text)
    s = re.sub(r"\d+(?:·\d+)*\s*주차\s*", "", text).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s:
        return s
    m = re.search(r"(\d+(?:·\d+)*)\s*주차", text)
    if not m:
        return text
    names = [CONCEPTS[n].split(" (")[0] for n in m.group(1).split("·") if n in CONCEPTS]
    return "·".join(dict.fromkeys(names))


def excel_path() -> Path | None:
    """data 폴더의 엑셀 파일 경로. 없으면 None."""
    for name in EXCEL_NAMES:
        p = DATA_DIR / name
        if p.exists():
            return p
    return None


@lru_cache(maxsize=4)
def _workbook(path: str, mtime: float) -> dict[str, pd.DataFrame]:
    """엑셀의 모든 시트를 한 번만 읽는다 (파일이 바뀌면 mtime이 달라져 다시 읽음). 킥마다 읽으면 약 1.7 s."""
    sheets = pd.read_excel(path, sheet_name=None, header=HEADER_ROW)
    out = {}
    for name, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        out[name] = df.dropna(how="all")
    return out


def _read(sheet: str, path: Path) -> pd.DataFrame:
    return _workbook(str(path), path.stat().st_mtime)[sheet].copy()


def load_kick_content(excel_name: str, path: Path | None = None) -> dict | None:
    """킥 하나의 개요·개념별 분석·개념 매트릭스와 공통 동작단계 표를 읽는다. 파일이 없으면 None."""
    path = path or excel_path()
    if path is None:
        return None
    overview = _read(SHEET_OVERVIEW, path)
    overview = overview[overview[KICK_COL].astype(str).str.startswith(excel_name)]

    df = _read(SHEET_CONCEPTS, path)
    df = df[df[KICK_COL] == excel_name].drop(columns=[KICK_COL])
    order = df["주차"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df = df.assign(_o=order).sort_values("_o")
    df.insert(0, CONCEPT_COL, df["주차"].map(concept_name))
    concepts = df.drop(columns=[c for c in ("주차", "강의 주제", "_o") if c in df.columns])

    phases = _read(SHEET_PHASES, path)
    if "관련 강의 주차" in phases.columns:
        phases = phases.rename(columns={"관련 강의 주차": "관련 개념"})
        phases["관련 개념"] = phases["관련 개념"].map(lambda v: strip_order(v) if pd.notna(v) else v)

    matrix = None
    if SHEET_MATRIX in _workbook(str(path), path.stat().st_mtime):
        m = _read(SHEET_MATRIX, path)
        m.columns = [strip_order(c.replace("\n", " ")) if "주차" in c else c for c in m.columns]
        m = m.rename(columns={"관련 주차 수": "관련 개념 수"})
        matrix = m[m[KICK_COL] == excel_name].drop(columns=[KICK_COL])
        # 개수 열은 엑셀 수식(COUNTIF)이라 파이썬으로 수정·저장한 파일에는 계산값이 없을 수 있다 → 직접 센다
        marks = [c for c in matrix.columns if c != "관련 개념 수"]
        matrix["관련 개념 수"] = (matrix[marks] == "●").sum(axis=1)
    return {
        "overview": overview.iloc[0].to_dict() if len(overview) else None,
        "concepts": concepts.fillna("").reset_index(drop=True),
        "phases": phases.fillna("").reset_index(drop=True),
        "matrix": matrix,
    }


def concept_row(content: dict | None, prefix: str) -> dict | None:
    """개념 이름이 prefix로 시작하는 행 (예: '근골격계')."""
    if content is None:
        return None
    df = content["concepts"]
    rows = df[df[CONCEPT_COL].str.startswith(prefix)]
    return rows.iloc[0].to_dict() if len(rows) else None


def phase_supplement(phases: pd.DataFrame, phase_ko: str) -> dict | None:
    """3_킥_동작단계 시트에서 단계 이름(예: '백스윙')이 들어간 행."""
    rows = phases[phases["동작 단계"].astype(str).str.contains(phase_ko)]
    return rows.iloc[0].to_dict() if len(rows) else None
