"""측면 + 후면 함께 분석: 따로 찍은 두 영상(같은 사람, 같은 킥 종류)을 짝지어 하나의 결과로 합친다.

두 영상은 서로 다른 킥이므로 프레임 단위로 합치지 않는다. 킥 종류는 사용자가 고른 킥 하나로 정하고,
측면 지표(시상면)와 후면 지표(관상면)를 한 리포트로 합친다.
후면 정보는 동작 모양 비교(참고용 자동 분류)에 쓰지 않는다: 실제 후면 영상에서 발 방향(인사이드 = 발끝 바깥, 아웃사이드 = 안쪽)이
기대와 반대로 측정되고 후면 임팩트 검출도 흔들려(2026-11-12, 킥모음 후면 7개), 넣으면 분류가 더 나빠진다.
"""

import re
from dataclasses import replace

from video.feedback import Feedback

VIEW_WORDS = re.compile(r"[_\-\s]*(측면|후면|옆|뒤|side|rear|back)[_\-\s]*", re.IGNORECASE)
BOTH_NOT_MEASURED = ["골반·몸통 회전, 고관절 내·외회전 (수평면 동작 — 위에서 촬영해야 보임)",
                     "공의 회전과 발의 어느 부위로 맞혔는지 (포즈만으로는 확인 어려움)"]
SEPARATE_KICKS_NOTE = ("측면과 후면은 **서로 다른 두 번의 킥**을 각각 분석한 결과입니다. 두 킥의 동작이 조금 다를 수 있습니다.")


def pair_key(name: str) -> str:
    """파일 이름에서 확장자와 촬영 방향 단어를 뺀 짝 이름 (예: '인스텝_측면.mp4' → '인스텝')."""
    stem = name.rsplit(".", 1)[0]
    return VIEW_WORDS.sub("_", stem).strip("_ ").lower()


def auto_pairs(items: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """{id: (파일 이름, 'side'|'rear')} → 이름이 같은 (측면 id, 후면 id) 짝 목록 (올린 순서)."""
    sides = {i: pair_key(n) for i, (n, v) in items.items() if v == "side"}
    rears = {i: pair_key(n) for i, (n, v) in items.items() if v == "rear"}
    out, used = [], set()
    for si, sk in sides.items():
        for ri, rk in rears.items():
            if ri not in used and rk == sk:
                out.append((si, ri))
                used.add(ri)
                break
    return out


def merge_feedback(side: Feedback, rear: Feedback) -> Feedback:
    """두 피드백을 하나로: 각 항목 규칙에 촬영 방향('측면'/'후면')을 붙인다. Top 3·잘한 점은 두 영상을 합쳐 고른다."""
    tag = lambda fb, v: [replace(j, rule={**j.rule, "view": v}) for j in fb.items]
    return Feedback(side.kick_id, tag(side, "측면") + tag(rear, "후면"))
