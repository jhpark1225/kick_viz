"""직접 찍은 측면 킥 영상 폴더 → 실측 참조 데이터(data/reference_kicks.json) + 교차검증 정확도.

파일 이름 첫 단어 = 정답 킥: 인스텝·인사이드·아웃사이드·감아차기·로빙·발리·토킥·하프발리 (영문 id도 가능).
  예) 인스텝_민수_1.mp4, 인스텝_민수_2.mp4, 로빙_지훈_1.mov   ('후면'/'rear'가 들어간 영상은 건너뜀)
권장: 킥 종류마다 10개 이상 (최소 5개), 가능하면 2~3명, 슬로모션.

사용 예:
  .venv311\\Scripts\\python scripts\\build_reference_set.py C:\\Users\\...\\킥데이터 --height 175 --fps 240
  (결과를 보고) 같은 명령에 --write를 붙이면 data/reference_kicks.json으로 저장 → 앱 분류가 실제 영상 비교로 바뀜.

교차검증(leave-one-out): 영상 하나를 빼고 나머지만 참조로 써서 그 영상을 맞히는 것을 모든 영상에 반복한다.
같은 영상으로 기준을 만들고 평가하지 않으므로 앱에서 새 영상을 넣었을 때의 정확도에 가깝다.
같은 사람 영상이 참조에 남아 있으면 낙관적으로 나올 수 있다(--by-person: 이름 두 번째 단어 = 사람, 사람 단위로 뺌).
관절 좌표는 <폴더>\\.pose_cache에 저장해 두고 다시 쓴다 (영상은 복사하지 않음).
"""

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biomech.kick_loader import available_kicks  # noqa: E402
from video.classify import classify, feature_matrix  # noqa: E402
from video.events import detect_events  # noqa: E402
from video.pose_extract import extract_from_bytes, find_model  # noqa: E402
from video.preprocess import preprocess  # noqa: E402
from video.reference import MIN_REFERENCE, REF_PATH, Example, counts, label_from_name, save_reference  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi"}
REAR_WORDS = ("후면", "rear", "back")
SHORT = {"instep": "인스텝", "inside": "인사이드", "outside": "아웃사", "curve": "감아차", "chip": "로빙",
         "volley": "발리", "toe": "토킥", "half_volley": "하프발"}


def load_raw(v: Path, model, cache: Path):
    c = cache / f"{v.stem}.pkl"
    if c.exists():
        return pickle.loads(c.read_bytes())
    raw, _ = extract_from_bytes(v.read_bytes(), v.name, model)
    cache.mkdir(exist_ok=True)
    c.write_bytes(pickle.dumps(raw))
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--height", type=float, default=175, help="선수 키 (cm, 전체 공통)")
    ap.add_argument("--fps", default=None, help="실제 촬영 fps (전체 공통, 슬로모션이면 입력). 없으면 메타데이터")
    ap.add_argument("--fps-of", action="append", default=[], metavar="이름=fps", help="영상별 fps (예: 인스텝_1=240)")
    ap.add_argument("--by-person", action="store_true", help="이름 두 번째 단어를 사람으로 보고 사람 단위로 빼서 검증")
    ap.add_argument("--write", action="store_true", help=f"{REF_PATH.relative_to(ROOT)}에 저장")
    a = ap.parse_args()

    model = find_model()
    if model is None:
        sys.exit("models/ 폴더에 pose_landmarker_full.task가 없습니다.")
    fps_of = {k: float(v) for k, v in (s.split("=", 1) for s in a.fps_of)}
    H = a.height / 100
    items = []          # (Example, Processed, Events, person)
    for v in sorted(a.folder.iterdir()):
        if v.suffix.lower() not in VIDEO_EXT or any(w in v.stem.lower() for w in REAR_WORDS):
            continue
        label = label_from_name(v.stem)
        if label is None:
            print(f"건너뜀 (이름으로 킥 종류를 알 수 없음): {v.name}")
            continue
        raw = load_raw(v, model, a.folder / ".pose_cache")
        fps = fps_of.get(v.stem, float(a.fps) if a.fps else raw.meta_fps)
        try:
            p = preprocess(raw, H, fps, view="side")
        except ValueError as e:
            print(f"건너뜀 (전처리 실패: {e}): {v.name}")
            continue
        ev = detect_events(p)
        if ev.impact is None:
            print(f"건너뜀 (임팩트 검출 실패): {v.name}")
            continue
        person = v.stem.replace("-", "_").split("_")[1] if a.by_person and "_" in v.stem else v.stem
        ex = Example(label, v.stem, feature_matrix(p, p.t[ev.impact]), float(ev.contact_height), fps)
        items.append((ex, p, ev, person))
        print(f"분석: {v.name} → {label} ({fps:.0f} fps)", flush=True)

    if not items:
        sys.exit("분석할 영상이 없습니다.")
    examples = [it[0] for it in items]
    n_by = counts(examples)
    print("\n킥별 영상 수: " + ", ".join(f"{SHORT[k]} {n_by.get(k, 0)}" for k in available_kicks()))
    few = [SHORT[k] for k in available_kicks() if n_by.get(k, 0) < 5]
    if few:
        print(f"⚠️ 5개 미만인 킥: {', '.join(few)} — 교차검증에서 이 킥은 맞히기 어렵습니다 (1개뿐이면 뺐을 때 참조가 없음).")

    kicks = available_kicks()
    conf = {t: {q: 0 for q in kicks} for t in kicks}
    res = {"ref": [0, 0], "tpl": [0, 0]}
    print(f"\n{'영상':24s} {'정답':6s} | 실측 비교 1위 (상위3 안?) | 애니메이션 비교 1위")
    for ex, p, ev, person in items:
        others = [o for o, _, _, per in items if (per != person if a.by_person else o.name != ex.name)]
        c_ref = classify(p, ev, reference=others)
        c_tpl = classify(p, ev, reference=[])
        r3 = [k for k, _ in c_ref.top3]
        t3 = [k for k, _ in c_tpl.top3]
        res["ref"][0] += r3[0] == ex.label
        res["ref"][1] += ex.label in r3
        res["tpl"][0] += t3[0] == ex.label
        res["tpl"][1] += ex.label in t3
        conf[ex.label][r3[0]] += 1
        mark = "○" if r3[0] == ex.label else ("△" if ex.label in r3 else "×")
        print(f"{ex.name:24s} {SHORT[ex.label]:6s} | {mark} {SHORT[r3[0]]:6s} ({c_ref.ranking[0][1]:.0%})"
              f"{'' if len(others) >= MIN_REFERENCE else ' [참조 부족→애니메이션]'} | {SHORT[t3[0]]}")
    n = len(items)
    print(f"\n교차검증 결과 ({n}개)")
    print(f"  실측 비교:       1위 정답 {res['ref'][0]}/{n} ({res['ref'][0] / n:.0%}), 상위 3개 안 {res['ref'][1]}/{n}")
    print(f"  애니메이션 비교: 1위 정답 {res['tpl'][0]}/{n} ({res['tpl'][0] / n:.0%}), 상위 3개 안 {res['tpl'][1]}/{n}")
    print(f"  무작위로 찍으면 1위 정답 약 {1 / len(kicks):.0%}")
    used = [k for k in kicks if n_by.get(k)]
    print("\n혼동표 (행 = 정답, 열 = 실측 비교 1위)\n" + " " * 7 + " ".join(f"{SHORT[k]:>5s}" for k in used))
    for t in used:
        print(f"{SHORT[t]:>6s} " + " ".join(f"{conf[t][q]:5d}" for q in used))

    if a.write:
        meta = {"created": date.today().isoformat(), "n": n, "counts": n_by, "height_cm": a.height,
                "loo_top1": round(res["ref"][0] / n, 3), "loo_top3": round(res["ref"][1] / n, 3),
                "by_person": a.by_person}
        save_reference(examples, meta)
        print(f"\n저장: {REF_PATH} (영상은 저장하지 않고 관절 각도 흐름만 저장)")


if __name__ == "__main__":
    main()
