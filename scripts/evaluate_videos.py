"""측면 킥 영상 여러 개로 분석 파이프라인 성능을 평가한다 (관절 좌표 정리 전후 비교).

파일 이름의 첫 단어로 정답 킥을 읽는다: 인스텝·인사이드·아웃사이드·감아차기·로빙·발리·토킥·하프발리
(예: 인스텝_측면.mp4, 로빙_2.mp4). 후면 영상(이름에 '후면')은 건너뛴다.

사용 예:
  .venv311\\Scripts\\python scripts\\evaluate_videos.py C:\\Users\\...\\킥모음 --height 175
  .venv311\\Scripts\\python scripts\\evaluate_videos.py 킥모음 --fps 인스텝_슬로우_측면=240 --fps 인스텝_덜슬로우_측면=120
결과: 영상별 표 + 요약을 화면에 출력하고, --csv로 저장할 수 있다.

지표 (정리 전 = clean 끔, 정리 후 = clean 켬):
  - 차는 발: 오른발(R)/왼발(L) 판정. 촬영한 선수의 실제 차는 발과 비교한다.
  - 단계 실패: 6개 단계(접근~팔로스루) 중 검출하지 못한 개수.
  - 분류 Top-1 / Top-3: 자동 분류 1위(또는 상위 3개)에 정답 킥이 있는지.
  - 각도 떨림: 다리·몸통 각도의 프레임 간 2차 차분 RMS (°). 작을수록 매끄럽다.
    (fps가 다르면 값의 크기가 달라지므로 같은 영상의 전후 비교에만 쓴다.)
  - 좌우 보정·스파이크: 정리 단계가 바로잡은 프레임·점 수.
주의: 임계값을 이 평가 영상에 맞춰 조정하면 평가가 낙관적으로 나온다. 조정용 영상과 평가용 영상은 나눠 쓴다.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video.classify import classify  # noqa: E402
from video.events import detect_events  # noqa: E402
from video.pose_extract import extract_from_bytes, find_model  # noqa: E402
from video.preprocess import preprocess  # noqa: E402

TRUTH = {"인스텝": "instep", "인사이드": "inside", "아웃사이드": "outside", "감아차기": "curve",
         "로빙": "chip", "발리": "volley", "토킥": "toe", "하프발리": "half_volley"}
JITTER_KEYS = ("r_hip", "r_knee", "l_hip", "l_knee", "trunk")


def jitter(p) -> float:
    """각도 떨림: 2차 차분 RMS 평균 (°/프레임²)."""
    return float(np.mean([np.sqrt(np.mean(np.diff(p.angles[k], 2) ** 2)) for k in JITTER_KEYS]))


def run_one(raw, H: float, fps: float, clean: bool, truth: str | None) -> dict:
    p = preprocess(raw, H, fps, view="side", clean=clean)
    ev = detect_events(p)
    c = classify(p, ev)
    rank = [k for k, _ in c.ranking]
    rep = p.clean_report
    return {
        "차는발": p.kick_side,
        "단계실패": len(ev.failed),
        "임팩트프레임": ev.impact,
        "디딤→임팩트(s)": None if ev.time("디딤") is None or ev.impact is None
        else round(ev.time("임팩트") - ev.time("디딤"), 3),
        "분류1위": rank[0], "신뢰도": round(c.ranking[0][1], 2),
        "Top1": None if truth is None else rank[0] == truth,
        "Top3": None if truth is None else truth in rank[:3],
        "각도떨림": round(jitter(p), 3),
        "좌우보정": rep.total_swapped if rep else 0,
        "스파이크": rep.spikes if rep else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--height", type=float, default=175, help="선수 키 (cm)")
    ap.add_argument("--fps", action="append", default=[], metavar="이름=fps",
                    help="영상별 실제 촬영 fps (슬로모션). 예: 인스텝_슬로우_측면=240. 없으면 메타데이터 fps")
    ap.add_argument("--csv", type=Path, help="결과를 CSV로 저장")
    a = ap.parse_args()

    fps_over = {k: float(v) for k, v in (s.split("=", 1) for s in a.fps)}
    model = find_model()
    if model is None:
        sys.exit("models/ 폴더에 pose_landmarker_full.task가 없습니다.")
    H = a.height / 100
    videos = sorted(v for v in a.folder.iterdir()
                    if v.suffix.lower() in {".mp4", ".mov", ".avi"} and "후면" not in v.stem)
    rows = []
    for v in videos:
        truth = TRUTH.get(v.stem.split("_")[0])
        print(f"분석: {v.name} (정답: {truth or '알 수 없음'})", flush=True)
        raw, _ = extract_from_bytes(v.read_bytes(), v.name, model)
        fps = fps_over.get(v.stem, raw.meta_fps)
        for clean in (False, True):
            r = run_one(raw, H, fps, clean, truth)
            rows.append({"영상": v.stem, "fps": round(fps, 1), "정리": "후" if clean else "전", "정답": truth, **r})

    keys = list(rows[0])
    print("\n" + " | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    print("\n요약 (정리 전 → 후)")
    for label in ("전", "후"):
        rs = [r for r in rows if r["정리"] == label]
        lab = [r for r in rs if r["Top1"] is not None]
        print(f"  정리 {label}: 차는발=R {sum(r['차는발'] == 'R' for r in rs)}/{len(rs)}, "
              f"단계 실패 합 {sum(r['단계실패'] for r in rs)}, "
              f"Top-1 {sum(r['Top1'] for r in lab)}/{len(lab)}, Top-3 {sum(r['Top3'] for r in lab)}/{len(lab)}, "
              f"각도 떨림 평균 {np.mean([r['각도떨림'] for r in rs]):.3f}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 저장: {a.csv}")


if __name__ == "__main__":
    main()
