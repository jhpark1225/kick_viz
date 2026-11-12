"""측면 킥 영상 여러 개로 분석 파이프라인 성능을 평가한다 (관절 좌표 정리 전후 비교).

앱과 같은 흐름으로 평가한다: 파일 이름의 첫 단어를 **사용자가 고른 킥**으로 보고, 그 킥 기준으로
단계 검출 → 피드백을 만든다. 킥 이름: 인스텝·인사이드·아웃사이드·감아차기·로빙·발리·토킥·하프발리
(예: 인스텝_측면.mp4, 로빙_2.mp4). 이름으로 킥을 알 수 없거나 후면 영상(이름에 '후면')은 건너뛴다.

사용 예:
  .venv311\\Scripts\\python scripts\\evaluate_videos.py C:\\Users\\...\\킥모음 --height 175
  .venv311\\Scripts\\python scripts\\evaluate_videos.py 킥모음 --fps 인스텝_슬로우_측면=240 --classify
결과: 영상별 표 + 요약을 화면에 출력하고, --csv로 저장할 수 있다.

지표 (정리 전 = clean 끔, 정리 후 = clean 켬):
  - 차는 발: 오른발(R)/왼발(L) 판정. 촬영한 선수의 실제 차는 발과 비교한다.
  - 단계 실패: 6개 단계(접근~팔로스루) 중 검출하지 못한 개수. 임팩트 프레임, 디딤→임팩트 시간(s).
  - 개선점 Top 3: 고른 킥 기준 피드백의 우선 개선점 이름 (임팩트를 못 찾으면 비움).
  - 측정 불확실: 피드백 항목 중 '측정 불확실'(fps < 60 각속도, 낮은 visibility, 단계 검출 실패 등) 개수.
  - 각도 떨림: 다리·몸통 각도의 프레임 간 2차 차분 RMS (°). 작을수록 매끄럽다.
    (fps가 다르면 값의 크기가 달라지므로 같은 영상의 전후 비교에만 쓴다.)
  - 좌우 보정·스파이크: 정리 단계가 바로잡은 프레임·점 수.
  - (--classify) 분류 1위·신뢰도·Top-1/Top-3: 참고용 자동 분류(동작 모양 비교)가 고른 킥을 맞히는지.
    앱에서는 피드백 기준을 정하지 않고 불일치 안내에만 쓴다.
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
from video.feedback import UNSURE, compute_metrics, make_feedback  # noqa: E402
from video.pose_extract import extract_from_bytes, find_model  # noqa: E402
from video.preprocess import preprocess  # noqa: E402

TRUTH = {"인스텝": "instep", "인사이드": "inside", "아웃사이드": "outside", "감아차기": "curve",
         "로빙": "chip", "발리": "volley", "토킥": "toe", "하프발리": "half_volley"}
JITTER_KEYS = ("r_hip", "r_knee", "l_hip", "l_knee", "trunk")


def jitter(p) -> float:
    """각도 떨림: 2차 차분 RMS 평균 (°/프레임²)."""
    return float(np.mean([np.sqrt(np.mean(np.diff(p.angles[k], 2) ** 2)) for k in JITTER_KEYS]))


def run_one(raw, H: float, fps: float, clean: bool, kick: str, with_classify: bool = False) -> dict:
    """고른 킥(kick) 기준으로 단계 검출 + 피드백. with_classify면 참고용 분류 결과도 붙인다."""
    p = preprocess(raw, H, fps, view="side", clean=clean)
    ev = detect_events(p)
    rep = p.clean_report
    row = {
        "차는발": p.kick_side,
        "단계실패": len(ev.failed),
        "임팩트프레임": ev.impact,
        "디딤→임팩트(s)": None if ev.time("디딤") is None or ev.impact is None
        else round(ev.time("임팩트") - ev.time("디딤"), 3),
        "개선점Top3": "",
        "측정불확실": None,
        "각도떨림": round(jitter(p), 3),
        "좌우보정": rep.total_swapped if rep else 0,
        "스파이크": rep.spikes if rep else 0,
    }
    if ev.impact is not None:
        fb = make_feedback(kick, compute_metrics(p, ev))
        row["개선점Top3"] = " / ".join(j.name for j in fb.top3)
        row["측정불확실"] = sum(j.status == UNSURE for j in fb.items)
    if with_classify:
        row.update({"분류1위": None, "신뢰도": None, "Top1": None, "Top3": None})
        if ev.impact is not None:
            c = classify(p, ev)
            rank = [k for k, _ in c.ranking]
            row.update({"분류1위": rank[0], "신뢰도": round(c.ranking[0][1], 2),
                        "Top1": rank[0] == kick, "Top3": kick in rank[:3]})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--height", type=float, default=175, help="선수 키 (cm)")
    ap.add_argument("--fps", action="append", default=[], metavar="이름=fps",
                    help="영상별 실제 촬영 fps (슬로모션). 예: 인스텝_슬로우_측면=240. 없으면 메타데이터 fps")
    ap.add_argument("--classify", action="store_true",
                    help="참고용 자동 분류(동작 모양 비교)의 1위·Top-1/Top-3도 출력")
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
        kick = TRUTH.get(v.stem.split("_")[0])
        if kick is None:
            print(f"건너뜀: {v.name} (파일 이름 첫 단어로 킥 종류를 알 수 없음)", flush=True)
            continue
        print(f"분석: {v.name} (고른 킥: {kick})", flush=True)
        raw, _ = extract_from_bytes(v.read_bytes(), v.name, model)
        fps = fps_over.get(v.stem, raw.meta_fps)
        for clean in (False, True):
            r = run_one(raw, H, fps, clean, kick, a.classify)
            rows.append({"영상": v.stem, "fps": round(fps, 1), "정리": "후" if clean else "전", "고른킥": kick, **r})
    if not rows:
        sys.exit("평가할 영상이 없습니다.")

    keys = list(rows[0])
    print("\n" + " | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    print("\n요약 (정리 전 → 후)")
    for label in ("전", "후"):
        rs = [r for r in rows if r["정리"] == label]
        unsure = [r["측정불확실"] for r in rs if r["측정불확실"] is not None]
        line = (f"  정리 {label}: 차는발=R {sum(r['차는발'] == 'R' for r in rs)}/{len(rs)}, "
                f"단계 실패 합 {sum(r['단계실패'] for r in rs)}, "
                f"피드백 생성 {len(unsure)}/{len(rs)}, 측정 불확실 평균 {np.mean(unsure) if unsure else float('nan'):.1f}개, "
                f"각도 떨림 평균 {np.mean([r['각도떨림'] for r in rs]):.3f}")
        if a.classify:
            lab = [r for r in rs if r["Top1"] is not None]
            line += (f" | (참고) 분류 Top-1 {sum(r['Top1'] for r in lab)}/{len(lab)}, "
                     f"Top-3 {sum(r['Top3'] for r in lab)}/{len(lab)}")
        print(line)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 저장: {a.csv}")


if __name__ == "__main__":
    main()
