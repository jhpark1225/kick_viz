"""숙련자 측면 킥 영상 → data/kicks/<kick_id>.json 템플릿.

사용 예:
  .venv311\\Scripts\\python scripts\\template_from_video.py instep 영상1.mp4 영상2.mp4 --height 178 --fps 240
  (확인 후) 같은 명령에 --write를 붙이면 기존 JSON을 data/_backup/kicks/에 백업하고 덮어쓴 뒤
  피드백 기준(data/feedback_rules.json)을 다시 만든다.
영상이 여러 개면 단계별 각도·시각을 평균한다. 앱 탭 5에서 만든 JSON을 받아 넣어도 된다 (--json).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biomech.kick_loader import KICKS_DIR, PHASES, load_kick, validate_kick  # noqa: E402
from video.events import detect_events  # noqa: E402
from video.pose_extract import extract_from_bytes, find_model  # noqa: E402
from video.preprocess import preprocess  # noqa: E402
from video.template_from_video import (average_keyframes, build_template, keyframes_from_video,  # noqa: E402
                                       problems)
from biomech.template_fit import check_kick  # noqa: E402


def measure(path: Path, height_m: float, fps: float | None) -> list[dict]:
    model = find_model()
    if model is None:
        sys.exit("models/ 폴더에 pose_landmarker_full.task가 없습니다.")
    raw, _ = extract_from_bytes(path.read_bytes(), path.name, model)
    p = preprocess(raw, height_m, fps or raw.meta_fps, view="side")
    for level, msg in p.warnings:
        print(f"  [{level}] {msg}")
    return keyframes_from_video(p, detect_events(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kick_id")
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--height", type=float, default=175, help="선수 키 (cm)")
    ap.add_argument("--fps", type=float, default=None, help="실제 촬영 fps (슬로모션이면 꼭 입력)")
    ap.add_argument("--json", type=Path, help="앱 탭 5에서 내려받은 템플릿 JSON을 그대로 반영")
    ap.add_argument("--write", action="store_true", help="data/kicks에 저장하고 피드백 기준 재생성")
    a = ap.parse_args()

    base = load_kick(a.kick_id)
    if a.json:
        kick = json.loads(a.json.read_text(encoding="utf-8"))
        validate_kick(kick)
        check = check_kick(kick)
    else:
        if not a.videos:
            sys.exit("영상 파일 또는 --json을 주세요.")
        runs = []
        for v in a.videos:
            print(f"분석: {v.name}")
            runs.append(measure(v, a.height / 100, a.fps))
        kfs = average_keyframes(runs) if len(runs) > 1 else runs[0]
        src = ", ".join(v.name for v in a.videos) + f" (키 {a.height:.0f} cm" + (f", {a.fps:.0f} fps)" if a.fps else ")")
        kick, check = build_template(kfs, base, src)

    print(f"\n{'단계':6s} {'t':>6s} {'몸통':>5s} {'고관절':>6s} {'무릎':>5s} {'발목':>5s} {'디딤무릎':>8s}  (기존 템플릿)")
    old = {kf["phase"]: kf for kf in base["keyframes"]}
    for kf in kick["keyframes"]:
        o = old.get(kf["phase"], {})
        print(f"{kf['phase']:6s} {kf['t']:6.3f} {kf['trunk']:5d} {kf['r_hip']:6d} {kf['r_knee']:5d} {kf['r_ankle']:5d} "
              f"{kf['l_knee']:8d}  ({o.get('t', 0):.2f}, {o.get('r_hip')}, {o.get('r_knee')}, {o.get('r_ankle')})")
    print(f"정강이 최고 각속도 {check['shank_peak']:.0f}°/s, 허벅지 {check['thigh_peak']:.0f}°/s")
    issues = problems(kick, check)
    print("\n남은 문제:" if issues else "\n모델 조건 통과 ✅", *[f"- {s}" for s in issues], sep="\n")

    if a.write:
        blocking = [s for s in issues if "지면" in s or "접촉 높이" in s or (a.kick_id == "instep" and "순차" in s)]
        if blocking:
            sys.exit("모델 조건을 통과하지 못해 저장하지 않았습니다.")
        dst = KICKS_DIR / f"{a.kick_id}.json"
        bak = ROOT / "data" / "_backup" / "kicks"
        bak.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, bak / f"{a.kick_id}.json")
        dst.write_text(json.dumps(kick, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"저장: {dst} (기존 파일은 {bak}에 백업)")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_feedback_rules.py")], check=True)


if __name__ == "__main__":
    main()
