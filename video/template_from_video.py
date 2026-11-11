"""숙련자 측면 영상 → 킥 템플릿(키프레임 JSON).

영상 분석 파이프라인(전처리 → 단계 검출)으로 얻은 6개 단계 프레임의 관절 각도를 키프레임으로 쓴다.
측면 2D 모델에 맞추기 위한 두 가지 보정만 한다 (나머지는 실측 그대로):
  1) 디딤 이후 디딤발 발목: 발바닥이 지면에 평평하게 붙도록 계산 (측면 영상의 발 랜드마크는 흔들림이 큼)
  2) kick_hip_lift: 공 접촉 높이·차는 발 지면 통과 조건을 맞추도록 자동 보정 (biomech.template_fit)
"""

import copy
from datetime import date

import numpy as np

from biomech.kick_loader import PHASES, validate_kick
from biomech.kinematics import ANGLE_KEYS
from biomech.template_fit import check_kick, fit_hip_lift
from video.events import Events
from video.pose_series import Processed

STANCE_FOOT_TILT = 2        # 디딤발 발끝을 살짝 든 각도 (°)
AIR_BALL_HEIGHT = 0.13      # 기존 템플릿 접촉 높이가 이보다 크면 공중의 공 (지면 공 ≈ 0.11 m)


def flat_stance_ankle(kf: dict) -> int:
    """디딤발 발바닥이 평평해지는 발목 각도: 발 절대각 = 정강이 절대각 − 저측굴곡 ≈ +2°."""
    return round(kf["l_hip"] - kf["trunk"] - kf["l_knee"] - STANCE_FOOT_TILT)


def keyframes_from_video(p: Processed, ev: Events) -> list[dict]:
    """6개 단계 프레임의 각도·시각 (접근 = 0 s). 검출 실패 단계가 있으면 ValueError."""
    if p.view != "side":
        raise ValueError("템플릿은 측면 영상으로만 만들 수 있습니다 (후면은 시상면 각도가 보이지 않음).")
    missing = [ph for ph in PHASES if ev.frames.get(ph) is None]
    if missing:
        raise ValueError(f"검출하지 못한 단계가 있습니다: {', '.join(missing)}. 단계 직접 지정으로 채운 뒤 다시 만드세요.")
    frames = [int(ev.frames[ph]) for ph in PHASES]
    t0 = float(p.t[frames[0]])
    out, last_t = [], -1.0
    for ph, f in zip(PHASES, frames):
        t = round(float(p.t[f]) - t0, 3)
        if t <= last_t:                       # 같은 프레임에 두 단계가 잡히면 한 프레임씩 뒤로
            t = round(last_t + 1 / p.fps, 3)
        last_t = t
        kf = {"t": t, "phase": ph, **{k: int(round(float(p.angles[k][f]))) for k in ANGLE_KEYS}}
        out.append(kf)
    return out


def build_template(kfs_measured: list[dict], base: dict, source: str) -> tuple[dict, dict]:
    """측정 키프레임 + 기존 템플릿(이름·임팩트 정보·설명 문구) → 새 템플릿과 검사 결과."""
    kick = copy.deepcopy(base)
    by_phase = {kf["phase"]: kf for kf in kick["keyframes"]}
    new = []
    for n, m in enumerate(kfs_measured):
        old = by_phase.get(m["phase"], kick["keyframes"][min(n, len(kick["keyframes"]) - 1)])
        kf = {**old, **m}
        if n >= 1 and kick["stance_pin"] == "L":
            kf["l_ankle"] = flat_stance_ankle(kf)
        new.append(kf)
    kick["keyframes"] = new
    i_imp = PHASES.index("임팩트")
    kick["impact"]["time"] = new[i_imp]["t"]
    kick, check = fit_hip_lift(kick)
    if base["impact"]["contact_height"] > AIR_BALL_HEIGHT and \
            abs(check["contact_height"] - kick["impact"]["contact_height"]) > 0.03:
        # 공중의 공(발리·하프 발리)은 영상마다 공 높이가 다르므로 이 영상의 접촉 높이를 쓴다
        kick["impact"]["contact_height"] = round(check["contact_height"], 2)
    kfs = kick["keyframes"]
    kfs[-1]["kick_hip_lift"] = max(kfs[-1]["kick_hip_lift"], kfs[i_imp]["kick_hip_lift"])  # 팔로스루 동안 골반 높이 유지
    kick["evidence"] = f"숙련자 영상 실측 ({source})"
    kick["sources"] = [f"측면 영상 {source} · MediaPipe 관절 좌표 → 같은 각도 정의(4.3) · {date.today().isoformat()} 생성",
                       *[s for s in base.get("sources", []) if not s.startswith("측면 영상")]]
    validate_kick(kick)
    return kick, check_kick(kick)


def average_keyframes(runs: list[list[dict]]) -> list[dict]:
    """여러 영상(같은 킥)의 측정 키프레임을 단계별로 합친다 (시각·각도 모두).

    단계(디딤·백스윙·가속·임팩트…)끼리 맞춰 합치므로 영상마다 동작 속도가 달라도 시간 정규화가 된다.
    영상이 3개 이상이면 평균 대신 중앙값을 쓴다: 한 영상에서 단계 검출이나 관절 좌표가 틀려도 결과가 끌려가지 않는다.
    """
    agg = np.median if len(runs) >= 3 else np.mean
    out = []
    for i, ph in enumerate(PHASES):
        rows = [r[i] for r in runs]
        kf = {"phase": ph, "t": round(float(agg([r["t"] for r in rows])), 3)}
        kf.update({k: int(round(float(agg([r[k] for r in rows])))) for k in ANGLE_KEYS})
        out.append(kf)
    # 중앙값은 단계마다 다른 영상에서 올 수 있으므로 시각이 거꾸로 되지 않게 한 프레임(1 ms)씩 민다
    for a, b in zip(out, out[1:]):
        if b["t"] <= a["t"]:
            b["t"] = round(a["t"] + 0.001, 3)
    return out


def problems(kick: dict, check: dict) -> list[str]:
    """테스트(test_kicks.py) 기준으로 남은 문제를 사람이 읽을 문장으로."""
    out = []
    if not kick["airborne"]:
        if check["kick_foot_min"] < 0:
            out.append(f"차는 발이 지면 아래로 {-check['kick_foot_min'] * 100:.1f} cm 내려갑니다 "
                       f"(t = {check['kick_foot_min_t']:.3f} s). 엉덩이 들림 보정 상한에 걸렸습니다.")
        if min(check["foot_min"]["l_toe"], check["foot_min"]["l_heel"]) < -1e-6:
            out.append("디딤발이 지면 아래로 내려갑니다.")
    if abs(check["contact_height"] - kick["impact"]["contact_height"]) > 0.03:
        out.append(f"임팩트 접촉 높이 {check['contact_height']:.2f} m가 공 높이 "
                   f"{kick['impact']['contact_height']:.2f} m와 3 cm 넘게 다릅니다.")
    if not check["t_thigh_peak"] < check["t_shank_peak"] <= kick["impact"]["time"] + 1e-9:
        out.append("허벅지 → 정강이 순서로 최고 각속도에 도달하지 않습니다 (근위→원위 순차).")
    return out
