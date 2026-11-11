"""킥 템플릿 보정: 키프레임 각도는 그대로 두고 `kick_hip_lift`만 조정해 모델 조건을 맞춘다.

선행연구 값이나 숙련자 영상에서 뽑은 각도는 측면 2D 모델(관상면 동작 없음)에 그대로 넣으면
차는 발이 땅속으로 들어가거나 임팩트 접촉 높이가 공 높이와 어긋난다. 실제로는 골반이 차는 쪽으로 들리고
몸이 디딤발 쪽으로 기울어 해결되는 문제이므로(4.5절), 그 보정값인 kick_hip_lift를 조정한다.
"""

import copy

import numpy as np

from biomech.kinematics import (angular_velocity, compute_motion, contact_point, frame_points,
                                segment_abs_angles)

H_CHECK = 1.75
N_CHECK = 240
CLEARANCE = 0.005          # 차는 발 최저점 여유 (m)
LIFT_MAX = 0.20            # 관상면 보정값 상한 (m)


def check_kick(kick: dict, H: float = H_CHECK, n: int = N_CHECK) -> dict:
    """test_kicks.py와 같은 조건을 계산한다: 발 최저 높이, 임팩트 접촉 높이, 근위→원위 순차, 각속도."""
    m = compute_motion(kick, H, n)
    imp = kick["impact"]
    i = int(np.argmin(np.abs(m.t - imp["time"])))
    lows = {k: float(m.points[k][:, 1].min()) for k in ("r_toe", "r_heel", "l_toe", "l_heel")}
    worst = min(("r_toe", "r_heel"), key=lambda k: lows[k])
    thigh, shank = segment_abs_angles(m.angles["trunk"], m.angles["r_hip"], m.angles["r_knee"])
    w_th, w_sh = angular_velocity(m.t, thigh), angular_velocity(m.t, shank)
    return {
        "foot_min": lows,
        "kick_foot_min": lows[worst],
        "kick_foot_min_t": float(m.t[np.argmin(m.points[worst][:, 1])]),
        "contact_height": float(contact_point(frame_points(m, i), imp["contact_point"])[1]),
        "t_thigh_peak": float(m.t[np.argmax(w_th)]),
        "t_shank_peak": float(m.t[np.argmax(w_sh)]),
        "shank_peak": float(w_sh.max()),
        "thigh_peak": float(w_th.max()),
    }


def fit_hip_lift(kick: dict, H: float = H_CHECK, iters: int = 40) -> tuple[dict, dict]:
    """kick_hip_lift를 조정한 사본과 최종 검사 결과를 돌려준다 (공중 동작은 그대로).

    1) 임팩트 키프레임: 접촉점 높이 = impact.contact_height가 되게 (접촉점 y는 lift에 선형).
    2) 차는 발이 지면 아래로 내려가는 시각에 가장 가까운 임팩트 외 키프레임의 lift를 올린다.
    """
    k = copy.deepcopy(kick)
    if k["airborne"]:
        return k, check_kick(k, H)
    kfs = k["keyframes"]
    t_imp = k["impact"]["time"]
    i_imp = int(np.argmin([abs(kf["t"] - t_imp) for kf in kfs]))
    for _ in range(iters):
        c = check_kick(k, H)
        err = k["impact"]["contact_height"] - c["contact_height"]
        if abs(err) > 0.002:
            kfs[i_imp]["kick_hip_lift"] = float(np.clip(kfs[i_imp]["kick_hip_lift"] + err, 0.0, LIFT_MAX))
            continue
        if c["kick_foot_min"] >= CLEARANCE:
            break
        others = [j for j in range(len(kfs)) if j != i_imp and kfs[j]["kick_hip_lift"] < LIFT_MAX]
        if not others:
            break
        j = min(others, key=lambda j: abs(kfs[j]["t"] - c["kick_foot_min_t"]))
        kfs[j]["kick_hip_lift"] = float(min(kfs[j]["kick_hip_lift"] + CLEARANCE - c["kick_foot_min"] + 0.002,
                                            LIFT_MAX))
    for kf in kfs:
        kf["kick_hip_lift"] = round(kf["kick_hip_lift"], 3)
    return k, check_kick(k, H)
