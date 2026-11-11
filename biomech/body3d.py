"""측면(시상면) 2D 모델 → 3D 좌표: 좌우 폭만 더한다 (운동면).

좌표: X = 앞(차는 방향), Y = 위, Z = 왼쪽(+). 다리·팔은 각자 고관절·어깨를 지나는 시상면 안에서 움직인다.
고관절 회전·몸통 비틀기 같은 수평면 동작은 모델에 없으므로 표현하지 않는다.
"""

import numpy as np

HIP_HALF_RATIO = 0.055        # 고관절 중심 사이 거리의 절반 (× H, 약 19 cm / 2)
SHOULDER_HALF_RATIO = 0.105   # 양 어깨 관절 사이 거리의 절반 (× H)

# 몸 윤곽(살) 반지름 (× H): (근위, 원위)
LIMB_RADII: dict[str, tuple[float, float]] = {
    "thigh": (0.040, 0.028), "shank": (0.027, 0.018), "foot": (0.016, 0.012),
    "upper_arm": (0.024, 0.019), "forearm": (0.019, 0.014), "neck": (0.025, 0.025),
}
TORSO_HALF_WIDTH = (0.085, 0.100)   # 골반, 가슴 (× H)
TORSO_HALF_DEPTH = (0.055, 0.060)   # 골반, 가슴 (× H) — 측면에서 본 두께의 절반


def side_z(name: str, H: float) -> float:
    """관절 이름 → Z(좌우) 위치."""
    if name.startswith("r_"):
        sign = -1.0
    elif name.startswith("l_"):
        sign = 1.0
    else:
        return 0.0
    upper = any(k in name for k in ("shoulder", "elbow", "hand", "wrist"))
    return sign * (SHOULDER_HALF_RATIO if upper else HIP_HALF_RATIO) * H


# 팔 벌림(외전, 관상면) — 키프레임에는 시상면 굴곡만 있으므로 3D 표시·후면 합성용 교육용 값.
# 차는 다리 반대쪽(왼) 팔을 옆으로 벌려 균형을 잡는다 (각운동량).
ARM_ABDUCTION_DEG = {"l": 40.0, "r": 15.0}


def foot_yaw_series(kick: dict, t: np.ndarray) -> np.ndarray:
    """차는 발 방향(발끝 바깥 +, 도)의 시간 변화: 디딤 때 0 → 임팩트 80 ms 전까지 킥별 값으로 돌림 →
    임팩트 50 ms 후까지 유지 → 동작 끝에 절반으로 (고관절 외회전·내회전, 수평면 동작)."""
    yaw = float(kick["impact"].get("foot_yaw_deg", 0.0))
    kfs = kick["keyframes"]
    t_imp = kick["impact"]["time"]
    t0 = next((kf["t"] for kf in kfs if kf["phase"] == "디딤"), kfs[0]["t"])
    t1 = max(t0 + 1e-3, t_imp - 0.08)
    t2 = t_imp + 0.05
    t3 = max(kfs[-1]["t"], t2 + 1e-3)
    return np.interp(t, [t0, t1, t2, t3], [0.0, yaw, yaw, 0.5 * yaw])


def to_3d(points: dict[str, np.ndarray], H: float, abduct_arms: bool = True,
          foot_yaw_deg: float | np.ndarray = 0.0) -> dict[str, np.ndarray]:
    """2D 관절 좌표(x 앞, y 위) → 3D (X, Y, Z). FK의 '어깨' 한 점은 양 어깨로 나눈다.

    abduct_arms: 팔 전체를 어깨를 지나는 앞뒤 축(X) 둘레로 바깥쪽으로 돌려 벌린다.
    foot_yaw_deg: 차는(오른) 발을 발목을 지나는 수직축 둘레로 돌린다 (발끝 바깥 +). 측면 2D 모델에는 없는
                  수평면 회전이라 3D에서만 보인다.
    """
    out = {}
    for k, v in points.items():
        v = np.asarray(v, dtype=float)
        z = np.full(v.shape[:-1] + (1,), side_z(k, H))
        out[k] = np.concatenate([v, z], axis=-1)
    if "shoulder" in out:
        for s, sign in (("r", -1.0), ("l", 1.0)):
            sh = out["shoulder"] + np.array([0.0, 0.0, sign * SHOULDER_HALF_RATIO * H])
            out[f"{s}_shoulder"] = sh
            th = np.radians(ARM_ABDUCTION_DEG[s]) if abduct_arms else 0.0
            for j in ("elbow", "hand"):
                if f"{s}_{j}" not in out:
                    continue
                d = out[f"{s}_{j}"] - sh                          # 어깨 기준 벡터 (Z 성분 0)
                dy = d[..., 1]
                out[f"{s}_{j}"] = sh + np.stack([d[..., 0], dy * np.cos(th), -sign * dy * np.sin(th)], axis=-1)
    if "r_ankle" in out and np.any(np.asarray(foot_yaw_deg) != 0):
        psi = np.radians(np.asarray(foot_yaw_deg, dtype=float))
        c, s = np.cos(psi), np.sin(psi)
        a = out["r_ankle"]
        for j in ("r_heel", "r_toe"):
            d = out[j] - a                      # 수직축(+Y) 둘레 회전: 발끝 +X → −Z(오른쪽 = 바깥)
            out[j] = a + np.stack([c * d[..., 0] + s * d[..., 2], d[..., 1], -s * d[..., 0] + c * d[..., 2]], axis=-1)
    return out
