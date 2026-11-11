"""순운동학(FK), PCHIP 키프레임 보간, 각속도 계산 (각운동학)."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from biomech.body import HEEL_OFFSET_RATIO, TOE_OFFSET_RATIO, segment_lengths

# 키프레임에 들어가는 관절 각도 키 (단위: 도)
ANGLE_KEYS: list[str] = [
    "trunk", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "r_shoulder", "l_shoulder", "r_elbow", "l_elbow",
]


def d(a_deg: float) -> np.ndarray:
    """절대 분절 각도 a(아래 수직에서 앞으로 회전한 각)의 방향 단위벡터."""
    a = np.radians(a_deg)
    return np.array([np.sin(a), -np.cos(a)])


def rot(v: np.ndarray, a_deg: float) -> np.ndarray:
    """벡터 v를 a_deg만큼 반시계 회전한다."""
    a = np.radians(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])


def leg(hip: np.ndarray, trunk: float, hip_flex: float, knee_flex: float, ankle_pf: float,
        L: dict[str, float], H: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """다리 FK: 무릎·발목·뒤꿈치·발끝 좌표를 반환한다 (관절 각도 정의)."""
    a_thigh = hip_flex - trunk          # 몸통이 앞으로 숙이면 허벅지 기준선은 뒤로 기움
    a_shank = a_thigh - knee_flex       # 무릎 굴곡 → 정강이가 뒤로 접힘
    knee  = hip  + L["thigh"] * d(a_thigh)
    ankle = knee + L["shank"] * d(a_shank)
    a_foot = a_shank - ankle_pf         # 저측굴곡 → 발끝이 아래로
    heel = ankle + rot(np.array(HEEL_OFFSET_RATIO) * H, a_foot)
    toe  = ankle + rot(np.array(TOE_OFFSET_RATIO) * H, a_foot)
    return knee, ankle, heel, toe


def arm(shoulder: np.ndarray, trunk: float, shoulder_flex: float, elbow_flex: float,
        L: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """팔 FK: 팔꿈치·손 좌표를 반환한다."""
    a_upper = shoulder_flex - trunk
    a_fore = a_upper + elbow_flex       # 팔꿈치는 앞으로 굽음
    elbow = shoulder + L["upper_arm"] * d(a_upper)
    hand = elbow + L["forearm"] * d(a_fore)
    return elbow, hand


def segment_abs_angles(trunk: np.ndarray, hip_flex: np.ndarray,
                       knee_flex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """허벅지·정강이의 절대 분절 각도(도)를 반환한다."""
    a_thigh = hip_flex - trunk
    return a_thigh, a_thigh - knee_flex


def pose(angles: dict[str, float], hip: np.ndarray, H: float,
         kick_hip_lift: float = 0.0) -> dict[str, np.ndarray]:
    """한 자세의 모든 관절 좌표를 계산한다. hip = 고관절 중심(루트)."""
    L = segment_lengths(H)
    hip = np.asarray(hip, dtype=float)
    t = angles["trunk"]
    up = np.array([np.sin(np.radians(t)), np.cos(np.radians(t))])
    shoulder = hip + L["trunk"] * up
    head = shoulder + L["neck_head"] * up
    r_hip = hip + np.array([0.0, kick_hip_lift])   # 관상면(옆으로 기울기) 보정
    l_hip = hip.copy()
    p: dict[str, np.ndarray] = {"hip": hip, "r_hip": r_hip, "l_hip": l_hip,
                                "shoulder": shoulder, "head": head}
    for side, h in (("r", r_hip), ("l", l_hip)):
        knee, ankle, heel, toe = leg(h, t, angles[f"{side}_hip"], angles[f"{side}_knee"],
                                     angles[f"{side}_ankle"], L, H)
        p.update({f"{side}_knee": knee, f"{side}_ankle": ankle,
                  f"{side}_heel": heel, f"{side}_toe": toe})
        elbow, hand = arm(shoulder, t, angles[f"{side}_shoulder"], angles[f"{side}_elbow"], L)
        p.update({f"{side}_elbow": elbow, f"{side}_hand": hand})
    return p


def interpolate(keyframes: list[dict], keys: list[str], n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """키프레임을 PCHIP으로 보간해 n개 프레임의 시간과 값 배열을 반환한다.

    일반 3차 스플라인은 키프레임 사이에서 값이 튀어(overshoot) 발이 땅속으로 들어갈 수 있다.
    """
    kt = np.array([k["t"] for k in keyframes], dtype=float)
    t = np.linspace(kt[0], kt[-1], n)
    out = {key: PchipInterpolator(kt, [k[key] for k in keyframes])(t) for key in keys}
    return t, out


def angular_velocity(t: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """각도(도) 시계열을 수치미분해 각속도(°/s)를 구한다 (ω = dθ/dt)."""
    return np.gradient(angle_deg, t)


# ---------------------------------------------------------------------------
# 몸 전체 동작: 보간 + 루트(골반) 위치 결정
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field

from biomech.body import SEGMENT_COM, SEGMENT_MASS

APPROACH_SPEED = 3.0       # 디딤 전 골반 수평 이동 속도 (m/s), 선운동학
ANIM_FRAMES = 120          # 0.6 s 동작 기준 약 120프레임 (5 ms 간격)
CONTACT_NAMES = {"instep": "발등", "inside": "발 안쪽", "outside": "발 바깥쪽",
                 "toe": "발끝", "under": "발끝(공 아래)"}


@dataclass
class Motion:
    """보간된 전체 동작. points[name]은 (n, 2) 배열."""
    t: np.ndarray
    angles: dict[str, np.ndarray]
    lift: np.ndarray
    points: dict[str, np.ndarray]
    phase: list[str]
    pin_point: str | None = None
    pin_xy: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def contact_point(p: dict[str, np.ndarray], kind: str) -> np.ndarray:
    """차는 발의 공 접촉점 좌표 (측면 투영 근사)."""
    ankle, heel, toe = p["r_ankle"], p["r_heel"], p["r_toe"]
    if kind == "instep":
        return (ankle + toe) / 2            # 발목·발끝 중점
    if kind == "inside":
        return (heel + toe) / 2             # 발 안쪽 아치 (측면 투영)
    if kind == "outside":
        return 0.25 * ankle + 0.75 * toe    # 새끼발가락 쪽 중족골
    return toe.copy()                       # toe, under


def phase_at(keyframes: list[dict], t: float) -> str:
    """시각 t에 해당하는 단계 이름 (t 이하인 마지막 키프레임)."""
    name = keyframes[0]["phase"]
    for kf in keyframes:
        if kf["t"] <= t + 1e-9:
            name = kf["phase"]
    return name


def _plant_time(kick: dict) -> float:
    for kf in kick["keyframes"]:
        if kf["phase"] == "디딤":
            return kf["t"]
    return kick["keyframes"][0]["t"]


def compute_motion(kick: dict, H: float, n: int = ANIM_FRAMES,
                   extra_times: list[float] | None = None) -> Motion:
    """키프레임을 보간하고 루트를 결정해 모든 프레임의 관절 좌표를 계산한다.

    디딤발 고정(stance_pin "L"): 디딤 이후 왼발의 고정점(디딤 순간 더 낮은 점)이 지면에 고정되도록
    루트를 역산한다. 디딤 전에는 접근 속도로 이동하고 가장 낮은 발 점이 y=0.
    공중 동작(stance_pin null): 키프레임의 root_y(와 선택적 root_x)를 사용한다.
    """
    kfs = kick["keyframes"]
    t, ang = interpolate(kfs, ANGLE_KEYS, n)
    # 키프레임·임팩트 시각이 프레임 격자에 정확히 들어가도록 추가
    special = sorted({kf["t"] for kf in kfs} | {kick["impact"]["time"]} | set(extra_times or []))
    t = np.unique(np.round(np.concatenate([t, special]), 9))
    extra_keys = ["kick_hip_lift"] + (["root_y"] if kick["stance_pin"] is None else [])
    if kick["stance_pin"] is None and all("root_x" in k for k in kfs):
        extra_keys.append("root_x")
    kt = np.array([k["t"] for k in kfs])
    ang = {k: PchipInterpolator(kt, [kf[k] for kf in kfs])(t) for k in ANGLE_KEYS + extra_keys}
    lift = ang.pop("kick_hip_lift")

    local = [pose({k: ang[k][i] for k in ANGLE_KEYS}, np.zeros(2), H, lift[i]) for i in range(len(t))]
    names = list(local[0])
    root = np.zeros((len(t), 2))
    pin_point, pin_xy = None, None

    if kick["stance_pin"] is None:
        root[:, 1] = ang.pop("root_y")
        root[:, 0] = ang.pop("root_x") if "root_x" in ang else 0.0
    else:
        t_plant = _plant_time(kick)
        i_plant = int(np.argmin(np.abs(t - t_plant)))
        lp = local[i_plant]
        pin_point = "l_heel" if lp["l_heel"][1] <= lp["l_toe"][1] else "l_toe"
        pin_xy = np.array([0.0, 0.0])
        root_plant_x = pin_xy[0] - lp[pin_point][0]
        feet = ["l_heel", "l_toe", "r_heel", "r_toe"]
        for i, p in enumerate(local):
            if i >= i_plant:
                root[i] = pin_xy - p[pin_point]
            else:
                root[i, 0] = root_plant_x + APPROACH_SPEED * (t[i] - t_plant)
                root[i, 1] = -min(p[f][1] for f in feet)

    points = {nm: np.array([local[i][nm] for i in range(len(t))]) + root for nm in names}
    phase = [phase_at(kfs, ti) for ti in t]
    return Motion(t=t, angles=ang, lift=lift, points=points, phase=phase,
                  pin_point=pin_point, pin_xy=pin_xy)


def frame_points(m: Motion, i: int) -> dict[str, np.ndarray]:
    """i번째 프레임의 관절 좌표 딕셔너리."""
    return {k: v[i] for k, v in m.points.items()}


def center_of_mass(p: dict[str, np.ndarray]) -> np.ndarray:
    """분절 질량 가중 평균으로 몸 전체 무게중심을 구한다 (질량중심)."""
    def seg(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
        return a + r * (b - a)
    acc = SEGMENT_MASS["head"] * p["head"]
    acc = acc + SEGMENT_MASS["trunk"] * seg(p["hip"], p["shoulder"], SEGMENT_COM["trunk"])
    for s in "rl":
        acc = acc + SEGMENT_MASS["upper_arm"] * seg(p["shoulder"], p[f"{s}_elbow"], SEGMENT_COM["upper_arm"])
        acc = acc + SEGMENT_MASS["forearm"] * seg(p[f"{s}_elbow"], p[f"{s}_hand"], SEGMENT_COM["forearm"])
        acc = acc + SEGMENT_MASS["thigh"] * seg(p[f"{s}_hip"], p[f"{s}_knee"], SEGMENT_COM["thigh"])
        acc = acc + SEGMENT_MASS["shank"] * seg(p[f"{s}_knee"], p[f"{s}_ankle"], SEGMENT_COM["shank"])
        acc = acc + SEGMENT_MASS["foot"] * (p[f"{s}_heel"] + p[f"{s}_toe"]) / 2
    return acc / sum(SEGMENT_MASS[k] * (1 if k in ("head", "trunk") else 2) for k in SEGMENT_MASS)


IMPACT_WINDOW = 0.03       # 임팩트 직전 발 최고 속도를 찾는 구간 (s)


def impact_analysis(kick: dict, H: float, n: int = 600) -> dict:
    """임팩트 발 접촉점 속도를 FK 수치미분으로 구한다 (v = dr/dt, 공/발 속도비).

    PCHIP은 키프레임(임팩트) 지점에서 기울기를 양쪽 구간의 조화평균으로 잡아 속도가 인위적으로
    줄어든다. 그래서 공 속도 기본값은 임팩트 직전 IMPACT_WINDOW 안의 최대 발 속도로 계산한다.
    """
    m = compute_motion(kick, H, n)
    imp = kick["impact"]
    i = int(np.argmin(np.abs(m.t - imp["time"])))
    cp = np.array([contact_point(frame_points(m, j), imp["contact_point"]) for j in range(len(m.t))])
    vel = np.gradient(cp, m.t, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    window = np.where((m.t >= imp["time"] - IMPACT_WINDOW) & (m.t <= imp["time"]))[0]
    j = window[np.argmax(speed[window])]
    foot_speed = float(speed[j])
    return {
        "contact": cp[i], "foot_velocity": vel[j], "foot_speed": foot_speed,
        "foot_speed_at_impact": float(speed[i]), "peak_time": float(m.t[j]),
        "ball_speed": foot_speed * imp["ball_foot_speed_ratio"],
        "direction": 1.0 if vel[i][0] >= 0 else -1.0,
    }
