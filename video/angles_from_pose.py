"""랜드마크 좌표 → CLAUDE.md 4.3 관절 각도 (4.4 FK의 정확한 역변환, 용어·운동면·각운동학).

입력 좌표는 몸 기준 좌표계(위 = +y, 차는 방향 = +x, r_ = 차는 다리)여야 한다.
방향·좌우 정규화는 `video.preprocess`가 담당한다.
"""

import numpy as np

# 내부 랜드마크 이름 → MediaPipe Pose 인덱스 (foot_index는 FK 이름에 맞춰 toe로 부른다)
LANDMARKS: dict[str, int] = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30, "l_toe": 31, "r_toe": 32,
}


def wrap180(a: np.ndarray | float) -> np.ndarray | float:
    """각도를 −180~180 범위로 정규화한다."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def seg_angle(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """분절 p→q의 절대각: 아래 수직 기준, 앞(+x)으로 회전하면 + (도)."""
    v = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))


def trunk_angle(hip_mid: np.ndarray, shoulder_mid: np.ndarray) -> np.ndarray:
    """몸통: 위 수직 기준, 앞으로 숙이면 + (도)."""
    v = np.asarray(shoulder_mid, dtype=float) - np.asarray(hip_mid, dtype=float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def foot_angle(heel: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """발 벡터(뒤꿈치→발끝)의 각: +x 기준 반시계 (도)."""
    v = np.asarray(toe, dtype=float) - np.asarray(heel, dtype=float)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def angles_from_pose(xy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """랜드마크 좌표(한 프레임 (2,) 또는 시계열 (n, 2))로 4.3 정의의 관절 각도(도)를 구한다.

    FK(4.4)의 역변환: hip = a_thigh + trunk, knee = a_thigh − a_shank, ankle = a_shank − φ_foot,
    shoulder = a_upper + trunk, elbow = a_fore − a_upper.
    """
    hip_mid = (xy["l_hip"] + xy["r_hip"]) / 2
    shoulder_mid = (xy["l_shoulder"] + xy["r_shoulder"]) / 2
    trunk = trunk_angle(hip_mid, shoulder_mid)
    out: dict[str, np.ndarray] = {"trunk": wrap180(trunk)}
    for s in "rl":
        a_thigh = seg_angle(xy[f"{s}_hip"], xy[f"{s}_knee"])
        a_shank = seg_angle(xy[f"{s}_knee"], xy[f"{s}_ankle"])
        phi_foot = foot_angle(xy[f"{s}_heel"], xy[f"{s}_toe"])
        out[f"{s}_hip"] = wrap180(a_thigh + trunk)
        out[f"{s}_knee"] = wrap180(a_thigh - a_shank)
        out[f"{s}_ankle"] = wrap180(a_shank - phi_foot)
        a_upper = seg_angle(xy[f"{s}_shoulder"], xy[f"{s}_elbow"])
        a_fore = seg_angle(xy[f"{s}_elbow"], xy[f"{s}_wrist"])
        out[f"{s}_shoulder"] = wrap180(a_upper + trunk)
        out[f"{s}_elbow"] = wrap180(a_fore - a_upper)
    return out


def frontal_angles(xy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """후면(관상면) 좌표(x = 차는 다리 쪽 +, y = 위)로 관상면 각도(도)를 구한다 (운동면).

    - stance_knee_valgus: 디딤 무릎 외반 — 허벅지·정강이 선이 꺾인 각, 무릎이 몸 안쪽(+x)으로 들어가면 +
    - trunk_lateral_lean: 몸통 옆 기울기 — 수직 기준, 디딤발 쪽(−x)으로 기울면 +
    - pelvic_obliquity: 골반 기울기 — 차는 쪽 고관절이 높으면 +
    - arm_abduction: 디딤발 쪽 팔 벌림 — 아래 수직 기준, 바깥쪽(−x)으로 벌리면 +
    """
    v1 = xy["l_knee"] - xy["l_hip"]
    v2 = xy["l_ankle"] - xy["l_knee"]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = np.sum(v1 * v2, axis=-1)
    trunk = (xy["l_shoulder"] + xy["r_shoulder"]) / 2 - (xy["l_hip"] + xy["r_hip"]) / 2
    pel = xy["r_hip"] - xy["l_hip"]
    arm = xy["l_wrist"] - xy["l_shoulder"]
    return {
        "stance_knee_valgus": np.degrees(np.arctan2(-cross, dot)),
        "trunk_lateral_lean": np.degrees(np.arctan2(-trunk[..., 0], trunk[..., 1])),
        "pelvic_obliquity": np.degrees(np.arctan2(pel[..., 1], np.abs(pel[..., 0]))),
        "arm_abduction": np.degrees(np.arctan2(-arm[..., 0], -arm[..., 1])),
    }


def landmarks_from_fk(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """FK 관절 좌표(`biomech.kinematics.pose`)를 랜드마크 이름으로 바꾼다 (합성 테스트용).

    FK는 어깨가 한 점이므로 양 어깨에 같은 좌표를, 손 중앙을 손목에, 머리 중심을 코에 쓴다.
    """
    lm = {"nose": p["head"], "l_shoulder": p["shoulder"], "r_shoulder": p["shoulder"]}
    for s in "rl":
        lm.update({f"{s}_elbow": p[f"{s}_elbow"], f"{s}_wrist": p[f"{s}_hand"],
                   f"{s}_hip": p[f"{s}_hip"], f"{s}_knee": p[f"{s}_knee"],
                   f"{s}_ankle": p[f"{s}_ankle"], f"{s}_heel": p[f"{s}_heel"],
                   f"{s}_toe": p[f"{s}_toe"]})
    return lm
