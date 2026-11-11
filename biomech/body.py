"""신체 분절 길이·질량 상수 (키 H 기준, Winter의 인체 분절 비율 근사)."""

DEFAULT_HEIGHT_M = 1.75
MIN_HEIGHT_CM = 150
MAX_HEIGHT_CM = 200

# 분절 길이 비율 (× H)
SEGMENT_RATIOS: dict[str, float] = {
    "thigh": 0.245,      # 고관절 → 무릎
    "shank": 0.246,      # 무릎 → 발목
    "trunk": 0.288,      # 고관절 중심 → 어깨
    "neck_head": 0.091,  # 어깨 → 머리 중심
    "upper_arm": 0.186,  # 어깨 → 팔꿈치
    "forearm": 0.200,    # 팔꿈치 → 손 중앙
}
HEAD_RADIUS_RATIO = 0.06

# 발목 기준 발 점 (× H), 발이 정강이와 직각일 때
HEEL_OFFSET_RATIO = (-0.03, -0.039)
TOE_OFFSET_RATIO = (0.12, -0.039)

# 분절 질량 비율 (체중 대비)과 근위단 기준 무게중심 위치 비율 (Winter) — 무게중심 표시용
SEGMENT_MASS: dict[str, float] = {
    "head": 0.081, "trunk": 0.497,
    "upper_arm": 0.028, "forearm": 0.022,
    "thigh": 0.100, "shank": 0.0465, "foot": 0.0145,
}
SEGMENT_COM: dict[str, float] = {
    "trunk": 0.5, "upper_arm": 0.436, "forearm": 0.682,
    "thigh": 0.433, "shank": 0.433,
}


def segment_lengths(height_m: float = DEFAULT_HEIGHT_M) -> dict[str, float]:
    """키(m)로부터 분절 길이(m)를 계산한다."""
    lengths = {name: ratio * height_m for name, ratio in SEGMENT_RATIOS.items()}
    lengths["head_radius"] = HEAD_RADIUS_RATIO * height_m
    return lengths
