"""킥 키프레임 JSON 로드 + 스키마 검증."""

import json
from pathlib import Path

from biomech.kinematics import ANGLE_KEYS

KICKS_DIR = Path(__file__).resolve().parent.parent / "data" / "kicks"

# 화면에 보여줄 순서
KICK_ORDER: list[str] = [
    "instep", "inside", "outside", "curve", "chip", "volley",
    "toe", "half_volley",
]
PHASES: list[str] = ["접근", "디딤", "백스윙", "가속", "임팩트", "팔로스루"]
CONTACT_POINTS = {"instep", "inside", "outside", "toe", "under"}

TOP_KEYS = ["id", "name_ko", "name_en", "excel_name", "is_example_values",
            "airborne", "stance_pin", "impact", "keyframes", "rotation_notes"]
IMPACT_KEYS = ["time", "contact_point", "contact_height", "ball_foot_speed_ratio", "launch_angle_deg"]
KEYFRAME_KEYS = ["t", "phase", *ANGLE_KEYS, "kick_hip_lift", "highlight", "note"]


class KickSchemaError(ValueError):
    """킥 JSON이 5장 형식을 따르지 않을 때."""


def validate_kick(kick: dict) -> None:
    """5장 스키마(필수 키, 시간 오름차순, 각도 범위 −180~180)를 검사한다."""
    def require(obj: dict, keys: list[str], where: str) -> None:
        missing = [k for k in keys if k not in obj]
        if missing:
            raise KickSchemaError(f"{where}: 필수 키 누락 {missing}")

    require(kick, TOP_KEYS, kick.get("id", "?"))
    require(kick["impact"], IMPACT_KEYS, f"{kick['id']}.impact")
    if kick["impact"]["contact_point"] not in CONTACT_POINTS:
        raise KickSchemaError(f"{kick['id']}: 알 수 없는 contact_point")
    imp = kick["impact"]
    if not -90 <= imp.get("foot_yaw_deg", 0) <= 90:
        raise KickSchemaError(f"{kick['id']}: foot_yaw_deg는 −90~90°")
    off = imp.get("contact_offset_m", {})
    if (off.get("side", 0) ** 2 + off.get("up", 0) ** 2) ** 0.5 >= 0.11:
        raise KickSchemaError(f"{kick['id']}: contact_offset_m가 공 반지름(0.11 m)보다 큼")
    if kick["stance_pin"] not in ("L", None):
        raise KickSchemaError(f"{kick['id']}: stance_pin은 'L' 또는 null")

    kfs = kick["keyframes"]
    if len(kfs) < 2:
        raise KickSchemaError(f"{kick['id']}: 키프레임이 2개 이상 필요")
    for i, kf in enumerate(kfs):
        require(kf, KEYFRAME_KEYS, f"{kick['id']}.keyframes[{i}]")
        if kick["airborne"] and "root_y" not in kf:
            raise KickSchemaError(f"{kick['id']}.keyframes[{i}]: 공중 동작은 root_y 필요")
        for key in ANGLE_KEYS:
            if not -180 <= kf[key] <= 180:
                raise KickSchemaError(f"{kick['id']}.keyframes[{i}].{key} 범위 초과")
    times = [kf["t"] for kf in kfs]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise KickSchemaError(f"{kick['id']}: 키프레임 시간이 오름차순이 아님")
    if not times[0] <= kick["impact"]["time"] <= times[-1]:
        raise KickSchemaError(f"{kick['id']}: 임팩트 시각이 동작 범위 밖")


def load_kick(kick_id: str) -> dict:
    """data/kicks/<id>.json을 읽고 검증한다."""
    with open(KICKS_DIR / f"{kick_id}.json", encoding="utf-8") as f:
        kick = json.load(f)
    validate_kick(kick)
    return kick


def available_kicks() -> list[str]:
    """data/kicks에 존재하는 킥 id를 화면 순서대로 반환한다."""
    present = {p.stem for p in KICKS_DIR.glob("*.json")}
    return [k for k in KICK_ORDER if k in present] + sorted(present - set(KICK_ORDER))
