"""동작 단계(접근·디딤·백스윙·가속·임팩트·팔로스루) 자동 검출 (선·각운동학)."""

from dataclasses import dataclass, field

import numpy as np

from biomech.kick_loader import PHASES
from video.angles_from_pose import seg_angle
from video.pose_series import Processed
from video.preprocess import lowpass, speed

PLANT_SPEED = 0.3           # 디딤발 발목 정지 판정 속도 (m/s)
APPROACH_BEFORE_PLANT = 0.2 # 접근 시작 = 디딤 0.2 s 전
AIR_WINDOW = 0.1            # 공중 동작 판정 구간: 임팩트 ±0.1 s
AIR_HEIGHT = 0.15           # 공중 동작 판정: 양발 최저점이 지면 + 0.15 m 이상
IMPACT_CUTOFF_HZ = 25.0     # 임팩트 검출용 약한 필터 (8 Hz 필터는 10 ms 충돌을 여러 프레임으로 번지게 함)
IMPACT_SEARCH = (0.02, 0.08)  # 발 속도 최고점 기준 급감속 탐색 구간 (전, 후) s
REAR_IMPACT_WINDOW = 0.1    # 후면: 발 속도 최고점 ±0.1 s 안의 최저점
REAR_PLANT_HEIGHT = 0.05    # 후면 디딤 판정: 디딤발 최저점이 지면 + 5 cm 이내


@dataclass
class Events:
    """검출된 이벤트 프레임 (None = 검출 실패)."""
    frames: dict[str, int | None]
    t: np.ndarray
    airborne: bool
    contact_height: float | None    # 임팩트 시 차는 발(발목·발끝 중점) 높이 (m)
    manual: set[str] = field(default_factory=set)

    @property
    def failed(self) -> list[str]:
        return [k for k in PHASES if self.frames.get(k) is None]

    def time(self, phase: str) -> float | None:
        i = self.frames.get(phase)
        return None if i is None else float(self.t[i])

    @property
    def impact(self) -> int | None:
        return self.frames.get("임팩트")


def kick_foot_point(xy: dict[str, np.ndarray]) -> np.ndarray:
    """차는 발 기준점 = 발목·발끝 중점."""
    return (xy["r_ankle"] + xy["r_toe"]) / 2


def light_filtered(p: Processed, names: list[str]) -> dict[str, np.ndarray]:
    """임팩트 근처 속도용으로 약하게(25 Hz) 필터한 좌표.

    8 Hz 필터는 임팩트 근처 최고 속도를 30% 가까이 깎는다 (킥 연구에서 알려진 문제).
    """
    cut = min(IMPACT_CUTOFF_HZ, 0.8 * p.fps / 2)
    return {k: lowpass(p.xy_unfiltered[k], p.fps, cut)[0] for k in names}


def _seg_omega(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ang = np.degrees(np.unwrap(np.radians(seg_angle(a, b))))
    return np.gradient(ang, t)


def thigh_angular_velocity(p: Processed, light: bool = False) -> np.ndarray:
    """차는 허벅지 절대 분절 각속도 (°/s, 앞으로 휘두르면 +) (ω = dθ/dt)."""
    xy = light_filtered(p, ["r_hip", "r_knee"]) if light else p.xy
    return _seg_omega(p.t, xy["r_hip"], xy["r_knee"])


def shank_angular_velocity(p: Processed, light: bool = False) -> np.ndarray:
    """차는 정강이 절대 분절 각속도 (°/s)."""
    xy = light_filtered(p, ["r_knee", "r_ankle"]) if light else p.xy
    return _seg_omega(p.t, xy["r_knee"], xy["r_ankle"])


def detect_impact(p: Processed) -> int | None:
    """임팩트: 발 속도 최고점 직후, 속도가 가장 급격히 감소하는 프레임 (공과의 충돌로 감속).

    최고점은 8 Hz 필터 데이터에서 찾고(노이즈에 강함), 급감속은 약하게 필터한 좌표에서
    후진차분(직전 프레임 대비 감소량)이 가장 큰 프레임으로 찾는다.
    """
    s8 = speed(kick_foot_point(p.xy), p.fps)
    ip = int(np.argmax(s8))
    raw = kick_foot_point(p.xy_unfiltered)
    f, _ = lowpass(raw, p.fps, min(IMPACT_CUTOFF_HZ, 0.8 * p.fps / 2))
    s = speed(f, p.fps)
    a = max(ip - int(round(IMPACT_SEARCH[0] * p.fps)), 1)
    b = min(ip + int(round(IMPACT_SEARCH[1] * p.fps)) + 1, len(s))
    if b - a < 1:
        return None
    back = s[a:b] - s[a - 1:b - 1]
    return a + int(np.argmin(back))


def detect_impact_rear(p: Processed) -> int | None:
    """후면 임팩트: 발 속도 최고점 ±0.1 s 안에서 차는 발이 가장 낮은 프레임.

    뒤에서 보면 발이 앞뒤로 움직이는 것은 화면에 거의 안 보이고, 다리가 진자처럼 내려와
    스윙 아크의 가장 낮은 점(지면의 공 높이)에서 공을 맞힌다.
    """
    s8 = speed(kick_foot_point(p.xy), p.fps)
    ip = int(np.argmax(s8))
    w = int(round(REAR_IMPACT_WINDOW * p.fps))
    a, b = max(ip - w, 0), min(ip + w + 1, p.n)
    return a + int(np.argmin(kick_foot_point(p.xy)[a:b, 1]))


def detect_plant(p: Processed, impact: int) -> int | None:
    """디딤: 임팩트 이전에 디딤발 발목 속도가 0.3 m/s 아래로 떨어져 임팩트까지 유지되기 시작한 프레임."""
    s = speed(p.xy["l_ankle"], p.fps)[:impact + 1]
    if len(s) < 2 or s[-1] >= PLANT_SPEED:
        return None
    fast = np.where(s >= PLANT_SPEED)[0]
    i = 0 if len(fast) == 0 else int(fast[-1]) + 1
    return i if i < impact else None


def detect_plant_rear(p: Processed, impact: int) -> int | None:
    """후면 디딤: 디딤발 최저점이 지면 근처(< 5 cm)이면서 발목 속도 < 0.3 m/s인 상태가 임팩트까지 이어지기 시작한 프레임.

    뒤에서 보면 접근(앞뒤) 움직임이 화면에 거의 안 보이므로 발이 지면에 닿았는지를 함께 본다.
    """
    low = np.minimum(p.xy["l_heel"][:, 1], p.xy["l_toe"][:, 1])[:impact + 1] < REAR_PLANT_HEIGHT
    slow = speed(p.xy["l_ankle"], p.fps)[:impact + 1] < PLANT_SPEED
    ok = low & slow
    if not ok[-1]:
        return None
    bad = np.where(~ok)[0]
    i = 0 if len(bad) == 0 else int(bad[-1]) + 1
    return i if i < impact else None


def detect_events(p: Processed, overrides: dict[str, int] | None = None) -> Events:
    """모든 이벤트를 검출한다. overrides로 사용자가 지정한 프레임을 쓰면 뒤따르는 단계는 그 값을 기준으로 다시 찾는다."""
    ov = {k: int(v) for k, v in (overrides or {}).items() if v is not None}
    n, fps = p.n, p.fps
    fr: dict[str, int | None] = {k: None for k in PHASES}

    def pick(name: str, fn) -> int | None:
        if name in ov:
            fr[name] = int(np.clip(ov[name], 0, n - 1))
        else:
            try:
                fr[name] = fn()
            except (ValueError, IndexError):
                fr[name] = None
        return fr[name]

    rear = p.view == "rear"
    imp = pick("임팩트", lambda: detect_impact_rear(p) if rear else detect_impact(p))
    if imp is not None:
        plant = pick("디딤", lambda: detect_plant_rear(p, imp) if rear else detect_plant(p, imp))
        start = plant if plant is not None else max(imp - int(0.3 * fps), 0)
        if rear:
            # 후면: 무릎 굴곡 각도 대신 높이·속도로 찾는다
            heel_y = p.xy["r_heel"][:, 1]
            foot = kick_foot_point(p.xy)
            s = speed(foot, fps)
            end = min(imp + int(0.5 * fps), n - 1)
            bs = pick("백스윙", lambda: start + int(np.argmax(heel_y[start:imp + 1])) if imp > start else None)
            pick("가속", lambda: bs + int(np.argmax(s[bs:imp + 1])) if bs is not None and imp > bs else None)
            pick("팔로스루", lambda: imp + int(np.argmax(foot[imp:end + 1, 1])) if imp < n - 1 else None)
        else:
            knee = p.angles["r_knee"]
            bs = pick("백스윙", lambda: start + int(np.argmax(knee[start:imp + 1])) if imp > start else None)
            w = thigh_angular_velocity(p)
            pick("가속", lambda: bs + int(np.argmax(w[bs:imp + 1])) if bs is not None and imp > bs else None)
            hip = p.angles["r_hip"]
            pick("팔로스루", lambda: imp + int(np.argmax(hip[imp:])) if imp < n - 1 else None)
        pick("접근", lambda: max(plant - int(round(APPROACH_BEFORE_PLANT * fps)), 0)
             if plant is not None else 0)

    airborne, h = False, None
    if imp is not None:
        lo, hi = max(imp - int(AIR_WINDOW * fps), 0), min(imp + int(AIR_WINDOW * fps), n - 1)
        feet = np.stack([p.xy[k][lo:hi + 1, 1] for k in ("r_heel", "r_toe", "l_heel", "l_toe")])
        airborne = bool(np.all(feet.min(axis=0) > AIR_HEIGHT))
        h = float(kick_foot_point(p.xy)[imp, 1])
    return Events(frames=fr, t=p.t, airborne=airborne, contact_height=h, manual=set(ov))
