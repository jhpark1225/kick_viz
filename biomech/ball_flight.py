"""발 방향·접촉 위치 → 공의 발사 방향·회전 → 3D 비행 궤적.

좌표: X = 앞(차는 방향), Y = 위, Z = 왼쪽(+). (biomech/body3d.py와 같음)

- 발사 방향 (운동량 전달): 공은 대체로 발의 접촉면이 향하는 쪽으로 나간다. 발을 기준보다 Δψ만큼
  바깥으로 돌리면(발끝 바깥 +) 접촉면도 같이 돌아가 공의 수평 방향이 STEER × Δψ만큼 오른쪽으로 바뀐다.
- 회전 (각운동역학): 충격량 J = mΔv가 공 중심에서 d만큼 벗어나 작용하면 각충격량 d × J가
  각운동량 Iω로 바뀐다. 얇은 구껍질 I = ⅔mr² → ω = 3 Δv (d × û) / (2r²).
- 궤적: 중력 + 이차 항력 ½ρC_dAv² + 마그누스 힘 ½ρC_LAv² (ω̂ × v̂) (보충 개념, 수업 범위 밖)을 RK4로 적분.
"""

from dataclasses import dataclass

import numpy as np

from biomech.projectile import AIR_DENSITY, BALL_AREA, BALL_MASS, BALL_RADIUS, DRAG_COEF, G

STEER = 0.7                 # 발 방향 변화가 공 방향으로 전달되는 비율 (교육용 근사)
UP = np.array([0.0, 1.0, 0.0])

# 킥 JSON에 값이 없을 때의 기본값
DEFAULT_FOOT = {"foot_yaw_deg": 0.0, "aim_deg": 0.0, "contact_offset_m": {"side": 0.0, "up": 0.0}}


def foot_params(kick: dict) -> dict:
    """킥 JSON의 발 방향·조준·접촉 위치 (없으면 기본값)."""
    imp = kick["impact"]
    off = imp.get("contact_offset_m", DEFAULT_FOOT["contact_offset_m"])
    return {"foot_yaw_deg": float(imp.get("foot_yaw_deg", 0.0)), "aim_deg": float(imp.get("aim_deg", 0.0)),
            "side": float(off.get("side", 0.0)), "up": float(off.get("up", 0.0))}


def launch_direction(elev_deg: float, az_deg: float) -> np.ndarray:
    """발사 단위벡터. az + = 왼쪽(+Z)."""
    e, a = np.radians(elev_deg), np.radians(az_deg)
    return np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])


def ball_axes(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """공을 뒤에서 볼 때의 (오른쪽, 위) 단위벡터 — 접촉 위치 좌표축."""
    right = np.cross(UP, u)
    right /= np.linalg.norm(right) or 1e-9
    up = np.cross(u, right)
    return right, up / (np.linalg.norm(up) or 1e-9)


def spin_from_contact(v_ball: float, u: np.ndarray, side: float, up: float) -> np.ndarray:
    """접촉 위치로 생기는 회전 ω (rad/s). side + = 공의 오른쪽, up + = 공 중심보다 위 (m).

    ω = (3/2) Δv (d × û) / r²  (얇은 구껍질, 충격량이 접촉점을 지나 발 진행 방향으로 작용)
    """
    right, upv = ball_axes(u)
    d = side * right + up * upv
    return 1.5 * v_ball * np.cross(d, u) / BALL_RADIUS**2


def lift_coef(spin_param: float) -> float:
    """양력(마그누스) 계수 C_L — 스핀 파라미터 S = rω/v의 경험식 (보충)."""
    s = abs(spin_param)
    return 0.0 if s < 1e-9 else s / (2.022 * s + 0.981)


@dataclass
class Flight:
    t: np.ndarray
    pos: np.ndarray            # (n, 3) 공 아래쪽 높이 기준 (y = 0이면 착지)
    omega: np.ndarray          # 회전 벡터 (rad/s)
    v0: np.ndarray             # 발사 속도 벡터

    @property
    def flight_time(self) -> float:
        return float(self.t[-1])

    @property
    def range_m(self) -> float:
        """착지점까지 수평 거리."""
        return float(np.hypot(self.pos[-1, 0], self.pos[-1, 2]))

    @property
    def lateral_m(self) -> float:
        """착지점의 좌우 위치 (왼쪽 +)."""
        return float(self.pos[-1, 2])

    @property
    def max_height(self) -> float:
        return float(self.pos[:, 1].max())

    @property
    def spin_rps(self) -> float:
        return float(np.linalg.norm(self.omega) / (2 * np.pi))


def simulate(v: float, elev_deg: float, az_deg: float = 0.0, h0: float = 0.0, omega: np.ndarray | None = None,
             drag: bool = True, magnus: bool = True, dt: float = 0.002, t_max: float = 15.0) -> Flight:
    """3D 공 궤적 (RK4). 착지(y < 0)하면 멈춘다."""
    omega = np.zeros(3) if omega is None else np.asarray(omega, dtype=float)
    k = 0.5 * AIR_DENSITY * BALL_AREA / BALL_MASS
    w = np.linalg.norm(omega)
    w_hat = omega / w if w > 0 else np.zeros(3)

    def acc(vel: np.ndarray) -> np.ndarray:
        a = np.array([0.0, -G, 0.0])
        sp = np.linalg.norm(vel)
        if sp < 1e-9:
            return a
        if drag:
            a = a - k * DRAG_COEF * sp * vel
        if magnus and w > 0:
            cl = lift_coef(BALL_RADIUS * w / sp)
            a = a + k * cl * sp * np.cross(w_hat, vel)        # ½ρC_LAv² (ω̂ × v̂) / m
        return a

    vel = v * launch_direction(elev_deg, az_deg)
    v0 = vel.copy()
    pos = np.array([0.0, h0, 0.0])
    ts, ps = [0.0], [pos.copy()]
    t = 0.0
    while t < t_max:
        k1v, k1p = acc(vel), vel
        k2v, k2p = acc(vel + 0.5 * dt * k1v), vel + 0.5 * dt * k1v
        k3v, k3p = acc(vel + 0.5 * dt * k2v), vel + 0.5 * dt * k2v
        k4v, k4p = acc(vel + dt * k3v), vel + dt * k3v
        new_vel = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        new_pos = pos + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        t += dt
        if new_pos[1] < 0:                    # 착지 순간을 선형 보간
            f = pos[1] / (pos[1] - new_pos[1])
            ts.append(t - dt + f * dt)
            ps.append(pos + f * (new_pos - pos))
            break
        vel, pos = new_vel, new_pos
        ts.append(t)
        ps.append(pos.copy())
    return Flight(np.array(ts), np.array(ps), omega, v0)


def mirror_flight(f: Flight) -> Flight:
    """좌우 거울상 (왼발잡이): 위치·속도의 Z를 뒤집고, 회전(축 벡터)은 (−ωx, −ωy, ωz)가 된다.

    백스핀(Z축 성분)은 그대로, 사이드스핀(Y축 성분)은 방향이 바뀌어 반대쪽으로 휜다.
    """
    flip = np.array([1.0, 1.0, -1.0])
    return Flight(f.t, f.pos * flip, f.omega * np.array([-1.0, -1.0, 1.0]), f.v0 * flip)


def kick_flight(v_center: float, elev_deg: float, h0: float, aim_deg: float, d_yaw_deg: float,
                side: float, up: float, drag: bool = True, magnus: bool = True, left: bool = False) -> Flight:
    """발 방향 변화(d_yaw, 발끝 바깥 +)·접촉 위치(side, up)로 공의 방향·회전·궤적을 계산한다.

    중심에서 벗어나 차면 에너지 일부가 회전으로 가서 직진 속도가 √(1 − (d/r)²)배로 준다.
    left=True(왼발잡이): aim_deg·d_yaw는 차는 발 기준(킥 JSON은 오른발 기준) 값이고, side는 실제 공의 좌우 위치다.
    오른발 거울상으로 계산한 뒤 좌우를 뒤집는다.
    """
    if left:
        return mirror_flight(kick_flight(v_center, elev_deg, h0, aim_deg, d_yaw_deg, -side, up, drag, magnus))
    az = aim_deg - STEER * d_yaw_deg
    d = np.hypot(side, up)
    v = v_center * np.sqrt(max(1.0 - (d / BALL_RADIUS) ** 2, 0.0))
    u = launch_direction(elev_deg, az)
    omega = spin_from_contact(v, u, side, up)
    return simulate(v, elev_deg, az, h0, omega, drag, magnus)


def ball_path_3d(t: np.ndarray, t_impact: float, path2d: np.ndarray, center: np.ndarray, z0: float,
                 flight: Flight) -> np.ndarray:
    """애니메이션용 공 중심 (n, 3): 임팩트 전 = 2D 경로(좌우 z0), 임팩트 후 = 3D 비행 궤적."""
    out = np.column_stack([path2d, np.full(len(t), z0)])
    after = t >= t_impact
    s = np.clip(t[after] - t_impact, 0, flight.t[-1])
    for j, off in ((0, center[0]), (1, center[1]), (2, z0)):
        out[after, j] = off + np.interp(s, flight.t, flight.pos[:, j] - flight.pos[0, j])
    return out


def spin_description(f: Flight) -> list[str]:
    """회전을 백스핀/톱스핀/사이드스핀 성분으로 설명한다."""
    if f.spin_rps < 0.3:
        return ["회전 거의 없음 → 마그누스 힘 ≈ 0, 공이 휘지 않음"]
    u = f.v0 / np.linalg.norm(f.v0)
    right, upv = ball_axes(u)
    back = float(np.dot(f.omega, -right)) / (2 * np.pi)      # 왼쪽 축(= −오른쪽) 성분: + 백스핀
    side = float(np.dot(f.omega, upv)) / (2 * np.pi)         # 위쪽 축 성분: + 반시계(위에서 볼 때) → 오른쪽으로 휨
    out = []
    if abs(back) >= 0.3:
        out.append(f"{'백스핀' if back > 0 else '톱스핀'} {abs(back):.1f} 회/s → "
                   + ("공이 위로 뜨는 힘 (더 높고 길게)" if back > 0 else "공이 아래로 눌리는 힘 (빨리 떨어짐)"))
    if abs(side) >= 0.3:
        out.append(f"사이드스핀 {abs(side):.1f} 회/s → 공이 {'오른쪽' if side > 0 else '왼쪽'}으로 휨")
    return out
