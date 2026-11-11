"""공의 포물체 운동 + 이차 항력 비교용 수치적분."""

import numpy as np

G = 9.81              # 중력가속도 (m/s²)
BALL_RADIUS = 0.11    # 공 반지름 (m)
BALL_MASS = 0.43      # 공 질량 (kg)
AIR_DENSITY = 1.2     # 공기 밀도 ρ (kg/m³)
DRAG_COEF = 0.25      # 항력계수 C_d
BALL_AREA = 0.038     # 단면적 A (m²)


def flight_time(v: float, angle_deg: float, h0: float = 0.0) -> float:
    """비행시간 (s). h0 = 착지면 기준 발사 높이 (y = h0 + v sinθ t − ½gt²)."""
    vy = v * np.sin(np.radians(angle_deg))
    return float((vy + np.sqrt(vy**2 + 2 * G * h0)) / G)


def max_height(v: float, angle_deg: float, h0: float = 0.0) -> float:
    """최고점 높이 (m) = h0 + (v sinθ)² / 2g. 아래로 차면 h0."""
    vy = v * np.sin(np.radians(angle_deg))
    return float(h0 + max(vy, 0.0)**2 / (2 * G))


def horizontal_range(v: float, angle_deg: float, h0: float = 0.0) -> float:
    """비거리 (m) = v cosθ × 비행시간."""
    return float(v * np.cos(np.radians(angle_deg)) * flight_time(v, angle_deg, h0))


def trajectory(v: float, angle_deg: float, h0: float = 0.0, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """공기저항 없는 궤적 (x, y), 착지(y=0)까지."""
    T = flight_time(v, angle_deg, h0)
    t = np.linspace(0, T, n)
    a = np.radians(angle_deg)
    return v * np.cos(a) * t, h0 + v * np.sin(a) * t - 0.5 * G * t**2


def trajectory_drag(v: float, angle_deg: float, h0: float = 0.0,
                    dt: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이차 항력 F = ½ρC_dAv² 를 포함한 궤적 (t, x, y). RK4 수치적분."""
    k = 0.5 * AIR_DENSITY * DRAG_COEF * BALL_AREA / BALL_MASS

    def acc(vel: np.ndarray) -> np.ndarray:
        return np.array([0.0, -G]) - k * np.linalg.norm(vel) * vel

    a = np.radians(angle_deg)
    pos, vel = np.array([0.0, h0]), v * np.array([np.cos(a), np.sin(a)])
    ts, xs, ys = [0.0], [pos[0]], [pos[1]]
    t = 0.0
    while pos[1] >= 0 and t < 30:
        k1v, k1p = acc(vel), vel
        k2v, k2p = acc(vel + 0.5*dt*k1v), vel + 0.5*dt*k1v
        k3v, k3p = acc(vel + 0.5*dt*k2v), vel + 0.5*dt*k2v
        k4v, k4p = acc(vel + dt*k3v), vel + dt*k3v
        vel = vel + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
        pos = pos + dt/6 * (k1p + 2*k2p + 2*k3p + k4p)
        t += dt
        ts.append(t); xs.append(pos[0]); ys.append(pos[1])
    return np.array(ts), np.array(xs), np.array(ys)


def best_angle(v: float, step: float = 0.1) -> float:
    """수치 탐색으로 비거리가 최대인 발사각(도)을 찾는다."""
    angles = np.arange(step, 90, step)
    return float(angles[np.argmax([horizontal_range(v, a) for a in angles])])


def ball_center_at_contact(contact: np.ndarray, contact_height: float, direction: float = 1.0) -> np.ndarray:
    """접촉점에서 공 중심 위치를 구한다. 땅에 있는 공은 중심 높이 = 반지름."""
    r = BALL_RADIUS
    if contact_height <= r + 0.01:
        dx = np.sqrt(max(r**2 - (r - contact[1])**2, 0.0))
        return np.array([contact[0] + direction * dx, r])
    return np.array([contact[0] + direction * r, contact[1]])


def ball_path(t: np.ndarray, t_impact: float, center: np.ndarray, v: float, angle_deg: float,
              direction: float, mode: str) -> np.ndarray:
    """애니메이션용 공 중심 좌표 (n, 2).

    임팩트 전: mode = "ground"(정지), "incoming"(날아오는 포물선, 발리),
    "bounce"(바운드 직후, 하프 발리).
    임팩트 후: 공기저항 없는 포물체 운동.
    """
    out = np.zeros((len(t), 2))
    g = np.array([0.0, -G])
    a = np.radians(angle_deg)
    v_out = v * np.array([direction * np.cos(a), np.sin(a)])
    for i, ti in enumerate(t):
        s = ti - t_impact
        if s >= 0:
            out[i] = center + v_out * s + 0.5 * g * s**2
        elif mode == "incoming":
            out[i] = center + np.array([-4.0, -1.5]) * s + 0.5 * g * s**2   # 앞(+x)에서 날아옴
        elif mode == "bounce":
            out[i] = _bounce(center, s, direction)
        else:
            out[i] = center
    if mode != "incoming":
        out[:, 1] = np.maximum(out[:, 1], BALL_RADIUS)
    return out


def _bounce(center: np.ndarray, s: float, direction: float, dt_b: float = 0.06) -> np.ndarray:
    """임팩트 dt_b초 전에 바운드한 공 (반발계수 0.7)."""
    r, vx = BALL_RADIUS, -3.0 * direction
    vy_up = (center[1] - r + 0.5 * G * dt_b**2) / dt_b
    sb = s + dt_b                       # 바운드 시각 기준 시간
    if sb >= 0:
        y = r + vy_up * sb - 0.5 * G * sb**2
    else:
        y = r - (vy_up / 0.7) * sb - 0.5 * G * sb**2
    return np.array([center[0] + vx * s, y])

