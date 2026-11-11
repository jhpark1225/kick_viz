"""사람 윤곽(살) 도형: 2D 측면 다각형, 3D 원뿔대·구 메시."""

import numpy as np

from biomech.body3d import LIMB_RADII, TORSO_HALF_DEPTH, TORSO_HALF_WIDTH

LIMBS = [("hip", "knee", "thigh"), ("knee", "ankle", "shank"), ("ankle", "toe", "foot"),
         ("elbow", "hand", "forearm")]


# ---------------------------------------------------------------------------
# 2D (측면)
# ---------------------------------------------------------------------------
def capsule(a: np.ndarray, b: np.ndarray, ra: float, rb: float, n: int = 7) -> np.ndarray:
    """a→b 분절을 감싸는 캡슐(양 끝 반원 + 옆선) 다각형 꼭짓점 (m, 2)."""
    d = b - a
    L = float(np.hypot(*d)) or 1e-9
    ang = np.arctan2(d[1], d[0])
    th_b = ang + np.linspace(-np.pi / 2, np.pi / 2, n)
    th_a = ang + np.linspace(np.pi / 2, 3 * np.pi / 2, n)
    pb = b + rb * np.stack([np.cos(th_b), np.sin(th_b)], axis=1)
    pa = a + ra * np.stack([np.cos(th_a), np.sin(th_a)], axis=1)
    return np.concatenate([pb, pa, pb[:1]])


def _join(polys: list[np.ndarray]) -> tuple[list, list]:
    """여러 다각형을 None으로 구분한 한 트레이스 좌표로 합친다 (fill='toself'가 각각 채움)."""
    xs, ys = [], []
    for p in polys:
        xs += list(p[:, 0]) + [None]
        ys += list(p[:, 1]) + [None]
    return xs, ys


def side_outline(p: dict[str, np.ndarray], H: float) -> dict[str, tuple[list, list]]:
    """측면 윤곽: {'far': 디딤(왼)쪽 팔다리, 'torso': 몸통·목, 'near': 차는(오른)쪽 팔다리}."""
    out = {}
    for s, key in (("l", "far"), ("r", "near")):
        polys = []
        for a, b, seg in LIMBS:
            ra, rb = LIMB_RADII[seg]
            pa = p[f"{s}_{a}"] if a != "hip" else p[f"{s}_hip"]
            polys.append(capsule(pa, p[f"{s}_{b}"], ra * H, rb * H))
        ra, rb = LIMB_RADII["upper_arm"]
        polys.append(capsule(p["shoulder"], p[f"{s}_elbow"], ra * H, rb * H))
        polys.append(np.array([p[f"{s}_heel"], p[f"{s}_toe"], p[f"{s}_ankle"], p[f"{s}_heel"]]))  # 발바닥 삼각형
        out[key] = _join(polys)

    hip, sh = p["hip"], p["shoulder"]
    u = (sh - hip) / (np.linalg.norm(sh - hip) or 1e-9)      # 몸통 위 방향
    fwd = np.array([u[1], -u[0]])                             # 몸통 앞 방향 (위 방향을 시계로 90°)
    dh, dc = TORSO_HALF_DEPTH[0] * H, TORSO_HALF_DEPTH[1] * H
    chest = sh - 0.06 * H * u
    waist = hip + 0.45 * (sh - hip)
    torso = np.array([hip - dh * fwd - 0.02 * H * u,          # 엉덩이 아래
                      waist - 0.85 * dh * fwd, chest - dc * fwd, sh - 0.6 * dc * fwd + 0.02 * H * u,
                      sh + 0.6 * dc * fwd + 0.02 * H * u, chest + dc * fwd, waist + 0.8 * dh * fwd,
                      hip + dh * fwd - 0.01 * H * u, hip - dh * fwd - 0.02 * H * u])
    rn = LIMB_RADII["neck"][0] * H
    neck = capsule(sh, p["head"] - 0.04 * H * u, rn, rn)
    out["torso"] = _join([torso, neck])
    return out


# ---------------------------------------------------------------------------
# 3D 메시
# ---------------------------------------------------------------------------
def _basis(axis: np.ndarray, hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """axis에 수직인 두 단위벡터."""
    axis = axis / (np.linalg.norm(axis) or 1e-9)
    ref = hint if hint is not None else (np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0]))
    e1 = ref - axis * np.dot(ref, axis)
    e1 /= np.linalg.norm(e1) or 1e-9
    return e1, np.cross(axis, e1)


def frustum(a: np.ndarray, b: np.ndarray, ra: tuple[float, float] | float, rb: tuple[float, float] | float,
            n: int = 10, hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """a→b 원뿔대(타원 단면 가능) 메시: (꼭짓점 (2n+2, 3), 삼각형 (4n, 3)). ra/rb = 반지름 또는 (e1, e2) 반지름."""
    e1, e2 = _basis(b - a, hint)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    def ring(c, r):
        r1, r2 = (r, r) if np.isscalar(r) else r
        return c + np.outer(np.cos(th) * r1, e1) + np.outer(np.sin(th) * r2, e2)
    verts = np.concatenate([ring(a, ra), ring(b, rb), [a, b]])
    i = np.arange(n)
    j = (i + 1) % n
    side = np.concatenate([np.stack([i, j, n + i], 1), np.stack([j, n + j, n + i], 1)])
    caps = np.concatenate([np.stack([np.full(n, 2 * n), j, i], 1), np.stack([np.full(n, 2 * n + 1), n + i, n + j], 1)])
    return verts, np.concatenate([side, caps])


def sphere(c: np.ndarray, r: float, nu: int = 10, nv: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """구 메시."""
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, np.pi, nv)
    verts = np.array([c + r * np.array([np.sin(b) * np.cos(a), np.cos(b), np.sin(b) * np.sin(a)])
                      for b in v for a in u])
    faces = []
    for k in range(nv - 1):
        for m in range(nu):
            p0, p1 = k * nu + m, k * nu + (m + 1) % nu
            q0, q1 = p0 + nu, p1 + nu
            faces += [(p0, p1, q0), (p1, q1, q0)]
    return verts, np.array(faces)


def merge(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """여러 메시를 하나로 합친다 (삼각형 인덱스를 꼭짓점 오프셋만큼 이동)."""
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def body_meshes(p3: dict[str, np.ndarray], H: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """3D 몸 메시: {'l': 왼쪽 팔다리, 'torso': 몸통·목·머리, 'r': 오른쪽 팔다리}."""
    out = {}
    for s in "lr":
        parts = []
        for a, b, seg in LIMBS:
            ra, rb = LIMB_RADII[seg]
            parts.append(frustum(p3[f"{s}_{a}"], p3[f"{s}_{b}"], ra * H, rb * H))
        ra, rb = LIMB_RADII["upper_arm"]
        parts.append(frustum(p3[f"{s}_shoulder"], p3[f"{s}_elbow"], ra * H, rb * H))
        parts.append(frustum(p3[f"{s}_heel"], p3[f"{s}_toe"], 0.018 * H, 0.012 * H))
        for j, r in (("knee", 0.028), ("ankle", 0.019), ("elbow", 0.019), ("shoulder", 0.026)):
            parts.append(sphere(p3[f"{s}_{j}"], r * H, 8, 5))
        out[s] = merge(parts)
    hip_c = (p3["l_hip"] + p3["r_hip"]) / 2
    sh_c = (p3["l_shoulder"] + p3["r_shoulder"]) / 2
    lat = p3["l_hip"] - p3["r_hip"]                      # 골반 좌우 축 (차는 쪽 엉덩이 들림 반영)
    lat_s = p3["l_shoulder"] - p3["r_shoulder"]
    hw, cw = TORSO_HALF_WIDTH[0] * H, TORSO_HALF_WIDTH[1] * H
    hd, cd = TORSO_HALF_DEPTH[0] * H, TORSO_HALF_DEPTH[1] * H
    u = (sh_c - hip_c) / (np.linalg.norm(sh_c - hip_c) or 1e-9)
    parts = [
        frustum(hip_c - 0.03 * H * u, hip_c + 0.45 * (sh_c - hip_c), (hw, hd), (0.9 * hw, 0.85 * hd), 16, lat),
        frustum(hip_c + 0.45 * (sh_c - hip_c), sh_c, (0.9 * hw, 0.85 * hd), (cw, cd), 16, lat_s),
        frustum(sh_c, sh_c + 0.03 * H * u, (cw, cd), (0.6 * cw, 0.7 * cd), 16, lat_s),
        frustum(sh_c, p3["head"], LIMB_RADII["neck"][0] * H, LIMB_RADII["neck"][1] * H, 8),
        sphere(p3["head"], 0.06 * H, 12, 9),
        sphere(p3["l_hip"], 0.04 * H, 8, 5), sphere(p3["r_hip"], 0.04 * H, 8, 5),
    ]
    out["torso"] = merge(parts)
    return out
