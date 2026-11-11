"""3D 스틱 피규어 + 몸 윤곽 애니메이션 (마우스로 360° 회전, 시점 버튼, 자동 회전).

측면 2D 모델에 좌우 폭만 더한 것이라(biomech/body3d.py) 고관절 회전 같은 수평면 동작은 보이지 않는다.
Plotly 3D 축: x = 앞(차는 방향), y = 좌우(왼쪽 +), z = 위.
"""

import numpy as np
import plotly.graph_objects as go

from biomech.body3d import foot_yaw_series, to_3d
from biomech.kinematics import Motion, frame_points
from biomech.projectile import BALL_RADIUS
from viz.body_shapes import body_meshes, sphere
from viz.stickfigure import BALL, BODY, DARK, HIGHLIGHT, JOINT_LABELS, LIGHT, joint_label

MAX_FRAMES_3D = 70          # 3D는 프레임마다 다시 그리므로 프레임 수를 줄인다
CAM_DIST = 2.0
CAMERAS: dict[str, dict] = {
    "측면": dict(eye=dict(x=0.0, y=-CAM_DIST, z=0.35), up=dict(x=0, y=0, z=1)),
    "사선": dict(eye=dict(x=1.2, y=-1.5, z=0.6), up=dict(x=0, y=0, z=1)),
    "정면": dict(eye=dict(x=CAM_DIST, y=0.0, z=0.35), up=dict(x=0, y=0, z=1)),
    "후면": dict(eye=dict(x=-CAM_DIST, y=0.0, z=0.35), up=dict(x=0, y=0, z=1)),
    "위": dict(eye=dict(x=0.0, y=-0.01, z=2.6), up=dict(x=1, y=0, z=0)),
}
_SKELETON = {
    "l": ["l_hip", "l_knee", "l_ankle", "l_heel", "l_toe", "l_ankle", None, "l_shoulder", "l_elbow", "l_hand"],
    "r": ["r_hip", "r_knee", "r_ankle", "r_heel", "r_toe", "r_ankle", None, "r_shoulder", "r_elbow", "r_hand"],
    "body": ["l_hip", "r_hip", None, "l_shoulder", "r_shoulder", None, "hip", "shoulder", "head"],
}


def _p(v: np.ndarray) -> tuple[float, float, float]:
    """모델 (X, Y, Z) → 그림 (x, y, z) = (앞, 좌우, 위)."""
    return float(v[0]), float(v[2]), float(v[1])


def _line(p3: dict, names: list, color: str, width: int) -> go.Scatter3d:
    xs, ys, zs = [], [], []
    for n in names:
        x, y, z = _p(p3[n]) if n else (None, None, None)
        xs.append(x); ys.append(y); zs.append(z)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", showlegend=False)


def _mesh(verts: np.ndarray, faces: np.ndarray, color: str, opacity: float) -> go.Mesh3d:
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 2], z=verts[:, 1], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color=color, opacity=opacity, flatshading=False, hoverinfo="skip", showscale=False,
                     lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.8),
                     lightposition=dict(x=100, y=-200, z=300))


def frame_traces_3d(p2: dict[str, np.ndarray], angles: dict[str, float], highlight: list[str], H: float,
                    show_labels: bool, ball: np.ndarray | None, outline: bool, foot_yaw: float = 0.0,
                    foot: str = "R") -> list:
    """한 프레임의 3D 트레이스 (개수·순서 고정). ball = (x, y) 또는 (x, y, z).

    foot = "L"(왼발잡이): 오른발 모델을 좌우(Z) 거울상으로 뒤집는다. r_ = 차는 다리는 그대로 진한 색.
    """
    p3 = to_3d(p2, H, foot_yaw_deg=foot_yaw)
    flip = np.array([1.0, 1.0, -1.0 if foot == "L" else 1.0])
    p3 = {k: v * flip for k, v in p3.items()}
    traces = []
    if outline:
        meshes = body_meshes(p3, H)
        traces += [_mesh(*meshes["l"], LIGHT, 0.55), _mesh(*meshes["torso"], "#8d99ae", 0.5),
                   _mesh(*meshes["r"], DARK, 0.45)]
    else:
        empty = np.zeros((3, 3)), np.array([[0, 1, 2]])
        traces += [_mesh(*empty, LIGHT, 0.0) for _ in range(3)]
    traces += [_line(p3, _SKELETON["l"], LIGHT, 7), _line(p3, _SKELETON["body"], BODY, 8),
               _line(p3, _SKELETON["r"], DARK, 8)]
    hl = [k for k in highlight if k in JOINT_LABELS]
    pts = [_p(p3[JOINT_LABELS[k][0]]) for k in hl]
    traces.append(go.Scatter3d(x=[q[0] for q in pts], y=[q[1] for q in pts], z=[q[2] for q in pts],
                               mode="markers+text", marker=dict(size=7, color=HIGHLIGHT),
                               text=[f"{joint_label(k, foot)} {angles[k]:.0f}°" if show_labels else "" for k in hl],
                               textfont=dict(color=HIGHLIGHT, size=13), textposition="middle right",
                               hoverinfo="skip", showlegend=False))
    if ball is not None and np.all(np.isfinite(ball)):
        z = ball[2] if len(ball) > 2 else -0.055 * H         # 2D 경로면 차는 발 앞에 둔다
        c = np.array([ball[0], ball[1], z]) * flip
        bv, bf = sphere(c, BALL_RADIUS, 14, 9)
        traces.append(_mesh(bv, bf, BALL, 0.95))
    else:
        traces.append(_mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), BALL, 0.0))
    return traces


def _ground(xr: list[float], yr: list[float]) -> go.Mesh3d:
    x = [xr[0], xr[1], xr[1], xr[0]]
    y = [yr[0], yr[0], yr[1], yr[1]]
    return go.Mesh3d(x=x, y=y, z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3], color="#b7e4c7", opacity=0.6,
                     hoverinfo="skip", showscale=False)


def _scene(m: Motion, ball: np.ndarray | None, H: float) -> tuple[list, list, list]:
    pts = np.concatenate(list(m.points.values()))
    if ball is not None:
        pts = np.concatenate([pts, ball[np.all(np.isfinite(ball), axis=1), :2]])
    xr = [pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3]
    zr = [min(pts[:, 1].min(), 0) - 0.05, pts[:, 1].max() + 0.2]
    half = max(xr[1] - xr[0], zr[1] - zr[0]) / 2
    yr = [-0.6 * half, 0.6 * half]
    if ball is not None and ball.shape[1] > 2:          # 왼발잡이면 좌우가 뒤집히므로 대칭으로 넓힌다
        bz = np.abs(ball[np.isfinite(ball[:, 2]), 2]).max() + 0.2
        yr = [min(yr[0], -bz), max(yr[1], bz)]
    return xr, yr, zr


def cameras(foot: str = "R") -> dict[str, dict]:
    """시점 프리셋. 왼발잡이면 측면·사선 카메라를 반대쪽(차는 다리 쪽)으로 옮긴다."""
    if foot != "L":
        return CAMERAS
    out = {}
    for name, cam in CAMERAS.items():
        eye = dict(cam["eye"])
        if name in ("측면", "사선", "위"):
            eye["y"] = -eye["y"]
        out[name] = dict(eye=eye, up=cam["up"])
    return out


def _orbit(i: int, n: int, foot: str = "R") -> dict:
    """i번째 프레임의 자동 회전 카메라 (차는 다리 쪽 측면에서 시작해 한 바퀴)."""
    th = (np.pi / 2 if foot == "L" else -np.pi / 2) + 2 * np.pi * i / max(n, 1)
    return dict(eye=dict(x=CAM_DIST * np.cos(th), y=CAM_DIST * np.sin(th), z=0.45), up=dict(x=0, y=0, z=1))


def animated_figure_3d(kick: dict, m: Motion, H: float, opts: dict, speed: float,
                       ball: np.ndarray | None = None, spin: bool = False, camera: str = "사선") -> go.Figure:
    """3D 애니메이션. spin=True면 재생하는 동안 카메라가 몸 둘레를 한 바퀴 돈다."""
    idx = np.unique(np.linspace(0, len(m.t) - 1, min(MAX_FRAMES_3D, len(m.t))).round().astype(int))
    xr, yr, zr = _scene(m, ball, H)
    ground = _ground(xr, yr)
    kf_by_phase = {kf["phase"]: kf for kf in kick["keyframes"]}
    yaw = foot_yaw_series(kick, m.t)
    foot = opts.get("foot", "R")
    cams = cameras(foot)

    def data(i: int) -> list:
        p = frame_points(m, i)
        kf = kf_by_phase.get(m.phase[i], kick["keyframes"][0])
        return frame_traces_3d(p, {k: float(v[i]) for k, v in m.angles.items()}, kf["highlight"], H,
                               opts.get("labels", True), ball[i] if ball is not None else None,
                               opts.get("outline", True), float(yaw[i]), foot)

    n_dyn = len(data(idx[0]))
    frames = []
    for f, i in enumerate(idx):
        fr = dict(data=data(i), name=str(f), traces=list(range(1, n_dyn + 1)))
        if spin:
            fr["layout"] = dict(scene=dict(camera=_orbit(f, len(idx), foot)))
        frames.append(go.Frame(**fr))
    fig = go.Figure(data=[ground] + list(frames[0].data), frames=frames)
    dt_ms = float(np.mean(np.diff(m.t[idx]))) * 1000 / speed if len(idx) > 1 else 50.0
    play = dict(frame=dict(duration=dt_ms, redraw=True), transition=dict(duration=0), fromcurrent=True,
                mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))
    axis = lambda r, title: dict(range=r, title=title, backgroundcolor="white", gridcolor="#e9ecef",
                                 showspikes=False)
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=30, b=90), showlegend=False,
        scene=dict(xaxis=axis(xr, "앞 (m)"), yaxis=axis(yr, "좌우 (m)"), zaxis=axis(zr, "위 (m)"),
                   aspectmode="data", camera=cams.get(camera, cams["사선"]), dragmode="orbit"),
        updatemenus=[
            dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top", showactive=False,
                 buttons=[dict(label="▶ 재생", method="animate", args=[None, play]),
                          dict(label="⏸ 일시정지", method="animate", args=[[None], pause])]),
            dict(type="buttons", direction="left", x=1, y=1.04, xanchor="right", yanchor="bottom", showactive=False,
                 buttons=[dict(label=name, method="relayout", args=[{"scene.camera": cam}])
                          for name, cam in cams.items()]),
        ],
        sliders=[dict(active=0, x=0.22, len=0.78, y=-0.0, yanchor="top", pad=dict(t=0),
                      currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=13)),
                      steps=[dict(method="animate", label=f"{m.t[i]:.3f}", args=[[str(f)], pause])
                             for f, i in enumerate(idx)])],
    )
    return fig
