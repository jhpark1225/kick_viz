"""Plotly 스틱 피규어: 정지 자세 + frames 애니메이션."""

import numpy as np
import plotly.graph_objects as go

from biomech.body import segment_lengths
from biomech.kinematics import Motion, center_of_mass, frame_points
from biomech.projectile import BALL_RADIUS
from viz.body_shapes import side_outline

DARK = "#1f3b73"        # 오른쪽(차는) 다리·팔
LIGHT = "#9db4e0"       # 왼쪽(디딤) 다리·팔
BODY = "#33415c"
HIGHLIGHT = "#e63946"
BALL = "#f4a261"
SKIN_NEAR = "rgba(31, 59, 115, 0.22)"    # 차는(오른)쪽 팔다리 윤곽
SKIN_FAR = "rgba(157, 180, 224, 0.28)"   # 디딤(왼)쪽 팔다리 윤곽
SKIN_BODY = "rgba(51, 65, 92, 0.24)"     # 몸통·목 윤곽

# 관절 각도 키 → (그림 위 위치, 한국어 라벨)
JOINT_LABELS: dict[str, tuple[str, str]] = {
    "trunk": ("shoulder", "몸통"),
    "r_hip": ("r_hip", "고관절"), "r_knee": ("r_knee", "무릎"), "r_ankle": ("r_ankle", "발목"),
    "l_hip": ("l_hip", "디딤 고관절"), "l_knee": ("l_knee", "디딤 무릎"), "l_ankle": ("l_ankle", "디딤 발목"),
    "r_shoulder": ("shoulder", "오른 어깨"), "l_shoulder": ("shoulder", "왼 어깨"),
    "r_elbow": ("r_elbow", "오른 팔꿈치"), "l_elbow": ("l_elbow", "왼 팔꿈치"),
}
JOINT_DOTS = ["hip", "r_hip", "shoulder", "r_knee", "r_ankle", "l_knee", "l_ankle",
              "r_elbow", "l_elbow", "r_hand", "l_hand"]
_CIRCLE = np.linspace(0, 2 * np.pi, 33)


def swap_lr(text: str) -> str:
    """'오른' ↔ '왼'을 맞바꾼다 (왼발잡이일 때 설명·라벨용)."""
    return str(text).replace("오른", "\0").replace("왼", "오른").replace("\0", "왼")


def joint_label(key: str, foot: str = "R") -> str:
    """관절 각도 키의 한국어 라벨. 왼발잡이면 '오른/왼'을 바꾼다 (r_ = 차는 쪽은 그대로)."""
    label = JOINT_LABELS[key][1]
    return swap_lr(label) if foot == "L" else label


def _xy(p: dict[str, np.ndarray], names: list[str]) -> tuple[list[float], list[float]]:
    return [p[n][0] for n in names], [p[n][1] for n in names]


def _circle(c: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    return c[0] + r * np.cos(_CIRCLE), c[1] + r * np.sin(_CIRCLE)


def outline_traces(p: dict[str, np.ndarray] | None, H: float) -> list[go.Scatter]:
    """사람 윤곽(살) 다각형 3개: 디딤(왼)쪽 → 몸통 → 차는(오른)쪽 순서로 뒤에서 앞으로 그린다.

    p가 None이면 빈 트레이스(애니메이션 트레이스 개수 유지용).
    """
    shapes = side_outline(p, H) if p is not None else {k: ([], []) for k in ("far", "torso", "near")}
    return [go.Scatter(x=shapes[k][0], y=shapes[k][1], mode="lines", fill="toself", fillcolor=fill,
                       line=dict(color=edge, width=1.5), hoverinfo="skip", showlegend=False)
            for k, fill, edge in (("far", SKIN_FAR, LIGHT), ("torso", SKIN_BODY, BODY), ("near", SKIN_NEAR, DARK))]


def frame_traces(p: dict[str, np.ndarray], angles: dict[str, float], highlight: list[str],
                 H: float, show_labels: bool, phase_text: str, text_xy: tuple[float, float],
                 trail: np.ndarray | None = None, com: np.ndarray | None = None,
                 ball: np.ndarray | None = None, outline: bool = False, foot: str = "R") -> list[go.Scatter]:
    """한 프레임의 트레이스 목록. 애니메이션을 위해 트레이스 개수·순서는 항상 같다.

    foot = "L"(왼발잡이)이어도 측면에서 본 그림은 같다(좌우 거울상은 측면 투영이 같음). 라벨만 바꾼다.
    """
    L = segment_lengths(H)
    line = lambda names, color, w: go.Scatter(
        x=_xy(p, names)[0], y=_xy(p, names)[1], mode="lines",
        line=dict(color=color, width=w), hoverinfo="skip", showlegend=False)
    neck_end = p["head"] - L["head_radius"] * (p["head"] - p["shoulder"]) / np.linalg.norm(p["head"] - p["shoulder"])
    hx, hy = _circle(p["head"], L["head_radius"])

    hl_x, hl_y, hl_text = [], [], []
    for key in highlight:
        where, label = JOINT_LABELS[key][0], joint_label(key, foot)
        hl_x.append(p[where][0]); hl_y.append(p[where][1])
        hl_text.append(f"{label} {angles[key]:.0f}°" if show_labels else "")

    bx, by = _circle(ball, BALL_RADIUS) if ball is not None else ([], [])
    traces = outline_traces(p if outline else None, H) + [
        line(["l_hip", "l_knee", "l_ankle", "l_heel", "l_toe", "l_ankle"], LIGHT, 6),
        line(["shoulder", "l_elbow", "l_hand"], LIGHT, 5),
        go.Scatter(x=[p["r_hip"][0], p["hip"][0], p["shoulder"][0], neck_end[0]],
                   y=[p["r_hip"][1], p["hip"][1], p["shoulder"][1], neck_end[1]],
                   mode="lines", line=dict(color=BODY, width=7), hoverinfo="skip", showlegend=False),
        go.Scatter(x=hx, y=hy, mode="lines", fill="toself", fillcolor="#e9edf5",
                   line=dict(color=BODY, width=3), hoverinfo="skip", showlegend=False),
        line(["r_hip", "r_knee", "r_ankle", "r_heel", "r_toe", "r_ankle"], DARK, 7),
        line(["shoulder", "r_elbow", "r_hand"], DARK, 5),
        go.Scatter(x=_xy(p, JOINT_DOTS)[0], y=_xy(p, JOINT_DOTS)[1], mode="markers",
                   marker=dict(size=8, color="white", line=dict(color=BODY, width=2)),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=hl_x, y=hl_y, mode="markers+text", text=hl_text, textposition="middle right",
                   textfont=dict(size=14, color=HIGHLIGHT),
                   marker=dict(size=13, color=HIGHLIGHT, line=dict(color="white", width=2)),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=trail[:, 0] if trail is not None else [], y=trail[:, 1] if trail is not None else [],
                   mode="lines", line=dict(color=DARK, width=2, dash="dot"), opacity=0.5,
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=[com[0]] if com is not None else [], y=[com[1]] if com is not None else [],
                   mode="markers+text", text=["무게중심"], textposition="bottom center",
                   marker=dict(size=12, color="#2a9d8f", symbol="circle-cross", line=dict(width=2)),
                   textfont=dict(color="#2a9d8f"), hoverinfo="skip", showlegend=False),
        go.Scatter(x=bx, y=by, mode="lines", fill="toself", fillcolor=BALL,
                   line=dict(color="#9c6644", width=2), hoverinfo="skip", showlegend=False),
        go.Scatter(x=[text_xy[0]], y=[text_xy[1]], mode="text", text=[phase_text],
                   textposition="middle right", textfont=dict(size=18, color=BODY),
                   hoverinfo="skip", showlegend=False),
    ]
    return traces


def _bounds(m: Motion, ball: np.ndarray | None, t_impact: float) -> tuple[list[float], list[float]]:
    pts = np.concatenate(list(m.points.values()))
    if ball is not None:
        pts = np.concatenate([pts, ball[m.t <= t_impact + 0.02]])
    x0, x1 = pts[:, 0].min() - 0.25, pts[:, 0].max() + 0.6
    y0, y1 = min(pts[:, 1].min(), 0) - 0.1, pts[:, 1].max() + 0.35
    return [x0, x1], [y0, y1]


def _layout(xr: list[float], yr: list[float], height: int = 560) -> dict:
    return dict(
        xaxis=dict(range=xr, title="x (m) — 차는 방향 →", zeroline=False, gridcolor="#eef1f6"),
        yaxis=dict(range=yr, title="y (m)", scaleanchor="x", scaleratio=1, zeroline=False,
                   gridcolor="#eef1f6"),
        height=height, margin=dict(l=40, r=20, t=30, b=40), plot_bgcolor="white",
        shapes=[dict(type="line", x0=xr[0], x1=xr[1], y0=0, y1=0, line=dict(color="#6b705c", width=3)),
                dict(type="rect", x0=xr[0], x1=xr[1], y0=yr[0], y1=0, fillcolor="#e9f5db",
                     line=dict(width=0), layer="below")],
    )


def _frame_args(kick: dict, m: Motion, i: int, H: float, opts: dict, ball: np.ndarray | None,
                text_xy: tuple[float, float], trail_from: int = 0) -> list[go.Scatter]:
    p = frame_points(m, i)
    kf = _keyframe_for(kick, m.phase[i])
    angles = {k: float(v[i]) for k, v in m.angles.items()}
    trail = m.points["r_toe"][trail_from:i + 1] if opts.get("trail") else None
    com = center_of_mass(p) if opts.get("com") else None
    return frame_traces(p, angles, kf["highlight"], H, opts.get("labels", True),
                        f"{m.phase[i]}  ·  t = {m.t[i]:.3f} s", text_xy, trail, com,
                        ball[i] if ball is not None else None, outline=opts.get("outline", False),
                        foot=opts.get("foot", "R"))


def _keyframe_for(kick: dict, phase: str) -> dict:
    return next(kf for kf in kick["keyframes"] if kf["phase"] == phase)


def static_figure(kick: dict, m: Motion, i: int, H: float, opts: dict,
                  ball: np.ndarray | None = None, height: int = 480) -> go.Figure:
    """i번째 프레임의 정지 그림."""
    xr, yr = _bounds(m, ball, kick["impact"]["time"])
    traces = _frame_args(kick, m, i, H, opts, ball, (xr[0] + 0.05, yr[1] - 0.12))
    return go.Figure(data=traces, layout=_layout(xr, yr, height))


def animated_figure(kick: dict, m: Motion, H: float, opts: dict, speed: float,
                    ball: np.ndarray | None = None) -> go.Figure:
    """재생/일시정지 버튼과 프레임 슬라이더가 있는 애니메이션."""
    xr, yr = _bounds(m, ball, kick["impact"]["time"])
    text_xy = (xr[0] + 0.05, yr[1] - 0.12)
    frames = [go.Frame(data=_frame_args(kick, m, i, H, opts, ball, text_xy), name=str(i))
              for i in range(len(m.t))]
    dt_ms = float(np.mean(np.diff(m.t))) * 1000 / speed
    fig = go.Figure(data=frames[0].data, frames=frames, layout=_layout(xr, yr))
    add_animation_controls(fig, [f"{t:.3f}" for t in m.t], dt_ms)
    return fig


def add_animation_controls(fig: go.Figure, labels: list[str], dt_ms: float,
                           prefix: str = "t = ") -> None:
    """재생/일시정지 버튼과 프레임 슬라이더를 붙인다. 프레임 이름은 '0', '1', ... 이어야 한다."""
    play = dict(frame=dict(duration=dt_ms, redraw=False), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.12, xanchor="left", yanchor="top",
                          pad=dict(r=10, t=0), showactive=False, buttons=[
                              dict(label="▶ 재생", method="animate", args=[None, play]),
                              dict(label="⏸ 일시정지", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                          ])],
        sliders=[dict(active=0, x=0.22, len=0.78, y=-0.08, yanchor="top",
                      currentvalue=dict(prefix=prefix, suffix=" s", font=dict(size=13)),
                      pad=dict(t=0),
                      steps=[dict(method="animate", label=lab,
                                  args=[[str(i)], dict(frame=dict(duration=0, redraw=False),
                                                       mode="immediate", transition=dict(duration=0))])
                             for i, lab in enumerate(labels)])],
        margin=dict(l=40, r=20, t=30, b=110), height=640,
    )
