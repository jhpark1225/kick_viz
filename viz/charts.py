"""관절 각도·각속도 그래프 (각운동학), 공 궤적 그래프 (포물체 운동)."""

import numpy as np
import plotly.graph_objects as go

from biomech.kinematics import Motion, angular_velocity, segment_abs_angles
from biomech.projectile import trajectory, trajectory_drag

PHASE_COLORS = {"접근": "#f1f3f5", "디딤": "#e7f5ff", "백스윙": "#fff4e6",
                "가속": "#fff0f6", "임팩트": "#ffe3e3", "팔로스루": "#ebfbee"}
LINE_COLORS = {"r_hip": "#1f3b73", "r_knee": "#e63946", "r_ankle": "#2a9d8f",
               "thigh": "#1f3b73", "shank": "#e63946"}


def _phase_bands(fig: go.Figure, kick: dict) -> None:
    kfs = kick["keyframes"]
    for a, b in zip(kfs, kfs[1:] + [None]):
        x1 = b["t"] if b else a["t"]
        if b is None:
            continue
        fig.add_vrect(x0=a["t"], x1=x1, fillcolor=PHASE_COLORS.get(a["phase"], "#f8f9fa"),
                      opacity=1, layer="below", line_width=0,
                      annotation_text=a["phase"], annotation_position="top left",
                      annotation_font=dict(size=11, color="#495057"))
    t_imp = kick["impact"]["time"]
    fig.add_vline(x=t_imp, line=dict(color="#e63946", dash="dash", width=1.5),
                  annotation_text=f"임팩트 {t_imp:.2f} s", annotation_position="bottom right",
                  annotation_font=dict(color="#e63946"))


def _base_layout(fig: go.Figure, ytitle: str) -> go.Figure:
    fig.update_layout(height=380, plot_bgcolor="white", margin=dict(l=50, r=20, t=40, b=40),
                      xaxis=dict(title="시간 (s)", gridcolor="#eef1f6"),
                      yaxis=dict(title=ytitle, gridcolor="#eef1f6", zerolinecolor="#ced4da"),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                      hovermode="x unified")
    return fig


def angle_chart(kick: dict, m: Motion) -> go.Figure:
    """오른쪽(차는) 다리의 고관절·무릎·발목 각도."""
    fig = go.Figure()
    _phase_bands(fig, kick)
    for key, name in (("r_hip", "고관절 굴곡"), ("r_knee", "무릎 굴곡"), ("r_ankle", "발목 저측굴곡")):
        fig.add_trace(go.Scatter(x=m.t, y=m.angles[key], name=name, mode="lines",
                                 line=dict(color=LINE_COLORS[key], width=3),
                                 hovertemplate="%{y:.0f}°"))
    return _base_layout(fig, "관절 각도 (°)")


def segment_velocities(m: Motion) -> tuple[np.ndarray, np.ndarray]:
    """허벅지·정강이 절대 분절 각속도 (°/s)."""
    thigh, shank = segment_abs_angles(m.angles["trunk"], m.angles["r_hip"], m.angles["r_knee"])
    return angular_velocity(m.t, thigh), angular_velocity(m.t, shank)


def velocity_chart(kick: dict, m: Motion) -> go.Figure:
    """허벅지·정강이 절대 각속도와 최고점 마커 (근위→원위 순차)."""
    w_thigh, w_shank = segment_velocities(m)
    fig = go.Figure()
    _phase_bands(fig, kick)
    for key, name, w in (("thigh", "허벅지", w_thigh), ("shank", "정강이", w_shank)):
        i = int(np.argmax(w))
        fig.add_trace(go.Scatter(x=m.t, y=w, name=f"{name} 각속도", mode="lines",
                                 line=dict(color=LINE_COLORS[key], width=3),
                                 hovertemplate="%{y:.0f}°/s"))
        fig.add_trace(go.Scatter(x=[m.t[i]], y=[w[i]], mode="markers+text", showlegend=False,
                                 marker=dict(size=13, color=LINE_COLORS[key], symbol="star",
                                             line=dict(color="white", width=1)),
                                 text=[f"{name} 최고 {w[i]:.0f}°/s @ {m.t[i]:.3f} s"],
                                 textposition="top left" if key == "thigh" else "top right",
                                 hoverinfo="skip"))
    return _base_layout(fig, "절대 분절 각속도 (°/s)")


def trajectory_chart(v: float, angle: float, h0: float, with_drag: bool) -> go.Figure:
    """공 궤적: 공기저항 없음(실선) vs 이차 항력(점선)."""
    x, y = trajectory(v, angle, h0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name="공기저항 없음 (포물선)", mode="lines",
                             line=dict(color="#1f3b73", width=3),
                             hovertemplate="x %{x:.1f} m, y %{y:.2f} m"))
    i = int(np.argmax(y))
    fig.add_trace(go.Scatter(x=[x[i]], y=[y[i]], mode="markers+text", showlegend=False,
                             marker=dict(size=10, color="#1f3b73"), text=[f"최고점 {y[i]:.2f} m"],
                             textposition="top center", hoverinfo="skip"))
    if with_drag:
        _, xd, yd = trajectory_drag(v, angle, h0)
        fig.add_trace(go.Scatter(x=xd, y=yd, name="공기저항 포함", mode="lines",
                                 line=dict(color="#e63946", width=3, dash="dash"),
                                 hovertemplate="x %{x:.1f} m, y %{y:.2f} m"))
    fig.update_layout(height=400, plot_bgcolor="white", margin=dict(l=50, r=20, t=30, b=40),
                      xaxis=dict(title="수평 거리 (m)", gridcolor="#eef1f6", rangemode="tozero"),
                      yaxis=dict(title="높이 (m)", gridcolor="#eef1f6", rangemode="tozero"),
                      legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
    return fig
