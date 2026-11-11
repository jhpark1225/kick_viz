"""발 방향·접촉 위치 → 공 방향·회전·궤적 그림 (탭 3)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from biomech.ball_flight import Flight
from biomech.projectile import BALL_RADIUS

FOOT_LEN, FOOT_WID = 0.26, 0.10      # 위에서 본 발 크기 (m)
SURFACE_NAMES = {"instep": "발등", "inside": "발 안쪽", "outside": "발 바깥쪽", "toe": "발끝", "under": "발끝(공 아래)"}
CUR, REF = "#e63946", "#adb5bd"


def _foot_outline() -> np.ndarray:
    """오른발을 위에서 본 윤곽 (발 좌표: f = 뒤꿈치→발끝, w = 안쪽 +). 왼발은 화면에서 좌우 반전."""
    th = np.linspace(0, 2 * np.pi, 60)
    f = FOOT_LEN / 2 * (1 - np.cos(th))
    w = FOOT_WID / 2 * np.sin(th) * (0.75 + 0.35 * f / FOOT_LEN)    # 앞볼이 넓다
    w = w - 0.012 * np.exp(-((f - 0.45 * FOOT_LEN) / 0.05) ** 2) * (w > 0)   # 안쪽 아치
    return np.stack([f, w], axis=1)


def _surface_point(kind: str) -> tuple[float, float]:
    """접촉면 위치 (발 좌표)."""
    return {"inside": (0.5 * FOOT_LEN, 0.85 * FOOT_WID / 2), "outside": (0.72 * FOOT_LEN, -0.9 * FOOT_WID / 2),
            "toe": (FOOT_LEN, 0.0), "under": (FOOT_LEN, 0.0)}.get(kind, (0.62 * FOOT_LEN, 0.0))


def foot_topview(yaw_deg: float, contact: str, az_deg: float, side_spin_rps: float, left: bool = False) -> go.Figure:
    """위에서 본 차는 발(발끝 바깥으로 돌림)과 공이 나가는 방향. left=True면 왼발(좌우 거울상).

    az_deg·side_spin_rps는 실제 값(왼쪽 +, 오른쪽으로 휘면 +)이다.
    """
    psi = np.radians(yaw_deg)
    sx = -1.0 if left else 1.0

    def to_screen(f, w):
        # 발 좌표 → 화면 (x = 오른쪽 +, y = 앞 +). 발끝 바깥으로 psi만큼 돌림 (왼발은 좌우 반전)
        return sx * (f * np.sin(psi) - w * np.cos(psi)), f * np.cos(psi) + w * np.sin(psi)

    o = _foot_outline()
    fx, fy = to_screen(o[:, 0], o[:, 1])
    sf, sw = _surface_point(contact)
    cx, cy = to_screen(sf, sw)
    a = np.radians(az_deg)
    d = np.array([-np.sin(a), np.cos(a)])                  # 공 방향 (화면): az + = 왼쪽
    ball = np.array([cx, cy]) + d * (BALL_RADIUS + 0.01)
    th = np.linspace(0, 2 * np.pi, 40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx, y=fy, fill="toself", fillcolor="rgba(31,59,115,0.25)", mode="lines",
                             line=dict(color="#1f3b73", width=2), name="왼발" if left else "오른발", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text", marker=dict(size=11, color=CUR),
                             text=[SURFACE_NAMES.get(contact, contact)], textposition="bottom center",
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ball[0] + BALL_RADIUS * np.cos(th), y=ball[1] + BALL_RADIUS * np.sin(th),
                             fill="toself", fillcolor="#f4a261", mode="lines", line=dict(color="#9c6644"),
                             showlegend=False, hoverinfo="skip"))
    tip = ball + d * 0.45
    fig.add_annotation(x=tip[0], y=tip[1], ax=ball[0], ay=ball[1], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=CUR, text="")
    if abs(side_spin_rps) >= 0.3:
        bend = -1 if side_spin_rps < 0 else 1      # 화면 기준: 왼쪽으로 휨 = −x
        fig.add_annotation(x=tip[0] + bend * 0.12, y=tip[1] + 0.02, text="휨 ↶" if bend < 0 else "휨 ↷",
                           showarrow=False, font=dict(color=CUR, size=13))
    fig.add_annotation(x=0, y=-0.08, text="뒤꿈치", showarrow=False, font=dict(size=11, color="#495057"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#e9f5db", showlegend=False,
                      title=dict(text="위에서 본 발 방향 → 공 방향", font=dict(size=14)),
                      xaxis=dict(range=[-0.45, 0.45], showticklabels=False, zeroline=False, showgrid=False),
                      yaxis=dict(range=[-0.12, 0.85], scaleanchor="x", showticklabels=False, zeroline=False,
                                 showgrid=False))
    return fig


def contact_diagram(side_cm: float, up_cm: float) -> go.Figure:
    """뒤에서 본 공: 발이 닿는 위치 (중심에서 벗어날수록 회전이 커짐)."""
    r = BALL_RADIUS * 100
    th = np.linspace(0, 2 * np.pi, 60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r * np.cos(th), y=r * np.sin(th), fill="toself", fillcolor="#f4a261",
                             mode="lines", line=dict(color="#9c6644", width=2), hoverinfo="skip"))
    for x0, y0, x1, y1 in ((-r, 0, r, 0), (0, -r, 0, r)):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#9c6644", dash="dot", width=1))
    fig.add_trace(go.Scatter(x=[side_cm], y=[up_cm], mode="markers+text", text=["발이 닿는 곳"],
                             textposition="top center", marker=dict(size=16, color=CUR, symbol="x-thin",
                                                                    line=dict(width=4, color=CUR)),
                             hoverinfo="skip"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", showlegend=False,
                      title=dict(text="뒤에서 본 공의 접촉 위치 (cm)", font=dict(size=14)),
                      xaxis=dict(range=[-14, 14], title="← 왼쪽 · 오른쪽 →", zeroline=False),
                      yaxis=dict(range=[-14, 14], scaleanchor="x", title="아래 · 위", zeroline=False))
    return fig


def flight_views(cur: Flight, ref: Flight | None) -> go.Figure:
    """위에서 본 궤적(좌우 휨) + 옆에서 본 궤적(높이). ref = 같은 방향·속도로 회전 없이 찬 공."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=("위에서 본 궤적 (좌우로 휘는 정도)", "옆에서 본 궤적 (높이)"))
    for f, name, color, dash in ((ref, "회전 없을 때", REF, "dash"), (cur, "현재 설정", CUR, None)):
        if f is None:
            continue
        fig.add_trace(go.Scatter(x=-f.pos[:, 2], y=f.pos[:, 0], name=name, mode="lines",
                                 line=dict(color=color, width=3, dash=dash), legendgroup=name,
                                 hovertemplate="좌우 %{x:.2f} m, 앞 %{y:.1f} m"), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.hypot(f.pos[:, 0], f.pos[:, 2]), y=f.pos[:, 1], name=name, mode="lines",
                                 line=dict(color=color, width=3, dash=dash), legendgroup=name, showlegend=False,
                                 hovertemplate="거리 %{x:.1f} m, 높이 %{y:.2f} m"), row=1, col=2)
    lat = max(1.5, float(np.abs(cur.pos[:, 2]).max()) * 1.3,
              float(np.abs(ref.pos[:, 2]).max()) * 1.3 if ref is not None else 0)
    fig.update_xaxes(title_text="← 왼쪽 · 오른쪽 → (m)", range=[-lat, lat], gridcolor="#eef1f6", row=1, col=1)
    fig.update_yaxes(title_text="앞 (m)", rangemode="tozero", gridcolor="#eef1f6", row=1, col=1)
    fig.update_xaxes(title_text="수평 거리 (m)", rangemode="tozero", gridcolor="#eef1f6", row=1, col=2)
    fig.update_yaxes(title_text="높이 (m)", rangemode="tozero", gridcolor="#eef1f6", row=1, col=2)
    fig.update_layout(height=420, plot_bgcolor="white", margin=dict(l=50, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
    return fig
