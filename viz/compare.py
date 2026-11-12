"""사용자 동작 vs 기준 템플릿 비교 그림: 스켈레톤 오버레이, 나란히/겹쳐 애니메이션, 각도 비교 그래프."""

import numpy as np
import plotly.graph_objects as go

from biomech.body import segment_lengths
from biomech.kick_loader import PHASES
from biomech.kinematics import Motion, compute_motion
from video.events import Events
from video.feedback import METRICS
from video.pose_series import Processed
from viz.stickfigure import JOINT_LABELS, _layout, add_animation_controls, frame_traces

MAX_FRAMES = 120            # 브라우저 부담을 줄이기 위한 애니메이션 최대 프레임 수
TEMPLATE_OPACITY = 0.45
USER_COLOR, TEMPLATE_COLOR = "#1f3b73", "#e76f51"

# 지표 → (관련 각도 키, 구간) — 각도 비교 그래프의 기준 범위 띠
METRIC_ANGLE: dict[str, tuple[str, str]] = {
    "max_r_knee_flex": ("r_knee", "swing"), "min_r_hip_ext": ("r_hip", "swing"),
    "impact_r_knee_flex": ("r_knee", "impact"), "impact_r_ankle_pf": ("r_ankle", "impact"),
    "impact_trunk": ("trunk", "impact"), "impact_l_knee_flex": ("l_knee", "impact"),
    "followthrough_r_hip_flex": ("r_hip", "follow"), "arm_balance": ("l_shoulder", "impact"),
}
ANGLE_NAMES = {"r_hip": "차는 다리 고관절 굴곡", "r_knee": "차는 다리 무릎 굴곡", "r_ankle": "차는 다리 발목 저측굴곡",
               "l_knee": "디딤 무릎 굴곡", "trunk": "몸통 기울기", "l_shoulder": "반대팔 어깨 굴곡"}

# 오버레이 색 (BGR): 차는 쪽 진하게, 디딤 쪽 연하게 (탭 1과 같은 규칙)
_BGR = {"r": (115, 59, 31), "l": (224, 180, 157), "body": (92, 65, 51), "hl": (70, 57, 230)}
_LIMB = [("hip", "knee"), ("knee", "ankle"), ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe"),
         ("shoulder", "elbow"), ("elbow", "wrist")]
_BONES = ([(f"l_{a}", f"l_{b}", "l") for a, b in _LIMB]
          + [("l_shoulder", "r_shoulder", "body"), ("l_hip", "r_hip", "body"),
             ("l_shoulder", "l_hip", "body"), ("r_shoulder", "r_hip", "body")]
          + [(f"r_{a}", f"r_{b}", "r") for a, b in _LIMB])
# 강조 관절 키 → 오버레이에서 표시할 랜드마크
_HL_LANDMARK = {"trunk": "r_shoulder", "r_hip": "r_hip", "r_knee": "r_knee", "r_ankle": "r_ankle",
                "l_hip": "l_hip", "l_knee": "l_knee", "l_ankle": "l_ankle", "r_shoulder": "r_shoulder",
                "l_shoulder": "l_shoulder", "r_elbow": "r_elbow", "l_elbow": "l_elbow"}


# ---------------------------------------------------------------------------
# 1) 영상 썸네일 위 스켈레톤
# ---------------------------------------------------------------------------
def overlay_skeleton(jpeg: bytes, xy_px: dict[str, np.ndarray], i: int, scale: float,
                     highlight: list[str]) -> np.ndarray:
    """i번째 프레임 썸네일(JPEG)에 관절 점·선을 그려 RGB 배열로 돌려준다. 강조 관절은 빨간색."""
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    pt = {k: (int(round(v[i, 0] * scale)), int(round(v[i, 1] * scale))) for k, v in xy_px.items()}
    w = max(2, img.shape[1] // 180)
    for a, b, color in _BONES:
        cv2.line(img, pt[a], pt[b], _BGR[color], w, cv2.LINE_AA)
    for k, p in pt.items():
        if k != "nose":
            cv2.circle(img, p, w + 1, (255, 255, 255), -1, cv2.LINE_AA)
    for key in highlight:
        lm = _HL_LANDMARK.get(key)
        if lm in pt:
            cv2.circle(img, pt[lm], w + 4, _BGR["hl"], -1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def phase_highlights(kick: dict) -> dict[str, list[str]]:
    """킥 템플릿 키프레임의 단계별 강조 관절."""
    return {kf["phase"]: kf["highlight"] for kf in kick["keyframes"]}


# ---------------------------------------------------------------------------
# 2·3) 나란히 / 겹쳐 애니메이션
# ---------------------------------------------------------------------------
def user_points(p: Processed) -> dict[str, np.ndarray]:
    """랜드마크 → 스틱 피규어(FK 이름) 좌표 시계열."""
    L = segment_lengths(p.H)
    xy = p.xy
    hip = (xy["l_hip"] + xy["r_hip"]) / 2
    shoulder = (xy["l_shoulder"] + xy["r_shoulder"]) / 2
    up = xy["nose"] - shoulder
    up = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    pts = {"hip": hip, "shoulder": shoulder, "head": shoulder + L["neck_head"] * up}
    for s in "rl":
        for k in ("hip", "knee", "ankle", "heel", "toe", "elbow"):
            pts[f"{s}_{k}"] = xy[f"{s}_{k}"]
        pts[f"{s}_hand"] = xy[f"{s}_wrist"]
    return pts


def user_phase(ev: Events, i: int) -> str:
    """사용자 i번째 프레임이 속한 단계 (검출된 이벤트 기준)."""
    name = PHASES[0]
    for ph in PHASES:
        f = ev.frames.get(ph)
        if f is not None and f <= i:
            name = ph
    return name


def _interp_points(pts: dict[str, np.ndarray], t: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.stack([np.interp(grid, t, v[:, 0]), np.interp(grid, t, v[:, 1])], axis=1) for k, v in pts.items()}


def comparison_figure(p: Processed, ev: Events, kick: dict, overlay: bool = False,
                      show_labels: bool = True) -> go.Figure:
    """왼쪽 = 내 동작, 오른쪽 = 기준 템플릿 (임팩트 시각을 맞춰 동기 재생).

    overlay=True면 디딤발 위치를 기준으로 겹친다 (내 동작 실선, 기준 반투명 점선).
    """
    H = p.H
    imp = ev.impact
    m: Motion = compute_motion(kick, H, 400)
    t_u = p.t - p.t[imp]
    t_k = m.t - kick["impact"]["time"]
    lo, hi = max(t_u[0], t_k[0]), min(t_u[-1], t_k[-1])
    n = int(min(MAX_FRAMES, max(2, round((hi - lo) * 100) + 1)))
    grid = np.linspace(lo, hi, n)

    up = _interp_points(user_points(p), t_u, grid)
    kp = _interp_points(m.points, t_k, grid)
    ua = {k: np.interp(grid, t_u, v) for k, v in p.angles.items()}
    ka = {k: np.interp(grid, t_k, v) for k, v in m.angles.items()}

    i0 = int(np.argmin(np.abs(grid)))       # 임팩트 프레임
    if overlay:
        if kick["stance_pin"] is None:
            ref_u, ref_k = up["hip"][i0], kp["hip"][i0]
        else:
            ref_u = (up["l_heel"][i0] + up["l_toe"][i0]) / 2
            ref_k = (kp["l_heel"][i0] + kp["l_toe"][i0]) / 2
        dx = ref_u[0] - ref_k[0]
    else:
        dx = np.concatenate([v[:, 0] for v in up.values()]).max() - \
             np.concatenate([v[:, 0] for v in kp.values()]).min() + 0.5
    kp = {k: v + np.array([dx, 0.0]) for k, v in kp.items()}

    allp = np.concatenate([np.concatenate(list(up.values())), np.concatenate(list(kp.values()))])
    xr = [allp[:, 0].min() - 0.25, allp[:, 0].max() + 0.4]
    yr = [min(allp[:, 1].min(), 0) - 0.1, allp[:, 1].max() + 0.35]
    text_xy = (xr[0] + 0.05, yr[1] - 0.12)
    hl = phase_highlights(kick)

    def frame_data(i: int) -> list[go.Scatter]:
        ph = user_phase(ev, int(np.argmin(np.abs(t_u - grid[i]))))
        uframe = {k: v[i] for k, v in up.items()}
        kframe = {k: v[i] for k, v in kp.items()}
        u_tr = frame_traces(uframe, {k: v[i] for k, v in ua.items()}, hl.get(ph, []), H, show_labels,
                            f"{ph}  ·  임팩트 기준 t = {grid[i]:+.3f} s", text_xy)
        k_tr = frame_traces(kframe, {k: v[i] for k, v in ka.items()}, [], H, False, "", text_xy)
        for tr in k_tr:
            if overlay:
                tr.opacity = TEMPLATE_OPACITY
                if tr.mode == "lines" and tr.fill is None:
                    tr.line.dash = "dot"
        return u_tr + k_tr

    frames = [go.Frame(data=frame_data(i), name=str(i)) for i in range(n)]
    fig = go.Figure(data=frames[0].data, frames=frames, layout=_layout(xr, yr))
    if not overlay:
        kx = float(np.median(kp["hip"][:, 0]))
        ux = float(np.median(up["hip"][:, 0]))
        fig.add_annotation(x=ux, y=yr[1] - 0.02, text="<b>내 동작</b>", showarrow=False, font=dict(color=USER_COLOR))
        fig.add_annotation(x=kx, y=yr[1] - 0.02, text=f"<b>기준: {kick['name_ko']}</b>", showarrow=False,
                           font=dict(color=TEMPLATE_COLOR))
    dt_ms = float(np.mean(np.diff(grid))) * 1000 / 0.25 if n > 1 else 40.0
    add_animation_controls(fig, [f"{g:+.3f}" for g in grid], dt_ms, prefix="임팩트 기준 t = ")
    fig.update_layout(xaxis_title="x (m) — 차는 방향 →")
    return fig


# ---------------------------------------------------------------------------
# 4) 각도 비교 그래프
# ---------------------------------------------------------------------------
def _windows(p: Processed, ev: Events) -> dict[str, tuple[float, float]]:
    imp = ev.impact
    t0 = p.t[imp]
    plant = ev.frames.get("디딤")
    start = p.t[plant] - t0 if plant is not None else -0.3
    ft = ev.frames.get("팔로스루")
    end = p.t[ft] - t0 if ft is not None else 0.2
    return {"swing": (start, 0.0), "impact": (-0.015, 0.015), "follow": (0.0, max(end, 0.03))}


def angle_compare_chart(p: Processed, ev: Events, key: str, template: Processed, t_impact_template: float,
                        kick_name: str, rules: list[dict]) -> go.Figure:
    """한 관절 각도: 내 시계열 + 기준 템플릿 시계열 + 기준 범위 띠 (x = 임팩트 기준 시간)."""
    t_u = p.t - p.t[ev.impact]
    t_k = template.t - t_impact_template
    yu, yk = p.angles[key], template.angles[key]
    fig = go.Figure()
    ymin = float(min(yu.min(), yk.min())) - 10
    ymax = float(max(yu.max(), yk.max())) + 10
    win = _windows(p, ev)
    for r in rules:
        link = METRIC_ANGLE.get(r["metric"])
        if link is None or link[0] != key or "good_range" not in r:
            continue
        lo, hi = r["good_range"]
        x0, x1 = win[link[1]]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=ymin if lo is None else lo, y1=ymax if hi is None else hi,
                      fillcolor="#2a9d8f", opacity=0.18, line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=ymax if hi is None else hi, yanchor="bottom", showarrow=False,
                           text=f"기준: {METRICS[r['metric']][0]}", font=dict(size=11, color="#2a9d8f"))
    fig.add_trace(go.Scatter(x=t_k, y=yk, name=f"기준: {kick_name}", mode="lines",
                             line=dict(color=TEMPLATE_COLOR, width=2.5, dash="dash"), hovertemplate="%{y:.0f}°"))
    fig.add_trace(go.Scatter(x=t_u, y=yu, name="내 동작", mode="lines",
                             line=dict(color=USER_COLOR, width=3), hovertemplate="%{y:.0f}°"))
    fig.add_vline(x=0, line=dict(color="#e63946", dash="dash", width=1.5), annotation_text="임팩트",
                  annotation_font=dict(color="#e63946"))
    fig.update_layout(height=340, plot_bgcolor="white", margin=dict(l=50, r=20, t=40, b=40),
                      title=dict(text=ANGLE_NAMES.get(key, key), font=dict(size=15)),
                      xaxis=dict(title="임팩트 기준 시간 (s)", gridcolor="#eef1f6",
                                 range=[max(t_u[0], t_k[0], -0.6), min(t_u[-1], t_k[-1], 0.4)]),
                      yaxis=dict(title="각도 (°)", gridcolor="#eef1f6", range=[ymin, ymax]),
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"), hovermode="x unified")
    return fig


def angle_label(key: str) -> str:
    return ANGLE_NAMES.get(key, JOINT_LABELS.get(key, ("", key))[1])


# ---------------------------------------------------------------------------
# 후면(관상면) 비교
# ---------------------------------------------------------------------------
REAR_ANGLE_NAMES = {"stance_knee_valgus": "디딤 무릎 외반", "trunk_lateral_lean": "몸통 옆 기울기 (디딤발 쪽 +)",
                    "pelvic_obliquity": "골반 기울기 (차는 쪽 들림 +)", "arm_abduction": "디딤발 쪽 팔 벌림"}
REAR_METRIC_ANGLE: dict[str, tuple[str, str]] = {
    "stance_knee_valgus": ("stance_knee_valgus", "plant"), "trunk_lateral_lean": ("trunk_lateral_lean", "impact"),
    "pelvic_obliquity": ("pelvic_obliquity", "impact"), "opposite_arm_abduction": ("arm_abduction", "impact"),
}
_REAR_LINES = [(["l_hip", "l_knee", "l_ankle", "l_heel", "l_toe", "l_ankle"], "#9db4e0", 6),
               (["l_shoulder", "l_elbow", "l_wrist"], "#9db4e0", 5),
               (["l_hip", "r_hip"], "#33415c", 6), (["l_shoulder", "r_shoulder"], "#33415c", 6),
               (["r_hip", "r_knee", "r_ankle", "r_heel", "r_toe", "r_ankle"], "#1f3b73", 7),
               (["r_shoulder", "r_elbow", "r_wrist"], "#1f3b73", 5)]


def rear_traces(xy: dict[str, np.ndarray], H: float, highlight: list[str], text: str,
                text_xy: tuple[float, float], faded: bool = False) -> list[go.Scatter]:
    """뒤에서 본 스틱 피규어 한 프레임 (x = 차는 다리 쪽 +, y = 위). 트레이스 개수 고정."""
    L = segment_lengths(H)
    hip_c = (xy["l_hip"] + xy["r_hip"]) / 2
    sh_c = (xy["l_shoulder"] + xy["r_shoulder"]) / 2
    up = (sh_c - hip_c) / (np.linalg.norm(sh_c - hip_c) or 1e-9)
    head = sh_c + L["neck_head"] * up
    th = np.linspace(0, 2 * np.pi, 33)
    op = TEMPLATE_OPACITY if faded else 1.0
    dash = "dot" if faded else None
    tr = [go.Scatter(x=[xy[n][0] for n in names], y=[xy[n][1] for n in names], mode="lines", opacity=op,
                     line=dict(color=c, width=w, dash=dash), hoverinfo="skip", showlegend=False)
          for names, c, w in _REAR_LINES]
    tr.append(go.Scatter(x=[hip_c[0], sh_c[0], head[0]], y=[hip_c[1], sh_c[1], head[1]], mode="lines", opacity=op,
                         line=dict(color="#33415c", width=7, dash=dash), hoverinfo="skip", showlegend=False))
    tr.append(go.Scatter(x=head[0] + L["head_radius"] * np.cos(th), y=head[1] + L["head_radius"] * np.sin(th),
                         mode="lines", fill="toself", fillcolor="#e9edf5", opacity=op,
                         line=dict(color="#33415c", width=3), hoverinfo="skip", showlegend=False))
    hl = [xy[k] for k in highlight if k in xy]
    tr.append(go.Scatter(x=[q[0] for q in hl], y=[q[1] for q in hl], mode="markers", opacity=op,
                         marker=dict(size=13, color="#e63946", line=dict(color="white", width=2)),
                         hoverinfo="skip", showlegend=False))
    tr.append(go.Scatter(x=[text_xy[0]], y=[text_xy[1]], mode="text", text=[text], textposition="middle right",
                         textfont=dict(size=16, color="#33415c"), hoverinfo="skip", showlegend=False))
    return tr


def template_rear_xy(kick: dict, H: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """템플릿을 3D 모델로 만들어 뒤에 둔 카메라로 원근으로 본 좌표 (시각은 임팩트 기준, 디딤발 깊이에서 m)."""
    from biomech.body3d import foot_yaw_series
    from video.synthetic import rear_camera_x, rear_landmarks
    m = compute_motion(kick, H, 300)
    return m.t - kick["impact"]["time"], rear_landmarks(m.points, H, rear_camera_x(kick, m),
                                                        foot_yaw_series(kick, m.t))


REAR_DEPTH_GUIDES_M = (-1.0, -0.5, 0.0, 0.5, 1.0)   # 디딤발(공) 기준 앞뒤 거리선 (m, + = 차는 방향)


def _rear_ground_y(offset_m: float, H: float) -> float:
    """디딤발에서 앞뒤로 offset_m 떨어진 바닥이 후면 카메라 화면에서 보이는 높이 (디딤발 깊이 기준 m)."""
    from video.synthetic import REAR_CAMERA_DIST_M as D, REAR_CAMERA_HEIGHT_RATIO
    yc = REAR_CAMERA_HEIGHT_RATIO * H
    return yc - yc * D / (D + offset_m)


def _rear_depth_guides(fig: go.Figure, kick: dict, H: float, dx: float) -> None:
    """기준 동작 쪽에 원근으로 본 바닥 거리선(디딤발 기준 앞뒤 0.5 m 간격)과 공을 그린다 (모든 프레임 공통 도형).

    원근에서는 먼 바닥일수록 화면 위쪽(카메라 높이 쪽)으로 모이므로, 발이 이 선들 사이에서 위아래로 움직이는 것이
    곧 앞뒤 움직임이다 (도움닫기·백스윙 = 아래쪽 선, 팔로스루 = 위쪽 선).
    """
    from biomech.body3d import side_z
    from video.synthetic import REAR_CAMERA_DIST_M as D, REAR_CAMERA_HEIGHT_RATIO
    yc = REAR_CAMERA_HEIGHT_RATIO * H
    half_w = 0.6
    for off in REAR_DEPTH_GUIDES_M:
        s = D / (D + off)
        y = _rear_ground_y(off, H)
        fig.add_shape(type="line", x0=dx - half_w * s, x1=dx + half_w * s, y0=y, y1=y, layer="below",
                      line=dict(color="#c9d3e3", width=2 if off == 0 else 1, dash=None if off == 0 else "dot"))
        fig.add_annotation(x=dx + half_w * s, y=y, xanchor="left", showarrow=False, font=dict(size=10, color="#8a97ad"),
                           text="디딤발" if off == 0 else f"{off:+.1f} m")
    # 공: 차는 다리 쪽, 디딤발보다 조금 앞 (반지름 0.11 m)
    r, off = 0.11, 0.10
    s = D / (D + off)
    bx, by = dx - side_z("r_ankle", H) * s, yc + (r - yc) * s
    fig.add_shape(type="circle", x0=bx - r * s, x1=bx + r * s, y0=by - r * s, y1=by + r * s, layer="below",
                  fillcolor="#f1f3f7", line=dict(color="#8a97ad", width=1.5))


def rear_comparison_figure(p: Processed, ev: Events, kick: dict, highlight: list[str] | None = None) -> go.Figure:
    """후면: 왼쪽 = 내 동작, 오른쪽 = 기준 템플릿(3D 모델을 뒤에서 본 모습). 임팩트 시각을 맞춰 재생."""
    H = p.H
    t_u = p.t - p.t[ev.impact]
    t_k, kxy = template_rear_xy(kick, H)
    lo, hi = max(t_u[0], t_k[0]), min(t_u[-1], t_k[-1])
    n = int(min(MAX_FRAMES, max(2, round((hi - lo) * 100) + 1)))
    grid = np.linspace(lo, hi, n)
    names = [k for k in p.xy if k != "nose"]
    up = _interp_points({k: p.xy[k] for k in names}, t_u, grid)
    kp = _interp_points({k: kxy[k] for k in names}, t_k, grid)
    # 바닥을 맞추고 템플릿을 오른쪽에 둔다
    span_u = np.concatenate([v[:, 0] for v in up.values()])
    span_k = np.concatenate([v[:, 0] for v in kp.values()])
    dx = span_u.max() - span_k.min() + 0.5
    kp = {k: v + np.array([dx, 0.0]) for k, v in kp.items()}
    allp = np.concatenate(list(up.values()) + list(kp.values()))
    xr = [allp[:, 0].min() - 0.3, max(allp[:, 0].max() + 0.3, dx + 1.2)]     # 바닥 거리선·라벨까지
    yr = [min(allp[:, 1].min(), _rear_ground_y(min(REAR_DEPTH_GUIDES_M), H), 0) - 0.1, allp[:, 1].max() + 0.45]
    text_xy = (xr[0] + 0.05, yr[1] - 0.1)
    hl = highlight if highlight is not None else ["l_knee"]

    def frame_data(i: int) -> list[go.Scatter]:
        ph = user_phase(ev, int(np.argmin(np.abs(t_u - grid[i]))))
        u = rear_traces({k: v[i] for k, v in up.items()}, H, hl, f"{ph}  ·  임팩트 기준 t = {grid[i]:+.3f} s",
                        text_xy)
        k = rear_traces({k: v[i] for k, v in kp.items()}, H, [], "", text_xy)
        return u + k

    frames = [go.Frame(data=frame_data(i), name=str(i)) for i in range(n)]
    fig = go.Figure(data=frames[0].data, frames=frames, layout=_layout(xr, yr))
    _rear_depth_guides(fig, kick, H, dx)
    fig.add_annotation(x=float(np.median(span_u)), y=yr[1] - 0.02, text="<b>내 동작 (뒤에서)</b>", showarrow=False,
                       font=dict(color=USER_COLOR))
    fig.add_annotation(x=float(np.median(span_k)) + dx, y=yr[1] - 0.02, text=f"<b>기준: {kick['name_ko']}</b>",
                       showarrow=False, font=dict(color=TEMPLATE_COLOR))
    dt_ms = float(np.mean(np.diff(grid))) * 1000 / 0.25 if n > 1 else 40.0
    add_animation_controls(fig, [f"{g:+.3f}" for g in grid], dt_ms, prefix="임팩트 기준 t = ")
    fig.update_layout(xaxis_title="좌우 (m) — 차는 다리 쪽 →")
    return fig


def rear_angle_chart(p: Processed, ev: Events, key: str, template: Processed | None, t_impact_template: float,
                     kick_name: str, rules: list[dict]) -> go.Figure:
    """후면 관상면 각도: 내 시계열 + 기준 템플릿(3D 모델) + 기준 범위 띠."""
    t_u = p.t - p.t[ev.impact]
    yu = p.angles[key]
    fig = go.Figure()
    ys = [yu] + ([template.angles[key]] if template is not None else [])
    ymin = float(min(y.min() for y in ys)) - 10
    ymax = float(max(y.max() for y in ys)) + 10
    plant = ev.frames.get("디딤")
    win = {"plant": ((p.t[plant] - p.t[ev.impact]) if plant is not None else -0.15, 0.05),
           "impact": (-0.015, 0.015)}
    for r in rules:
        link = REAR_METRIC_ANGLE.get(r["metric"])
        if link is None or link[0] != key:
            continue
        lo, hi = r["good_range"]
        x0, x1 = win[link[1]]
        y0, y1 = ymin if lo is None else lo, ymax if hi is None else hi
        ymin, ymax = min(ymin, y0 - 5), max(ymax, y1 + 5)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor="#2a9d8f", opacity=0.18,
                      line_width=0, layer="below")
    if template is not None:
        t_k = template.t - t_impact_template
        fig.add_trace(go.Scatter(x=t_k, y=template.angles[key], name=f"기준: {kick_name}", mode="lines",
                                 line=dict(color=TEMPLATE_COLOR, width=2.5, dash="dash"), hovertemplate="%{y:.0f}°"))
    fig.add_trace(go.Scatter(x=t_u, y=yu, name="내 동작", mode="lines", line=dict(color=USER_COLOR, width=3),
                             hovertemplate="%{y:.0f}°"))
    fig.add_vline(x=0, line=dict(color="#e63946", dash="dash", width=1.5), annotation_text="임팩트",
                  annotation_font=dict(color="#e63946"))
    fig.update_layout(height=320, plot_bgcolor="white", margin=dict(l=50, r=20, t=40, b=40),
                      title=dict(text=REAR_ANGLE_NAMES.get(key, key), font=dict(size=15)),
                      xaxis=dict(title="임팩트 기준 시간 (s)", gridcolor="#eef1f6",
                                 range=[max(t_u[0], -0.6), min(t_u[-1], 0.4)]),
                      yaxis=dict(title="각도 (°)", gridcolor="#eef1f6", range=[ymin, ymax]),
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"), hovermode="x unified")
    return fig
