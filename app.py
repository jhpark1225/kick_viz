"""축구 킥 생체역학 스틱 피규어 웹 (Streamlit 진입점). 실행: streamlit run app.py"""

import numpy as np
import pandas as pd
import streamlit as st

from biomech.body import DEFAULT_HEIGHT_M, MAX_HEIGHT_CM, MIN_HEIGHT_CM
from biomech.kick_loader import PHASES, available_kicks, load_kick
from biomech.kinematics import (ANGLE_KEYS, CONTACT_NAMES, compute_motion, contact_point,
                                frame_points, impact_analysis)
from biomech.projectile import (BALL_MASS, BALL_RADIUS, G, ball_center_at_contact, ball_path,
                                flight_time, horizontal_range, max_height)
from content.excel_loader import concept_row, load_kick_content, phase_supplement
from viz.charts import angle_chart, segment_velocities, trajectory_chart, velocity_chart
from viz.stickfigure import animated_figure, joint_label, static_figure, swap_lr
from viz.stickfigure3d import CAMERAS, animated_figure_3d
from biomech.ball_flight import (STEER, ball_path_3d, foot_params, kick_flight, simulate, spin_description)
from biomech.body3d import foot_yaw_series, to_3d
from viz.ball_charts import contact_diagram, flight_views, foot_topview
import video_tab

CONTACT_TIME = 0.010    # 임팩트 접촉시간 (s), 선운동역학
SPEEDS = [0.1, 0.25, 0.5, 0.75, 1.0]

# `python app.py`(VS Code 실행 버튼)로 실행해도 Streamlit 서버로 다시 띄운다
if __name__ == "__main__" and not st.runtime.exists():
    import sys
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(cli.main())

st.set_page_config(page_title="축구 킥 생체역학", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# 캐시된 계산 (킥 + 키 조합)
# ---------------------------------------------------------------------------
@st.cache_data
def get_kick(kick_id: str) -> dict:
    """킥 JSON."""
    return load_kick(kick_id)


@st.cache_data
def get_motion(kick_id: str, H: float):
    """애니메이션용 동작 (약 120프레임)."""
    return compute_motion(get_kick(kick_id), H)


@st.cache_data
def get_dense_motion(kick_id: str, H: float):
    """그래프용 촘촘한 동작."""
    return compute_motion(get_kick(kick_id), H, 400)


@st.cache_data
def get_impact(kick_id: str, H: float) -> dict:
    """임팩트 발 속도·공 속도."""
    return impact_analysis(get_kick(kick_id), H)


@st.cache_data
def get_ball(kick_id: str, H: float) -> tuple[np.ndarray, np.ndarray]:
    """애니메이션 프레임별 공 중심 좌표와 임팩트 순간 공 중심."""
    kick, m, imp = get_kick(kick_id), get_motion(kick_id, H), get_impact(kick_id, H)
    i = int(np.argmin(np.abs(m.t - kick["impact"]["time"])))
    cp = contact_point(frame_points(m, i), kick["impact"]["contact_point"])
    center = ball_center_at_contact(cp, kick["impact"]["contact_height"], imp["direction"])
    path = ball_path(m.t, kick["impact"]["time"], center, imp["ball_speed"],
                     kick["impact"]["launch_angle_deg"], imp["direction"],
                     kick.get("ball_start", "ground"))
    return path, center


@st.cache_resource(max_entries=16)   # 그림 객체는 복사(pickle) 없이 메모리에 그대로 둔다 — cache_data는 복원에 약 2 s
def get_animation(kick_id: str, H: float, speed: float, labels: bool, trail: bool, com: bool,
                  outline: bool = True, foot: str = "R"):
    """Plotly 애니메이션 그림. 왼발잡이도 측면 그림은 같고 라벨만 바뀐다."""
    ball, _ = get_ball(kick_id, H)
    return animated_figure(get_kick(kick_id), get_motion(kick_id, H), H,
                           {"labels": labels, "trail": trail, "com": com, "outline": outline, "foot": foot},
                           speed, ball)


@st.cache_data
def get_ball_3d(kick_id: str, H: float) -> np.ndarray:
    """3D 애니메이션용 공 중심 (n, 3): 발 방향(수평면 회전)을 반영한 접촉점 앞에 공을 두고,
    임팩트 후에는 발 방향·접촉 위치로 정해진 방향·회전으로 날아간다."""
    kick, m, imp = get_kick(kick_id), get_motion(kick_id, H), get_impact(kick_id, H)
    path, center = get_ball(kick_id, H)
    t_imp = kick["impact"]["time"]
    i = int(np.argmin(np.abs(m.t - t_imp)))
    p3 = to_3d(frame_points(m, i), H, foot_yaw_deg=foot_yaw_series(kick, m.t[i:i + 1])[0])
    cp3 = contact_point(p3, kick["impact"]["contact_point"])
    fp = foot_params(kick)
    az = np.radians(fp["aim_deg"])
    c3 = cp3 + BALL_RADIUS * np.array([np.cos(az), 0.0, np.sin(az)])
    c3[1] = center[1]
    flight = kick_flight(imp["ball_speed"], kick["impact"]["launch_angle_deg"], max(center[1] - BALL_RADIUS, 0.0),
                         fp["aim_deg"], 0.0, fp["side"], fp["up"])
    path = path + np.array([c3[0] - center[0], 0.0])        # 임팩트 순간 공이 3D 접촉점 앞에 오도록 이동
    return ball_path_3d(m.t, t_imp, path, c3[:2], float(c3[2]), flight)


@st.cache_resource(max_entries=8)
def get_animation_3d(kick_id: str, H: float, speed: float, labels: bool, outline: bool, spin: bool,
                     camera: str, foot: str = "R"):
    """3D 애니메이션 (마우스로 360° 회전). 왼발잡이는 몸 전체·공 궤적을 좌우 거울상으로 뒤집는다."""
    ball = get_ball_3d(kick_id, H)
    return animated_figure_3d(get_kick(kick_id), get_motion(kick_id, H), H,
                              {"labels": labels, "outline": outline, "foot": foot}, speed, ball, spin=spin,
                              camera=camera)


@st.cache_data
def get_content(excel_name: str) -> dict | None:
    """엑셀 설명 (없으면 None)."""
    try:
        return load_kick_content(excel_name)
    except Exception as e:  # 시트 이름이 다르거나 파일이 깨진 경우 앱은 계속 동작
        st.warning(f"엑셀 파일을 읽지 못했습니다: {e}")
        return None


def kick_concepts(content: dict | None) -> list[str]:
    """이 킥에 쓰이는 생체역학 개념 이름 (엑셀 개념별 분석 시트 기준, 괄호 앞 짧은 이름)."""
    if content is None:
        return []
    return [c.split(" (")[0] for c in content["concepts"]["개념"]]


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
kick_ids = available_kicks()
with st.sidebar:
    st.header("⚙️ 설정")
    kick_id = st.selectbox("킥 선택", kick_ids, format_func=lambda k: get_kick(k)["name_ko"])
    foot_label = st.radio("차는 발", ["오른발", "왼발"], horizontal=True,
                          help="왼발잡이는 오른발 킥의 좌우 거울상입니다. 관절 각도는 같고 좌우와 공이 휘는 방향만 바뀝니다.")
    FOOT = "L" if foot_label == "왼발" else "R"
    LEFT = FOOT == "L"
    height_cm = st.slider("키 (cm)", MIN_HEIGHT_CM, MAX_HEIGHT_CM, int(DEFAULT_HEIGHT_M * 100))
    speed = st.select_slider("재생 속도", SPEEDS, value=0.25, format_func=lambda s: f"{s}×",
                             help="실제 킥은 0.6초 정도로 매우 빨라서 느리게 재생합니다.")
    show_labels = st.checkbox("관절 각도 라벨", value=True)
    show_outline = st.checkbox("사람 윤곽(몸 형태)", value=True, help="관절·분절 선 위에 몸의 살 윤곽을 덧그립니다.")
    show_trail = st.checkbox("발끝 궤적(잔상)", value=False)
    show_com = st.checkbox("무게중심 표시", value=False)

H = height_cm / 100
kick = get_kick(kick_id)
m = get_motion(kick_id, H)
imp = get_impact(kick_id, H)
content = get_content(kick["excel_name"])

st.title(f"⚽ {kick['name_ko']}")
st.caption(f"{kick['name_en']} · 측면(시상면) 스틱 피규어 · {foot_label} 킥, 차는 방향 →")


def lr(text: str) -> str:
    """왼발잡이면 설명 속 '오른/왼'을 바꾼다 (킥 JSON 설명은 오른발 기준으로 쓰여 있음)."""
    return swap_lr(text) if LEFT else str(text)


KICK_SIDE, STANCE_SIDE = ("왼쪽", "오른쪽") if LEFT else ("오른쪽", "왼쪽")
if kick["is_example_values"]:
    st.caption("※ 각도는 교육용 근사값입니다")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🦵 동작 애니메이션", "📈 관절 각도 그래프", "⚽ 공의 궤적",
                                        "📚 생체역학 설명", "🎥 내 킥 분석"])

# ---------------------------------------------------------------------------
# 탭 1 — 동작 애니메이션
# ---------------------------------------------------------------------------
with tab1:
    col_fig, col_info = st.columns([3, 1.2])
    with col_fig:
        v1, v2, v3 = st.columns([1.4, 1, 1])
        view = v1.radio("보기", ["측면 2D", "3D (360° 회전)"], horizontal=True, label_visibility="collapsed")
        if view == "측면 2D":
            st.plotly_chart(get_animation(kick_id, H, speed, show_labels, show_trail, show_com, show_outline, FOOT),
                            width="stretch", config={"displayModeBar": False})
        else:
            cam = v2.selectbox("시작 시점", list(CAMERAS), index=1, label_visibility="collapsed")
            spin = v3.toggle("재생 중 자동 회전", value=False, help="▶ 재생하면 카메라가 몸 둘레를 한 바퀴 돕니다.")
            st.plotly_chart(get_animation_3d(kick_id, H, speed, show_labels, show_outline, spin, cam, FOOT),
                            width="stretch", config={"displayModeBar": False})
            st.caption("🖱 드래그 = 회전 · 휠 = 확대/축소 · 오른쪽 위 버튼 = 시점 전환. "
                       "※ 측면 각도 모델에 좌우 폭만 더한 입체 그림이라 고관절 회전·몸통 비틀기(수평면 동작)는 "
                       "표현되지 않습니다(오른쪽 '회전 동작' 배지 참고).")
    with col_info:
        st.markdown(f"**진한 색** = {KICK_SIDE}(차는) 다리 · **연한 색** = {STANCE_SIDE}(디딤) 다리")
        if LEFT:
            st.caption("왼발잡이는 오른발 킥의 좌우 거울상입니다. 측면 2D 그림은 오른쪽에서 본 모습이라 오른발 킥과 같게 보이고, "
                       "3D 보기에서 왼다리로 차는 모습을 볼 수 있습니다.")
        st.markdown("🔴 빨간 점 = 이 단계에서 강조하는 관절")
        if content is not None:
            st.markdown("**이 킥에 쓰이는 생체역학 개념**")
            st.markdown(" ".join(f":blue-badge[{c}]" for c in kick_concepts(content)))
        if kick["rotation_notes"]:
            st.markdown("**측면에서 안 보이는 회전 동작**")
            for note in kick["rotation_notes"]:
                st.badge(lr(note), icon="🔄", color="blue")
        lift = max(kf["kick_hip_lift"] for kf in kick["keyframes"])
        with st.expander("관상면(옆으로 기울기) 보정값이란?"):
            st.write(
                "실제 킥에서는 골반이 차는 쪽으로 들리고(골반 경사), 몸 전체가 디딤발 쪽으로 옆으로 기울어 "
                "다리가 땅에 걸리지 않습니다. 측면 2D 그림에서는 이 동작이 보이지 않으므로 "
                f"차는 쪽({KICK_SIDE}) 고관절을 위로 올려 그렸습니다 (`kick_hip_lift`, 최대 {lift:.2f} m)."
            )
        if kick["airborne"]:
            st.info("공중 동작: 디딤발 고정 없이 키프레임의 골반 높이를 그대로 사용합니다.")

    st.subheader("동작 단계별 보기")
    state_key = f"phase_{kick_id}"
    phases = [kf["phase"] for kf in kick["keyframes"]]
    cols = st.columns(len(phases))
    for c, ph in zip(cols, phases):
        if c.button(ph, key=f"btn_{kick_id}_{ph}", width="stretch",
                    type="primary" if st.session_state.get(state_key) == ph else "secondary"):
            st.session_state[state_key] = ph
            st.rerun()

    sel = st.session_state.get(state_key)
    if sel is None:
        st.caption("단계 버튼을 누르면 그 순간의 정지 화면과 설명이 나옵니다.")
    else:
        kf = next(k for k in kick["keyframes"] if k["phase"] == sel)
        i = int(np.argmin(np.abs(m.t - kf["t"])))
        ball, _ = get_ball(kick_id, H)
        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.plotly_chart(static_figure(kick, m, i, H, {"labels": True, "trail": show_trail,
                                                          "com": show_com, "outline": show_outline, "foot": FOOT},
                                          ball, height=460),
                            width="stretch", config={"displayModeBar": False}, key=f"static_{sel}")
        with c2:
            st.markdown(f"### {sel} · t = {kf['t']:.2f} s")
            st.write(lr(kf["note"]))
            if kf["highlight"]:
                st.markdown("**강조 관절:** " + ", ".join(
                    f"{joint_label(j, FOOT)} {kf[j]:.0f}°" for j in kf["highlight"]))
            angle_rows = {joint_label(k, FOOT): f"{kf[k]:.0f}°" for k in ANGLE_KEYS}
            st.dataframe(pd.DataFrame(angle_rows.items(), columns=["관절", "각도"]),
                         hide_index=True, height=240, width="stretch")
            if content is not None:
                sup = phase_supplement(content["phases"], sel)
                if sup:
                    st.caption(f"📘 {sup['설명']} — 관련 개념: {sup.get('관련 개념', '')} · "
                               f"분석 변수: {sup['분석 변수']}")
        # 주동근(근골격계)·부상 위험 패널 (엑셀 자료가 있을 때)
        muscle, injury = concept_row(content, "근골격계"), concept_row(content, "부상 위험")
        if muscle or injury:
            a, b = st.columns(2)
            if muscle:
                a.info(f"**💪 주동근 (근골격계)**\n\n{lr(muscle['분석 포인트'])}")
            if injury:
                b.warning(f"**⚠️ 부상 위험**\n\n{lr(injury['분석 포인트'])}")

# ---------------------------------------------------------------------------
# 탭 2 — 관절 각도 그래프
# ---------------------------------------------------------------------------
with tab2:
    md = get_dense_motion(kick_id, H)
    st.subheader(f"{KICK_SIDE}(차는) 다리 관절 각도")
    st.plotly_chart(angle_chart(kick, md), width="stretch")
    st.subheader("허벅지·정강이 절대 분절 각속도")
    st.plotly_chart(velocity_chart(kick, md), width="stretch")
    w_thigh, w_shank = segment_velocities(md)
    t_thigh, t_shank = md.t[np.argmax(w_thigh)], md.t[np.argmax(w_shank)]
    c1, c2, c3 = st.columns(3)
    c1.metric("허벅지 최고 각속도", f"{w_thigh.max():.0f} °/s", f"{t_thigh:.3f} s", delta_color="off")
    c2.metric("정강이 최고 각속도", f"{w_shank.max():.0f} °/s", f"{t_shank:.3f} s", delta_color="off")
    c3.metric("허벅지 → 정강이 시간차", f"{(t_shank - t_thigh) * 1000:.0f} ms")
    st.info("허벅지가 먼저 최고 속도에 도달한 뒤 감속하고, 정강이가 이어서 최고 속도에 도달합니다 "
            "(근위→원위 순차, 각운동학·각운동역학).")
    st.caption("절대 분절 각속도 = 분절이 수직선 기준으로 회전하는 속도 (ω = dθ/dt). "
               "허벅지 각도 = 고관절 굴곡 − 몸통 기울기, 정강이 각도 = 허벅지 각도 − 무릎 굴곡.")

# ---------------------------------------------------------------------------
# 탭 3 — 공의 궤적
# ---------------------------------------------------------------------------
@st.fragment
def render_ball_tab() -> None:
    """탭 3 전체. fragment라서 이 탭의 슬라이더를 움직이면 이 탭만 다시 그린다 (다른 탭 재계산·재전송 없음)."""
    ratio = kick["impact"]["ball_foot_speed_ratio"]
    st.subheader("임팩트: 발 속도 → 공 속도 (선운동역학)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("임팩트 직전 최대 발 속도", f"{imp['foot_speed']:.1f} m/s",
              f"{imp['peak_time']:.3f} s", delta_color="off")
    c2.metric("임팩트 순간 발 속도", f"{imp['foot_speed_at_impact']:.1f} m/s")
    c3.metric("공/발 속도비", f"{ratio:.2f}")
    c4.metric("계산된 공 속도", f"{imp['ball_speed']:.1f} m/s")
    st.caption(f"접촉 부위: {CONTACT_NAMES[kick['impact']['contact_point']]}. 발 속도는 접촉점 좌표를 FK로 "
               "계산해 수치미분(v = dr/dt)한 값입니다. 키프레임 보간 특성상 임팩트 키프레임에서 속도가 "
               "줄어드는 근사 오차가 있어, 임팩트 직전 30 ms 안의 최대 발 속도를 기본값으로 씁니다.")

    st.subheader("공의 포물체 운동")
    s1, s2 = st.columns(2)
    v = s1.slider("공 속도 (m/s)", 3.0, 40.0, float(np.clip(round(imp["ball_speed"], 1), 3, 40)), 0.1,
                  key=f"v_{kick_id}_{height_cm}")
    ang = s2.slider("발사각 (°)", -10.0, 80.0, float(kick["impact"]["launch_angle_deg"]), 0.5,
                    key=f"a_{kick_id}")
    _, center = get_ball(kick_id, H)
    h0 = max(float(center[1]) - BALL_RADIUS, 0.0)
    with_drag = st.checkbox("공기저항 포함(비교용)")

    T, hmax, R = flight_time(v, ang, h0), max_height(v, ang, h0), horizontal_range(v, ang, h0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("비행시간", f"{T:.2f} s")
    m2.metric("최고점", f"{hmax + BALL_RADIUS:.2f} m", help="공 중심 높이 (지면 기준)")
    m3.metric("비거리", f"{R:.1f} m")
    m4.metric("발사 높이", f"{h0 + BALL_RADIUS:.2f} m", help="임팩트 순간 공 중심 높이")
    st.plotly_chart(trajectory_chart(v, ang, h0, with_drag), width="stretch")

    if h0 > 0.005:
        st.latex(r"y(t) = h_0 + v\sin\theta\,t - \tfrac{1}{2}gt^2,\quad "
                 r"T = \frac{v\sin\theta + \sqrt{(v\sin\theta)^2 + 2gh_0}}{g}")
    else:
        st.latex(r"T = \frac{2v\sin\theta}{g}")
    st.latex(r"h_{max} = h_0 + \frac{(v\sin\theta)^2}{2g},\qquad R = v\cos\theta \cdot T,\qquad g = 9.81\ \mathrm{m/s^2}")
    if with_drag:
        st.latex(r"F_D = \tfrac{1}{2}\rho C_d A v^2 \quad (\rho = 1.2\ \mathrm{kg/m^3},\ C_d = 0.25,\ "
                 r"A = 0.038\ \mathrm{m^2},\ m = 0.43\ \mathrm{kg})")
        st.caption("공기저항 궤적은 RK4로 수치적분했습니다. 공기저항이 있으면 비거리가 줄고 궤적이 비대칭이 됩니다.")
    # --- 발 방향·접촉 위치 → 공의 방향·회전·휘는 궤적 -------------------------------------------
    st.subheader("🦶 발 방향·접촉 위치 → 공의 방향과 회전 (+ 보충)")
    fp = foot_params(kick)
    mirror = -1.0 if LEFT else 1.0              # 킥 JSON의 좌우 값은 오른발 기준 → 왼발이면 실제 좌우가 반대
    st.caption(f"{kick['name_ko']} ({foot_label}) 기본값: 발 방향 {fp['foot_yaw_deg']:+.0f}° (발끝 바깥 +), "
               f"조준 {mirror * fp['aim_deg']:+.0f}° (왼쪽 +), 접촉 위치 좌우 {mirror * fp['side'] * 100:+.1f} cm · "
               f"위아래 {fp['up'] * 100:+.1f} cm. 슬라이더로 바꿔 보세요.")
    if LEFT:
        st.caption("왼발잡이는 오른발 킥의 좌우 거울상입니다. 같은 감아차기라도 공이 **오른쪽으로** 휘어 들어오고, "
                   "아웃사이드 킥은 왼쪽으로 휩니다. 백스핀·톱스핀(위아래 회전)은 같습니다.")
    k1, k2, k3 = st.columns(3)
    d_yaw = k1.slider("발 방향 조정 Δψ (°)", -30.0, 30.0, 0.0, 1.0, key=f"yaw_{kick_id}_{FOOT}",
                      help="기본 발 방향 대비 발끝을 바깥(+)·안쪽(−)으로 더 돌림. 공에 닿는 발 면이 같이 돌아갑니다.")
    side_cm = k2.slider("공 좌우 접촉 위치 (cm, 오른쪽 +)", -6.0, 6.0, round(mirror * fp["side"] * 100, 1), 0.5,
                        key=f"side_{kick_id}_{FOOT}", help="공을 뒤에서 볼 때 중심에서 벗어난 거리. 옆을 차면 사이드스핀.")
    up_cm = k3.slider("공 위아래 접촉 위치 (cm, 위 +)", -6.0, 6.0, round(fp["up"] * 100, 1), 0.5,
                      key=f"up_{kick_id}_{FOOT}", help="아래를 차면 백스핀(뜨는 공), 위를 차면 톱스핀(떨어지는 공).")
    fl = st.cache_data(kick_flight)(v, ang, h0, fp["aim_deg"], d_yaw, side_cm / 100, up_cm / 100, left=LEFT)
    az = mirror * (fp["aim_deg"] - STEER * d_yaw)   # 실제 발사 방향 (왼쪽 +)
    ref = st.cache_data(simulate)(float(np.linalg.norm(fl.v0)), ang, az, h0)   # 같은 방향·속도, 회전 없음
    spin_lines = spin_description(fl)
    side_rps = float(np.dot(fl.omega, [0.0, 1.0, 0.0])) / (2 * np.pi)

    g1, g2 = st.columns(2)
    g1.plotly_chart(foot_topview(fp["foot_yaw_deg"] + d_yaw, kick["impact"]["contact_point"], az, side_rps, LEFT),
                    width="stretch", config={"displayModeBar": False})
    g2.plotly_chart(contact_diagram(side_cm, up_cm), width="stretch", config={"displayModeBar": False})
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("발사 방향", f"{az:+.1f}°", help="정면 0°, 왼쪽 +")
    n2.metric("공 회전", f"{fl.spin_rps:.1f} 회/s", f"{fl.spin_rps * 60:.0f} rpm", delta_color="off")
    n3.metric("착지 좌우 위치", f"{-fl.lateral_m:+.2f} m", help="오른쪽 +", delta=f"휨 {-(fl.lateral_m - ref.lateral_m):+.2f} m",
              delta_color="off")
    n4.metric("비거리 (공기저항 포함)", f"{fl.range_m:.1f} m", f"회전 없을 때 {ref.range_m:.1f} m", delta_color="off")
    for line in spin_lines:
        st.markdown(f"- {line}")
    st.plotly_chart(flight_views(fl, ref), width="stretch")
    with st.expander("어떻게 계산했나요? (공식)"):
        st.markdown("**① 발 방향 → 공 방향 (운동량 전달)** — 공은 대체로 발의 접촉면이 향하는 쪽으로 나갑니다. "
                    f"발을 Δψ만큼 더 돌리면 접촉면도 돌아가 공의 수평 방향이 약 {STEER:.1f} × Δψ 바뀝니다 (교육용 근사).")
        st.latex(r"\phi_{공} = \phi_{조준} - " + f"{STEER:.1f}" + r"\,\Delta\psi")
        st.markdown("**② 접촉 위치 → 회전 (각충격량 = 각운동량 변화)** — 충격량 J가 공 중심에서 d만큼 벗어나 "
                    "작용하면 각충격량 d × J가 공을 돌립니다. 공을 얇은 구껍질(I = ⅔mr²)로 보면")
        st.latex(r"I\omega = d\,J = d\,m\Delta v \;\Rightarrow\; \omega = \frac{3\,\Delta v\, d}{2r^2}")
        st.markdown("중심을 벗어나 차면 에너지 일부가 회전으로 가서 직진 속도는 √(1 − (d/r)²)배로 줄어듭니다.")
        st.markdown("**③ 회전 → 휘는 궤적 (보충: 마그누스 효과, 수업 범위 밖)** — 회전하는 공 주변 공기 흐름이 한쪽으로 "
                    "휘어 회전축과 진행 방향에 수직인 힘이 생깁니다. 공기저항과 함께 RK4로 수치적분했습니다.")
        st.latex(r"\vec F_M = \tfrac{1}{2}\rho C_L A v^2\,(\hat\omega \times \hat v),\qquad "
                 r"C_L \approx \frac{S}{2.022S + 0.981},\quad S = \frac{r\omega}{v}")
    st.caption("※ 발 방향·조준·접촉 위치 기본값과 계수는 교육용 근사값입니다. 회전 감쇠와 공의 변형, "
               "회전이 거의 없는 공의 불규칙한 흔들림은 계산하지 않습니다.")

    with st.expander("충격량 (J = mΔv)"):
        J = BALL_MASS * v
        st.latex(rf"J = m\Delta v = {BALL_MASS}\ \mathrm{{kg}} \times {v:.1f}\ \mathrm{{m/s}} = {J:.2f}\ \mathrm{{N\cdot s}}")
        st.latex(rf"\bar F = \frac{{J}}{{\Delta t}} = \frac{{{J:.2f}}}{{{CONTACT_TIME}}} \approx {J / CONTACT_TIME:,.0f}\ \mathrm{{N}}")
        st.caption("정지한 공 기준, 접촉시간 약 10 ms 가정. 접촉시간이 짧을수록 평균 충격력이 커집니다.")


with tab3:
    render_ball_tab()

# ---------------------------------------------------------------------------
# 탭 4 — 생체역학 설명
# ---------------------------------------------------------------------------
with tab4:
    if content is None:
        st.info("data 폴더에 엑셀 파일을 넣어주세요 (`data/축구킥_생체역학_정리.xlsx`).")
    else:
        ov = content["overview"]
        if ov:
            st.subheader(ov["킥 종류"])
            a, b, c = st.columns(3)
            a.markdown(f"**사용 발 부위**\n\n{ov['사용 발 부위']}")
            b.markdown(f"**주 사용 상황**\n\n{ov['주 사용 상황']}")
            c.markdown(f"**공의 특성**\n\n{ov['공의 특성 (속도·회전·궤적)']}")
            st.success(f"**핵심 생체역학 포인트** — {lr(ov['핵심 생체역학 포인트'])}")
        st.subheader(f"{kick['name_ko']}에 쓰이는 생체역학 개념")
        concepts = content["concepts"]
        st.caption(" · ".join(kick_concepts(content)))
        for _, row in concepts.iterrows():
            with st.expander(f"**{row['개념']}**", expanded=False):
                st.markdown(lr(row["분석 포인트"]))
                if row.get("측정 변수 / 지표"):
                    st.markdown(f"**측정 변수 / 지표** — {row['측정 변수 / 지표']}")
                if row.get("참고 수치·비고"):
                    st.caption(f"참고 수치·비고: {row['참고 수치·비고']}")
        with st.expander("표로 보기"):
            st.dataframe(concepts, hide_index=True, width="stretch")
        if content["matrix"] is not None and len(content["matrix"]):
            st.subheader("개념 한눈에 보기 (● = 이 킥에 쓰임)")
            st.dataframe(content["matrix"].fillna(""), hide_index=True, width="stretch")
        with st.expander("킥 동작 단계 (공통)"):
            st.dataframe(content["phases"], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 탭 5 — 내 킥 분석 (영상 업로드 → 분류 → 피드백)
# ---------------------------------------------------------------------------
with tab5:
    video_tab.render(height_cm, FOOT)

st.divider()
st.caption("한계: 각도는 교육용 근사값이며, 2D 측면 모델이라 고관절 회전·골반 경사 등 관상면·수평면 동작은 "
           "보정값과 배지로만 표현합니다. 공의 회전(스핀)과 휘는 궤적은 계산하지 않습니다.")
