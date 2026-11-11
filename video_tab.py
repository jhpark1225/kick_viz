"""탭 5 — 🎥 내 킥 분석: 영상 업로드 → 킥 자동 분류 → 수정점 피드백 → 비교 시각화."""

import numpy as np
import pandas as pd
import streamlit as st

from biomech.kick_loader import PHASES, available_kicks, load_kick
from video.classify import GROUP_NOTE, GROUPS, classify, template_processed
from video.events import detect_events
from video.feedback import (IMPROVE, SAFETY_NOTE, SIDE_VIEW_LIMITS, SIDE_VIEW_NOTE, UNSURE, compute_metrics,
                            format_range, format_value, load_rules, make_feedback, split_message)
from video.pose_extract import (MAX_SECONDS, THUMB_WIDTH, VideoError, extract_from_bytes, find_model,
                                info_from_bytes, model_help)
from video.preprocess import CUTOFF_RANGE_HZ, DEFAULT_CUTOFF_HZ, LOW_FPS, preprocess
from video.synthetic import synthesize
from video.frontal import (REAR_LIMIT_NOTE, REAR_NOT_MEASURED, REAR_NOT_MEASURED_NOTE, compute_frontal_metrics,
                           rear_rules)
from viz.compare import (REAR_ANGLE_NAMES, angle_compare_chart, angle_label, comparison_figure, overlay_skeleton,
                         phase_highlights, rear_angle_chart, rear_comparison_figure)

FPS_HELP = ("스마트폰 슬로모션 영상은 재생 fps(30)로 저장되는 경우가 많습니다. "
            "120/240fps로 찍었다면 실제 촬영 fps를 입력하세요.")
GUIDE = """
**측면 촬영** (무릎·발목 각도, 근위→원위 순차, 킥 종류 자동 분류)
- 선수 옆 **90°**에서, 차는 방향이 화면 좌우가 되게

**후면 촬영** (디딤 무릎 외반, 골반·몸통 옆 기울기, 디딤발 위치, 팔 벌림)
- 선수 **바로 뒤**에서, 차는 방향을 정면으로 바라보게 (카메라 높이는 허리 정도)

**공통**
- **고정 카메라** (삼각대 사용, 손으로 들고 따라가지 않기)
- **전신 + 공**이 모두 화면 안에 보이게, **밝은 곳**에서
- 가능하면 **120fps 이상 슬로모션**
- **한 영상에 킥 1회**만, 최대 {sec:.0f}초
- 두 방향을 모두 찍으면 서로 못 보는 부분을 채울 수 있습니다 (함께 올리면 각각 분석하고, 결과를 골라서 봅니다)
"""
VIEW_OPTIONS = {"자동 감지": "auto", "측면": "side", "후면": "rear"}
VIEW_NAMES = {"side": "측면 (시상면)", "rear": "후면 (관상면)"}
REAR_ANGLES = ["stance_knee_valgus", "pelvic_obliquity", "trunk_lateral_lean", "arm_abduction"]
COMPARE_ANGLES = ["r_knee", "r_hip", "r_ankle", "l_knee", "trunk", "l_shoulder"]
STATE = "video_analysis"          # 지금 보고 있는 분석 결과의 id
ANALYSES = "video_analyses"       # id → 분석 결과 (여러 영상)
MAX_FILES = 4                     # 무료 서버 메모리(약 1 GB)를 생각해 한 번에 올리는 영상 수 제한


def _fig_key(p, ev, *extra) -> tuple:
    """비교 그림 캐시 키: 같은 분석 결과(fps·필터·촬영 방향·단계 프레임)면 그림을 다시 만들지 않는다."""
    return (p.n, p.fps, p.cutoff_hz, p.view, p.H, float(p.xy["r_toe"][:, 0].sum()),
            tuple(sorted((k, v) for k, v in ev.frames.items())), *extra)


@st.cache_resource(max_entries=8)
def _comparison(key: tuple, _p, _ev, _kick, overlay: bool):
    return comparison_figure(_p, _ev, _kick, overlay=overlay)


@st.cache_resource(max_entries=8)
def _rear_comparison(key: tuple, _p, _ev, _kick, highlight: tuple):
    return rear_comparison_figure(_p, _ev, _kick, list(highlight))


def _kick_name(k: str) -> str:
    return load_kick(k)["name_ko"]


def _input_section(default_height_cm: int, foot: str = "R") -> None:
    """업로드(또는 예시 데이터) + 키 + [분석 시작]."""
    mode = st.radio("입력", ["내 영상 업로드", "예시 데이터로 체험 (템플릿 합성)"], horizontal=True,
                    label_visibility="collapsed")
    c1, c2 = st.columns([2, 1])
    height_cm = c2.number_input("내 키 (cm)", 120, 220, default_height_cm, 1,
                                help="픽셀 → m 변환에 씁니다 (대퇴+하퇴 길이 = 0.491 × 키).")
    model = find_model()

    if mode == "내 영상 업로드":
        if model is None:
            st.warning(model_help())
            return
        ups = c1.file_uploader(f"킥 영상 (mp4, mov, avi · 최대 30초, 200 MB · 한 번에 {MAX_FILES}개까지)",
                               type=["mp4", "mov", "avi"], accept_multiple_files=True,
                               help="측면·후면 영상이나 여러 번 찬 영상을 함께 올리면 각각 분석합니다.")
        if not ups:
            return
        if len(ups) > MAX_FILES:
            st.warning(f"영상은 한 번에 {MAX_FILES}개까지 분석합니다. 앞의 {MAX_FILES}개만 사용합니다.")
            ups = ups[:MAX_FILES]
        infos = st.session_state.setdefault("video_infos", {})
        ready = []
        for up in ups:
            if up.file_id not in infos:
                try:
                    infos[up.file_id] = info_from_bytes(up.getvalue(), "." + up.name.rsplit(".", 1)[-1].lower())
                except VideoError as e:
                    st.error(f"**{up.name}**: {e}")
                    continue
            info = infos[up.file_id]
            if info["duration"] > MAX_SECONDS + 0.5:
                st.error(f"**{up.name}**: 영상 길이가 {info['duration']:.1f} s입니다. "
                         f"{MAX_SECONDS:.0f} s 이하로 잘라서 올려주세요.")
                continue
            st.caption(f"**{up.name}** — {info['width']}×{info['height']}, 메타데이터 {info['fps']:.0f} fps, "
                       f"{info['frames']}프레임, {info['duration']:.1f} s")
            ready.append((up, info))
        if not ready:
            return
        if st.button(f"▶ 분석 시작 ({len(ready)}개)" if len(ready) > 1 else "▶ 분석 시작", type="primary"):
            analyses = st.session_state.setdefault(ANALYSES, {})
            first = None
            for n, (up, info) in enumerate(ready, 1):
                label = f"[{n}/{len(ready)}] {up.name} — 관절 좌표를 추출하는 중…"
                bar = st.progress(0.0, text=label)
                try:
                    raw, thumbs = extract_from_bytes(up.getvalue(), up.name, model,
                                                     lambda x, label=label: bar.progress(x, text=f"{label} {x:.0%}"))
                except VideoError as e:
                    bar.empty()
                    st.error(f"**{up.name}**: {e}")
                    continue
                bar.progress(1.0, text=f"[{n}/{len(ready)}] {up.name} — 추출 완료, 임시 영상 파일을 삭제했습니다.")
                analyses[up.file_id] = {"id": up.file_id, "raw": raw, "thumbs": thumbs, "name": up.name,
                                        "height_cm": height_cm, "meta_fps": info["fps"], "demo": None}
                first = first or up.file_id
            if first:
                st.session_state[STATE] = first
    else:
        kicks = available_kicks()
        d1, d2, d3, d4 = c1.columns(4)
        demo_kick = d1.selectbox("예시 킥", kicks, format_func=_kick_name)
        demo_view = d2.selectbox("촬영 방향", ["side", "rear"], format_func=lambda v: VIEW_NAMES[v])
        demo_fps = d3.selectbox("촬영 fps", [30, 60, 120, 240], index=2)
        noise = d4.checkbox("측정 노이즈", value=True, help="σ = 키의 0.5% 가우시안 노이즈")
        st.caption("기준 템플릿으로 가짜 관절 좌표를 만들어 전체 분석 과정을 체험합니다 (실제 영상 아님). "
                   "후면 예시는 3D 모델을 뒤에서 본 좌표입니다.")
        if st.button("▶ 분석 시작", type="primary"):
            raw = synthesize(load_kick(demo_kick), height_cm / 100, demo_fps,
                             noise_frac=0.005 if noise else 0.0, view=demo_view,
                             left_footed=foot == "L", seed=1)
            sid = f"demo_{demo_kick}_{demo_view}_{demo_fps}_{noise}_{height_cm}_{foot}"
            st.session_state.setdefault(ANALYSES, {})[sid] = {
                "id": sid, "raw": raw, "thumbs": None,
                "name": f"예시: {_kick_name(demo_kick)} · {VIEW_NAMES[demo_view]} ({'왼발' if foot == 'L' else '오른발'})",
                "height_cm": height_cm, "meta_fps": float(demo_fps), "demo": demo_kick}
            st.session_state[STATE] = sid


def _event_section(p, sid: str):
    """단계 검출 결과 + 수동 지정."""
    key = f"ev_override_{sid}"
    overrides = st.session_state.get(key, {})
    ev = detect_events(p, overrides)
    rows = []
    for ph in PHASES:
        f = ev.frames[ph]
        state = "검출 실패" if f is None else ("직접 지정" if ph in ev.manual else "자동")
        rows.append({"단계": ph, "프레임": "–" if f is None else f,
                     "시각 (s)": "–" if f is None else f"{p.t[f]:.3f}", "상태": state})
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with c2:
        st.metric("공중 동작", "예" if ev.airborne else "아니오")
        if ev.contact_height is not None:
            st.metric("임팩트 접촉 높이 (발목·발끝 중점)", f"{ev.contact_height:.2f} m")
    failed = [ph for ph in ev.failed if not (ev.airborne and ph == "디딤")]
    if ev.airborne and "디딤" in ev.failed:
        st.info("공중 동작이라 디딤발이 지면에 고정되는 '디딤' 단계가 없습니다.")
    if failed:
        st.warning("검출하지 못한 단계가 있습니다: " + ", ".join(failed) + ". 아래에서 직접 지정할 수 있습니다.")
    with st.expander("🛠 단계 직접 지정 (프레임 슬라이더)", expanded=bool(failed)):
        with st.form(f"ev_form_{sid}"):
            vals = {}
            for ph in PHASES:
                cur = ev.frames[ph] if ev.frames[ph] is not None else 0
                vals[ph] = st.slider(ph, 0, p.n - 1, int(cur), format="%d 프레임")
            a, b = st.columns(2)
            if a.form_submit_button("적용", type="primary"):
                new = {ph: v for ph, v in vals.items() if ev.frames[ph] is None and v != 0
                       or ev.frames[ph] is not None and v != ev.frames[ph] or ph in overrides}
                st.session_state[key] = new
                st.rerun()
            if b.form_submit_button("자동 검출로 되돌리기"):
                st.session_state[key] = {}
                st.rerun()
    return ev


def _classification_section(p, ev, sid: str) -> str:
    """분류 결과 + 수동 선택. 최종 선택된 킥 id를 반환."""
    c = classify(p, ev)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown(f"**상위 3개 후보** · 규칙 게이트: {c.gate}")
        for k, conf in c.top3:
            st.progress(float(conf), text=f"{_kick_name(k)} — {conf:.0%}")
    with col2:
        st.markdown("**판단 근거**")
        for e in c.evidence:
            st.markdown(f"- {e}")
    if c.group:
        st.info(f"**{c.group}**로 보입니다 (상위 후보 신뢰도 차이 < 15%p). {GROUP_NOTE}")
    elif c.uncertain:
        st.warning("신뢰도가 낮아 하나의 킥으로 단정하지 않습니다. 아래에서 직접 선택하세요.")
    elif any(c.best in m for m in GROUPS.values()):
        st.caption(f"참고: {GROUP_NOTE}")
    kicks = available_kicks()
    sel = st.selectbox("분석할 킥 종류 (직접 고칠 수 있습니다)", kicks, index=kicks.index(c.best),
                       format_func=_kick_name, key=f"kick_sel_{sid}")
    return sel


def _feedback_section(fb, view: str = "side") -> pd.DataFrame:
    """Top 3 카드 → 잘한 점 → 전체 표 → 이 촬영 방향으로 판단 불가 항목. 전체 표 DataFrame을 반환."""
    if view == "rear":
        st.caption("※ 기준값은 교육용 근사값입니다 (후면 공통 기준, `data/feedback_rules_rear.json`).")
    else:
        st.caption("※ 기준값은 교육용 근사값입니다 (템플릿 ± 허용폭, 인스텝은 선행연구 범위 우선).")
    st.markdown("#### 🔧 우선 개선점 Top 3")
    if not fb.items:
        st.info("이 킥은 후면 영상 기준이 없습니다 (공중 동작은 몸이 뒤집혀 관상면 분석이 의미가 없음). "
                "측면 영상으로 분석해 주세요.")
        return pd.DataFrame()
    if not fb.top3:
        st.success("기준 범위를 벗어난 항목이 없습니다. 👍")
    cols = st.columns(max(len(fb.top3), 1))
    for col, (i, j) in zip(cols, enumerate(fb.top3, 1)):
        what, why = split_message(j.message)
        with col.container(border=True):
            st.markdown(f"**{i}. {j.name}**  \n<small>{j.rule['phase']}</small>", unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric("내 값", format_value(j.value, j.unit))
            b.metric("기준 범위", format_range(j.rule))
            st.markdown(f"**무엇을 고칠지** — {what}")
            if why:
                st.markdown(f"**왜 (생체역학)** — {why}")
            st.caption(f"📘 관련 개념: {j.rule['concept']}")
    if fb.strengths:
        st.markdown("#### 👍 잘한 점")
        for j in fb.strengths:
            st.success(f"**{j.name}**: {format_value(j.value, j.unit)} (기준 {format_range(j.rule)}) — {j.rule['concept']}")
    st.markdown("#### 📋 전체 항목")
    table = pd.DataFrame([{
        "단계": j.rule["phase"], "지표": j.name, "내 값": format_value(j.value, j.unit),
        "기준 범위": format_range(j.rule), "판정": j.status, "관련 개념": j.rule["concept"],
        "비고": j.reason if j.status == UNSURE else (j.message if j.status == IMPROVE else ""),
    } for j in fb.items])
    st.dataframe(table, hide_index=True, width="stretch")
    if view == "rear":
        st.markdown("#### 📐 후면 영상으로 판단할 수 없는 항목")
        st.info("\n".join(f"- {x}" for x in REAR_NOT_MEASURED) + f"\n\n{REAR_NOT_MEASURED_NOTE}")
    else:
        st.markdown("#### 📐 측면 영상으로 판단할 수 없는 항목")
        st.info("\n".join(f"- {x}" for x in SIDE_VIEW_LIMITS) + f"\n\n{SIDE_VIEW_NOTE} "
                "(무릎 외반·골반·디딤발 좌우 거리는 **후면 영상**으로 분석할 수 있습니다)")
    st.caption(SAFETY_NOTE)
    return table


def _rear_visual_section(p, ev, kick_id: str, data: dict, fb) -> None:
    """후면: 썸네일 스켈레톤 + 뒤에서 본 나란히 애니메이션 + 관상면 각도 그래프."""
    kick = load_kick(kick_id)
    thumbs = data.get("thumbs")
    # 개선이 필요한 지표와 관련된 관절을 빨간색으로
    hl_map = {"stance_knee_valgus": "l_knee", "pelvic_obliquity": "r_hip", "trunk_lateral_lean": "r_shoulder",
              "plant_foot_distance": "l_ankle", "opposite_arm_abduction": "l_wrist"}
    hl = [hl_map[j.rule["metric"]] for j in fb.top3 if j.rule["metric"] in hl_map] or ["l_knee"]
    if thumbs:
        st.markdown("**영상 위 스켈레톤 (주요 이벤트 프레임)** · 🔴 = 개선이 필요한 관절")
        scale = THUMB_WIDTH / data["raw"].width
        hl_keys = [{"l_knee": "l_knee", "r_hip": "r_hip", "r_shoulder": "r_shoulder", "l_ankle": "l_ankle",
                    "l_wrist": "l_elbow"}[h] for h in hl]
        shown = [(ph, ev.frames[ph]) for ph in PHASES if ev.frames[ph] is not None]
        cols = st.columns(3)
        for n, (ph, f) in enumerate(shown):
            img = overlay_skeleton(thumbs[f], p.xy_px, f, scale, hl_keys)
            cols[n % 3].image(img, caption=f"{ph} · {p.t[f]:.3f} s", width="stretch")
    st.plotly_chart(_rear_comparison(_fig_key(p, ev, kick_id, tuple(hl)), p, ev, kick, tuple(hl)), width="stretch",
                    config={"displayModeBar": False},
                    key="rear_cmp")
    st.caption("왼쪽 = 내 동작(뒤에서 본 모습), 오른쪽 = 기준 템플릿을 3D 모델로 만들어 뒤에서 본 모습. "
               "템플릿은 측면 각도 모델이라 무릎 외반·몸통 옆 기울기가 0°이고, 골반 기울기는 '관상면 보정값'"
               "(kick_hip_lift)에서 나옵니다.")
    st.markdown("**관상면 각도** · 초록 띠 = 기준 범위")
    keys = st.multiselect("각도", REAR_ANGLES, default=["stance_knee_valgus", "pelvic_obliquity"],
                          format_func=lambda k: REAR_ANGLE_NAMES[k], key="rear_angles")
    tpl = None if kick["airborne"] else template_processed(kick_id, p.H, p.cutoff_hz or DEFAULT_CUTOFF_HZ, "rear")
    t_imp = kick["impact"]["time"] - kick["keyframes"][0]["t"]
    for k in keys:
        st.plotly_chart(rear_angle_chart(p, ev, k, tpl, t_imp, kick["name_ko"], rear_rules(kick_id)),
                        width="stretch", key=f"rear_ang_{k}")


def _visual_section(p, ev, kick_id: str, data: dict) -> None:
    kick = load_kick(kick_id)
    thumbs = data.get("thumbs")
    if thumbs:
        st.markdown("**영상 위 스켈레톤 (주요 이벤트 프레임)** · 🔴 = 이 단계에서 강조하는 관절")
        hl = phase_highlights(kick)
        scale = THUMB_WIDTH / data["raw"].width
        shown = [(ph, ev.frames[ph]) for ph in PHASES if ev.frames[ph] is not None]
        cols = st.columns(3)
        for n, (ph, f) in enumerate(shown):
            img = overlay_skeleton(thumbs[f], p.xy_px, f, scale, hl.get(ph, []))
            cols[n % 3].image(img, caption=f"{ph} · {p.t[f]:.3f} s", width="stretch")
    mode = st.radio("비교 보기", ["나란히", "겹쳐 보기 (디딤발 기준)"], horizontal=True)
    overlay = mode != "나란히"
    st.plotly_chart(_comparison(_fig_key(p, ev, kick_id, overlay), p, ev, kick, overlay), width="stretch",
                    config={"displayModeBar": False}, key=f"cmp_{overlay}")
    if overlay:
        st.caption("실선 = 내 동작, 반투명 점선 = 기준 템플릿. 임팩트 시각과 디딤발 위치를 맞췄습니다.")
    else:
        st.caption("왼쪽 = 내 동작(영상에서 추출), 오른쪽 = 기준 템플릿. 임팩트 시각을 맞춰 동기 재생합니다 (0.25× 속도).")

    st.markdown("**관절 각도 비교** · 초록 띠 = 기준 범위")
    keys = st.multiselect("각도", COMPARE_ANGLES, default=["r_knee", "r_hip", "r_ankle"], format_func=angle_label)
    tpl = template_processed(kick_id, p.H, p.cutoff_hz or DEFAULT_CUTOFF_HZ)
    t_imp = kick["impact"]["time"] - kick["keyframes"][0]["t"]
    rules = load_rules()[kick_id]
    for k in keys:
        st.plotly_chart(angle_compare_chart(p, ev, k, tpl, t_imp, kick["name_ko"], rules), width="stretch",
                        key=f"ang_{k}")


def render(default_height_cm: int, foot: str = "R") -> None:
    """탭 5 전체."""
    st.subheader("🎥 내 킥 분석")
    st.caption("🔒 영상은 분석 후 저장되지 않습니다. 분석을 위해 이 앱의 서버에 임시 파일로 잠시 올라가며, "
               "관절 좌표 추출 직후 삭제합니다. 다른 곳으로 보내지 않습니다. "
               "추출한 관절 좌표와 작은 썸네일은 이 브라우저 세션 메모리에만 있고, 창을 닫으면 사라집니다.")
    with st.expander("📹 촬영 가이드", expanded=False):
        st.markdown(GUIDE.format(sec=MAX_SECONDS))

    _input_section(default_height_cm, foot)
    analyses = st.session_state.get(ANALYSES, {})
    if not analyses:
        return
    ids = list(analyses)
    if st.session_state.get(STATE) not in analyses:
        st.session_state[STATE] = ids[-1]
    if len(ids) > 1:
        st.radio("분석 결과 선택", ids, key=STATE, format_func=lambda i: analyses[i]["name"], horizontal=True,
                 help="올린 영상마다 따로 분석했습니다. 골라서 결과를 보세요.")
    data = analyses[st.session_state[STATE]]
    sid = data["id"]

    with st.sidebar:
        st.divider()
        st.subheader("🎥 영상 분석 설정")
        fps = st.number_input("실제 촬영 fps", 10.0, 1000.0, float(data["meta_fps"]), 1.0, key=f"fps_{sid}",
                              help=FPS_HELP)
        st.caption(FPS_HELP)
        cutoff = st.slider("저역통과 필터 차단 주파수 (Hz)", *CUTOFF_RANGE_HZ, DEFAULT_CUTOFF_HZ, 0.5,
                           help="4차 Butterworth, 양방향(filtfilt). 생체역학 동작 데이터의 표준 처리입니다.")
        view_label = st.radio("촬영 방향", list(VIEW_OPTIONS), key=f"view_{sid}",
                              help="자동 감지: 양 어깨가 벌어져 보이면 후면, 겹쳐 보이면 측면으로 판단합니다.")
        if st.button("분석 결과 지우기" if len(ids) == 1 else f"분석 결과 모두 지우기 ({len(ids)}개)"):
            st.session_state.pop(STATE, None)
            st.session_state.pop(ANALYSES, None)
            st.rerun()

    st.divider()
    st.markdown(f"### 분석 대상: {data['name']}")
    H = data["height_cm"] / 100
    view_req = VIEW_OPTIONS[view_label]
    try:
        p = preprocess(data["raw"], H, fps, cutoff, view=view_req)
    except ValueError as e:
        st.error(f"관절 좌표 전처리에 실패했습니다: {e}")
        return
    st.caption(f"촬영 방향: **{VIEW_NAMES[p.view]}**" + (" (자동 감지 — 틀렸다면 사이드바에서 바꾸세요)"
                                                       if view_req == "auto" else ""))

    # 3) 품질 경고
    for level, msg in p.warnings:
        (st.warning if level == "warning" else st.info)(msg)
    if not p.warnings:
        st.success(f"품질 검사 통과 · {fps:.0f} fps · 필터 {p.cutoff_hz:.1f} Hz")

    st.markdown("### ① 동작 단계 검출")
    ev = _event_section(p, f"{sid}_{p.view}")
    if ev.impact is None:
        st.error("임팩트를 찾지 못해 분류·피드백을 진행할 수 없습니다. 위에서 임팩트 프레임을 직접 지정하세요.")
        return

    if p.view == "rear":
        st.markdown("### ② 킥 종류 선택")
        st.info(REAR_LIMIT_NOTE)
        kicks = available_kicks()
        default = data.get("demo") if data.get("demo") in kicks else "instep"
        kick_id = st.selectbox("분석할 킥 종류", kicks, index=kicks.index(default), format_func=_kick_name,
                               key=f"kick_sel_rear_{sid}")
        st.markdown(f"### ③ 피드백 — 후면(관상면) · {_kick_name(kick_id)}")
        fb = make_feedback(kick_id, compute_frontal_metrics(p, ev), rear_rules(kick_id))
        table = _feedback_section(fb, "rear")
        st.markdown("### ④ 내 동작 vs 기준 동작 (뒤에서 본 모습)")
        _rear_visual_section(p, ev, kick_id, data, fb)
    else:
        st.markdown("### ② 킥 분류")
        kick_id = _classification_section(p, ev, sid)

        st.markdown(f"### ③ 피드백 — {_kick_name(kick_id)} 기준")
        if fps < LOW_FPS:
            st.warning("촬영 fps가 60 미만이라 각속도·속도 지표는 '측정 불확실'로 표시합니다.")
        fb = make_feedback(kick_id, compute_metrics(p, ev))
        table = _feedback_section(fb)

        st.markdown("### ④ 내 동작 vs 기준 동작")
        _visual_section(p, ev, kick_id, data)

    st.markdown("### ⑤ 결과 다운로드")
    t0 = p.t[ev.impact]
    series = pd.DataFrame({"t (s)": p.t, "임팩트 기준 t (s)": p.t - t0,
                           "단계": [_phase_of(ev, i) for i in range(p.n)],
                           **{f"{k} (°)": np.round(v, 2) for k, v in p.angles.items()}})
    a, b = st.columns(2)
    a.download_button("피드백 표 (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                      file_name="kick_feedback.csv", mime="text/csv")
    b.download_button("각도 시계열 (CSV)", series.to_csv(index=False).encode("utf-8-sig"),
                      file_name="kick_angles.csv", mime="text/csv")


def _phase_of(ev, i: int) -> str:
    name = ""
    for ph in PHASES:
        f = ev.frames.get(ph)
        if f is not None and f <= i:
            name = ph
    return name
