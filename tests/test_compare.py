import numpy as np
import pytest

from biomech.kick_loader import available_kicks, load_kick
from video.classify import template_processed
from video.events import detect_events
from video.feedback import load_rules
from video.preprocess import preprocess
from video.synthetic import synthesize
from viz.compare import MAX_FRAMES, angle_compare_chart, comparison_figure, overlay_skeleton

H = 1.75
FPS = 120


@pytest.fixture(scope="module", params=available_kicks())
def analysed(request):
    kick = load_kick(request.param)
    p = preprocess(synthesize(kick, H, FPS), H, FPS)
    return kick, p, detect_events(p)


@pytest.mark.parametrize("overlay", [False, True])
def test_comparison_figure(analysed, overlay):
    kick, p, ev = analysed
    fig = comparison_figure(p, ev, kick, overlay=overlay)
    assert 2 <= len(fig.frames) <= MAX_FRAMES
    assert len(fig.frames[0].data) == len(fig.data)


def test_overlay_aligns_stance_foot():
    """겹쳐 보기: 템플릿 자신을 겹치면 임팩트 프레임에서 디딤발 위치가 일치한다."""
    kick = load_kick("instep")
    p = preprocess(synthesize(kick, H, FPS), H, FPS)
    fig = comparison_figure(p, detect_events(p), kick, overlay=True)
    i0 = int(np.argmin([abs(float(s.label)) for s in fig.layout.sliders[0].steps]))
    data = fig.frames[i0].data
    n = len(data) // 2
    leg = lambda trs: next(t for t in trs if t.x is not None and len(t.x) == 6)   # 디딤 다리 선 (l_hip … l_ankle)
    user_leg, tpl_leg = leg(data[:n]), leg(data[n:])
    mid = lambda tr: (np.array([tr.x[3], tr.y[3]]) + np.array([tr.x[4], tr.y[4]])) / 2
    assert np.linalg.norm(mid(user_leg) - mid(tpl_leg)) < 0.02


def test_angle_chart(analysed):
    kick, p, ev = analysed
    tpl = template_processed(kick["id"], H)
    t_imp = kick["impact"]["time"] - kick["keyframes"][0]["t"]
    fig = angle_compare_chart(p, ev, "r_knee", tpl, t_imp, kick["name_ko"], load_rules()[kick["id"]])
    assert len(fig.data) == 2


def test_overlay_skeleton_draws_on_thumbnail():
    cv2 = pytest.importorskip("cv2")
    kick = load_kick("instep")
    raw = synthesize(kick, H, FPS)
    p = preprocess(raw, H, FPS)
    thumb = np.full((203, 360, 3), 255, np.uint8)
    jpeg = cv2.imencode(".jpg", thumb)[1].tobytes()
    img = overlay_skeleton(jpeg, p.xy_px, 10, 360 / raw.width, ["r_knee"])
    assert img.shape == (203, 360, 3)
    assert (img < 200).any()        # 선이 그려짐
