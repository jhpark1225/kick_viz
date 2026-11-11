"""측면 + 후면 함께 분석: 짝짓기·피드백 합치기."""

from biomech.kick_loader import load_kick
from video.combined import auto_pairs, merge_feedback, pair_key
from video.events import detect_events
from video.feedback import IMPROVE, compute_metrics, make_feedback
from video.frontal import compute_frontal_metrics, rear_rules
from video.preprocess import preprocess
from video.synthetic import synthesize

H, FPS = 1.75, 120


def test_pair_key_strips_view_words():
    assert pair_key("인스텝_측면.mp4") == pair_key("인스텝_후면.MOV") == "인스텝"
    assert pair_key("감아차기 side.mp4") == pair_key("감아차기-rear.mp4") == "감아차기"
    assert pair_key("인스텝_슬로우_측면.mp4") == "인스텝_슬로우"


def test_auto_pairs_by_name():
    items = {"a": ("인스텝_측면.mp4", "side"), "b": ("로빙_후면.mp4", "rear"),
             "c": ("인스텝_후면.mp4", "rear"), "d": ("로빙_측면.mp4", "side"), "e": ("모르는킥.mp4", "side")}
    assert auto_pairs(items) == [("a", "c"), ("d", "b")]


def test_merge_feedback_tags_view_and_ranks_together():
    kick = load_kick("instep")
    ps = preprocess(synthesize(kick, H, FPS), H, FPS, view="side")
    pr = preprocess(synthesize(kick, H, FPS, view="rear"), H, FPS, view="rear")
    side = make_feedback("instep", compute_metrics(ps, detect_events(ps)))
    rear = make_feedback("instep", compute_frontal_metrics(pr, detect_events(pr)), rear_rules("instep"))
    both = merge_feedback(side, rear)
    assert len(both.items) == len(side.items) + len(rear.items)
    assert {j.rule["view"] for j in both.items} == {"측면", "후면"}
    assert "view" not in side.items[0].rule          # 원래 규칙은 바꾸지 않음
    # 한쪽 지표만 크게 벗어나게 만들면 Top 3에 그 방향 항목이 들어간다
    bad = [j for j in rear.items]
    bad[0] = type(bad[0])(rule=bad[0].rule, value=bad[0].value, status=IMPROVE, score=99.0, message="x")
    both = merge_feedback(side, type(rear)(rear.kick_id, bad))
    assert both.top3[0].rule["view"] == "후면"
