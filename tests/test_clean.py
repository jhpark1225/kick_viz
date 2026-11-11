"""원시 랜드마크 정리(video/clean.py)와 실제 영상형 잡음에 대한 단계 검출 강건성 테스트.

실제 킥모음 영상에서 관찰된 세 가지 오류를 합성 데이터에 넣어 확인한다:
좌우 다리 이름 뒤바뀜(몇 프레임), 1~2프레임 스파이크, 차는 발이 디딤발을 지날 때 디딤발 점이 끌려가는 현상.
"""

import numpy as np
import pytest

from biomech.kick_loader import PHASES, available_kicks, load_kick
from video.clean import SWAP_GROUPS, clean_pose, fix_swaps, remove_spikes, torso_length
from video.events import detect_events
from video.preprocess import preprocess
from video.synthetic import synthesize

H = 1.75
FPS = 120
LEG = SWAP_GROUPS["다리"]


def _swap(raw, frames: slice, parts=LEG):
    """frames 구간의 좌우 다리 이름을 일부러 바꾼다 (MediaPipe 오류 흉내)."""
    for m in parts:
        l, r = raw.xy[f"l_{m}"][frames].copy(), raw.xy[f"r_{m}"][frames].copy()
        raw.xy[f"l_{m}"][frames], raw.xy[f"r_{m}"][frames] = r, l
    return raw


def _leg_err(a, b) -> float:
    return max(float(np.max(np.abs(a.xy[k] - b.xy[k]))) for k in a.xy)


@pytest.mark.parametrize("kick_id", available_kicks())
def test_clean_data_untouched(kick_id):
    """오류가 없는 데이터(잡음 포함)는 좌우를 바꾸지 않고 스파이크도 거의 찾지 않는다."""
    raw = synthesize(load_kick(kick_id), H, FPS, noise_frac=0.003)
    out, rep = clean_pose(raw)
    assert rep.total_swapped == 0
    assert rep.spikes <= 2
    assert _leg_err(out, raw) == 0.0


@pytest.mark.parametrize("kick_id", ["instep", "inside", "toe", "volley"])
def test_leg_swap_corrected(kick_id):
    """접근 구간에서 8프레임 동안 다리 이름이 바뀌어도 원래대로 되돌린다."""
    truth = synthesize(load_kick(kick_id), H, FPS)
    bad = _swap(synthesize(load_kick(kick_id), H, FPS), slice(10, 18))
    fixed, counts = fix_swaps(bad)
    assert counts["다리"] == 8
    assert _leg_err(fixed, truth) < 1e-9


def test_leg_swap_at_crossing():
    """두 다리가 교차하는 순간 바로 뒤에서 바뀐 경우도 바로잡는다 (등속 예측 덕분)."""
    kick = load_kick("instep")
    truth = synthesize(kick, H, FPS)
    d = np.abs(truth.xy["l_ankle"][:, 0] - truth.xy["r_ankle"][:, 0])
    i = int(np.argmin(d[5:len(d) // 2])) + 5            # 접근~스윙 중 두 발목이 가장 가까운 프레임
    bad = _swap(synthesize(kick, H, FPS), slice(i + 1, i + 6))
    fixed, _ = fix_swaps(bad)
    assert _leg_err(fixed, truth) < 1e-9


def test_spike_removed_and_interpolated():
    """1프레임·2프레임 스파이크는 결측 처리되어 보간되고, 필터 후 각도 오차가 크게 줄어든다."""
    kick = load_kick("instep")
    clean = preprocess(synthesize(kick, H, FPS), H, FPS)
    raw = synthesize(kick, H, FPS)
    tl = torso_length(raw.xy)
    raw.xy["l_knee"][20] += np.array([0.6 * tl, -0.3 * tl])
    raw.xy["l_ankle"][40:42] += np.array([-0.5 * tl, 0.0])
    _, n = remove_spikes(raw)
    assert n == 3
    with_clean = preprocess(raw, H, FPS)
    no_clean = preprocess(raw, H, FPS, clean=False)
    err = lambda p: float(np.max(np.abs(p.angles["l_knee"] - clean.angles["l_knee"])))
    assert err(with_clean) < 0.3 * err(no_clean)
    assert err(with_clean) < 3.0


def test_fast_kick_not_spike():
    """임팩트 근처의 빠른 발 이동(30 fps에서 프레임당 수십 cm)은 스파이크로 지우지 않는다."""
    for kick_id in available_kicks():
        raw = synthesize(load_kick(kick_id), H, 30.0)
        _, n = remove_spikes(raw)
        assert n == 0, kick_id


def test_first_frame_glitch_ignored():
    """영상 첫 프레임만 튄 경우(실제 로빙 영상): 차는 다리·임팩트가 첫 프레임에 잡히지 않는다."""
    kick = load_kick("instep")
    raw = synthesize(kick, H, FPS)
    tl = torso_length(raw.xy)
    for m in ("toe", "heel", "ankle"):
        raw.xy[f"l_{m}"][0] += np.array([-1.5 * tl, 0.0])
    p = preprocess(raw, H, FPS)
    ev = detect_events(p)
    t_true = kick["impact"]["time"] - kick["keyframes"][0]["t"]
    assert p.kick_side == "R"
    assert abs(ev.time("임팩트") - t_true) * FPS <= 1 + 1e-9


def _drag_stance_foot(raw, p_base, H):
    """두 발이 OCCLUDE_RATIO × 키 안으로 겹친 프레임에서 디딤발 점을 차는 발 쪽으로 40% 당긴다 (실제 영상 현상)."""
    from video.events import OCCLUDE_RATIO
    d = np.linalg.norm(p_base.xy["r_ankle"] - p_base.xy["l_ankle"], axis=1)
    near = d < OCCLUDE_RATIO * H
    for m in ("ankle", "heel", "toe"):
        raw.xy[f"l_{m}"][near] += 0.4 * (raw.xy["r_ankle"][near] - raw.xy[f"l_{m}"][near])
    return raw, int(near.sum())


@pytest.mark.parametrize("kick_id", ["instep", "inside", "curve", "toe"])
def test_plant_robust_to_occlusion_drag(kick_id, monkeypatch):
    """차는 발이 디딤발 옆을 지날 때 디딤발 점이 끌려가도(실제 영상) 디딤을 검출한다.

    킥모음 영상처럼 스윙이 느린 경우(time_scale=2)는 두 발이 겹친 시간이 임팩트 직전 허용 구간(30 ms)보다 길다.
    """
    kick = load_kick(kick_id)
    p_base = preprocess(synthesize(kick, H, FPS, time_scale=2.0), H, FPS)
    base = detect_events(p_base)
    raw, n_near = _drag_stance_foot(synthesize(kick, H, FPS, time_scale=2.0), p_base, H)
    assert n_near > 0.03 * FPS
    ev = detect_events(preprocess(raw, H, FPS))
    assert ev.frames["디딤"] is not None
    assert abs(ev.frames["디딤"] - base.frames["디딤"]) <= int(0.03 * FPS)
    # 겹침 처리를 끄면 검출에 실패한다 → 이 테스트가 실제로 그 처리를 확인하고 있음
    monkeypatch.setattr("video.events.OCCLUDE_RATIO", 0.0)
    assert detect_events(preprocess(raw, H, FPS)).frames["디딤"] is None


def test_events_with_realistic_noise():
    """키의 0.8% 잡음 + 다리 뒤바뀜 + 스파이크가 섞여도 6개 단계를 모두 검출한다."""
    for kick_id in ("instep", "inside", "chip"):
        kick = load_kick(kick_id)
        raw = synthesize(kick, H, FPS, noise_frac=0.008, seed=1)
        raw = _swap(raw, slice(6, 12))
        tl = torso_length(raw.xy)
        raw.xy["r_knee"][15] += np.array([0.5 * tl, 0.0])
        ev = detect_events(preprocess(raw, H, FPS))
        assert not ev.failed, (kick_id, ev.failed)
        assert all(ev.frames[a] <= ev.frames[b] for a, b in zip(PHASES, PHASES[1:])), kick_id
