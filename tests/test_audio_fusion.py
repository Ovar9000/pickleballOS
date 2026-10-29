import numpy as np
from scipy.signal import butter, sosfilt

from pbos.audio import SR, Onset, OnsetDetector
from pbos.ball import BallEvent
from pbos.fusion import Contact, ContactFusion, PlayerSnapshot, RallyTracker


def pop(rng, ms=12):
    """A paddle-pop-like burst: band-limited noise with a fast exponential decay."""
    n = int(SR * ms / 1000)
    sos = butter(4, (1500, 6000), btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, rng.normal(0, 1, n)) * np.exp(-np.arange(n) / (n / 5))


def test_onsets_found_at_the_right_times_and_not_in_noise():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.01, SR * 6).astype(np.float32)  # 6 s of background noise
    times = [1.0, 1.9, 2.55, 3.8, 5.1]
    for t in times:
        k = int(t * SR)
        p = pop(rng) * 0.5
        audio[k:k + len(p)] += p.astype(np.float32)
    onsets = OnsetDetector().detect(audio, t0=100.0)
    assert len(onsets) == len(times)
    for o, t in zip(onsets, times):
        assert abs(o.t - (100.0 + t)) < 0.01
        assert o.rise_db > 12


def player(x, y, wrist):
    return PlayerSnapshot(np.array([x - 30, y - 180, x + 30, y], float), np.array([wrist], float),
                          np.array([5.0, 10.0]))


def run_fusion(vision=(), onsets=(), ball_uv=None, until=2.0, in_flight=True, require_audio=True):
    f = ContactFusion(require_audio=require_audio)
    players = {"P1": player(400, 500, (430, 380))}  # wrist at (430, 380), box height 180
    f.add_onsets(list(onsets))
    out = []
    for k in range(int(until * 30)):
        t = k / 30
        f.add_frame(t, None if ball_uv is None else np.array(ball_uv, float), players, in_flight)
        f.add_vision([e for e in vision if k / 30 - 1 / 30 < e.t <= t])
        out += f.resolve(t)
    return out


def ev(kind, t, uv, xy=(5.0, 10.0)):
    return BallEvent(kind, t, np.array(uv, float), np.array(xy), 12.0)


def test_direction_change_at_the_paddle_with_a_pop_is_a_hit():
    out = run_fusion(vision=[ev("bounce", 0.5, (440, 390))], onsets=[Onset(0.52, 20, 5)])
    assert [(c.kind, c.source, c.slot) for c in out] == [("hit", "audio+vision", "P1")]
    assert abs(out[0].t - 0.52) < 1e-9  # audio sets the time


def test_low_falling_change_away_from_players_is_a_bounce():
    out = run_fusion(vision=[ev("bounce", 0.5, (900, 600), xy=(12.0, 30.0))])
    assert [(c.kind, c.source) for c in out] == [("bounce", "vision")]
    np.testing.assert_allclose(out[0].court_xy, (12.0, 30.0))


def test_dead_ball_direction_change_without_pop_or_paddle_is_dropped():
    assert run_fusion(vision=[ev("hit", 0.5, (900, 600))]) == []


def test_silent_direction_change_at_a_hand_needs_audio_unless_there_is_none():
    caught = [ev("hit", 0.5, (440, 390))]  # e.g. a player catching the ball between points
    assert run_fusion(vision=caught) == []
    assert [c.kind for c in run_fusion(vision=caught, require_audio=False)] == ["hit"]


def test_pop_without_direction_change_counts_only_with_the_ball_at_a_paddle():
    hidden = run_fusion(onsets=[Onset(0.5, 20, 5)], ball_uv=(425, 385))
    assert [(c.kind, c.source, c.slot) for c in hidden] == [("hit", "audio", "P1")]
    held = run_fusion(onsets=[Onset(0.5, 20, 5)], ball_uv=(425, 385), in_flight=False)
    assert held == []  # ball resting in a hand while another court pops
    other_court = run_fusion(onsets=[Onset(0.5, 20, 5)], ball_uv=(1100, 300))
    assert other_court == []


def test_rallies_split_on_long_gaps():
    r = RallyTracker(gap_s=3.0)
    for t in (0.0, 1.0, 1.4, 2.2):
        r.on_contact(Contact(t, "hit", "audio+vision"))
    r.on_contact(Contact(2.6, "bounce", "vision"))
    for t in (9.0, 10.1):
        r.on_contact(Contact(t, "hit", "audio"))
    r.step(20.0)
    assert [(x.hits, x.bounces) for x in r.rallies] == [(4, 1), (2, 0)]
