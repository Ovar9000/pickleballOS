import numpy as np

from pbos.players import LEFT_ANKLE, RIGHT_ANKLE, PersonDetection
from pbos.tracking import CourtTracker, DistanceMeter, Measurement

FPS = 30.0


def m(x, y, sigma=0.3):
    return Measurement(np.array([x, y], dtype=float), sigma)


def run(tracker, positions_fn, seconds, det_every=2, drop=lambda i: False, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if i % det_every == 0 and not drop(i):
            ms = [m(*(np.array(p) + rng.normal(0, noise, 2))) for p in positions_fn(t)]
            tracker.step(t, ms)
        else:
            tracker.step(t, None)
    return tracker


def doubles(t):
    # Near team at the kitchen line, far team at the baseline, all shuffling slowly sideways.
    s = np.sin(t)
    return [(5 + s, 14), (15 + s, 14), (5 - s, 42), (15 - s, 42)]


def test_four_players_get_four_slots_ordered_near_to_far():
    tr = run(CourtTracker(), doubles, 2.0)
    players = tr.players()
    assert sorted(players) == ["P1", "P2", "P3", "P4"]
    assert players["P1"].xy[1] < 22 and players["P2"].xy[1] < 22  # near side first
    assert players["P1"].xy[0] < players["P2"].xy[0]  # then left to right


def test_fifth_person_does_not_get_a_slot():
    tr = run(CourtTracker(), lambda t: doubles(t) + [(10, -8)], 2.0)
    assert len(tr.players()) == 4


def test_third_person_on_one_side_stays_unassigned():
    # A spectator standing behind the far baseline while one near player is still undetected.
    pts = lambda t: [doubles(t)[0]] + doubles(t)[2:] + [(10, 52)]
    tr = run(CourtTracker(), pts, 2.0)
    players = tr.players()
    assert len(players) == 3
    assert sum(p.xy[1] > 22 for p in players.values()) == 2


def test_short_occlusion_keeps_slot():
    # P2 disappears for 0.5 s (under the max age) and must come back as P2.
    first = run(CourtTracker(), doubles, 2.0)
    p2_id = first.players()["P2"].id
    tr = first
    for i in range(int(2.0 * FPS), int(3.0 * FPS)):
        t = i / FPS
        pts = doubles(t)
        if i < int(2.5 * FPS):
            pts = [pts[0]] + pts[2:]
        tr.step(t, [m(*p) for p in pts] if i % 2 == 0 else None)
    assert tr.players()["P2"].id == p2_id


def test_long_occlusion_reacquires_same_slot_by_position():
    tr = run(CourtTracker(), doubles, 2.0)
    for i in range(int(2.0 * FPS), int(6.0 * FPS)):
        t = i / FPS
        pts = doubles(t)
        if i < int(4.0 * FPS):  # P3 gone for 2 s: the track expires, the slot is released
            pts = pts[:2] + pts[3:]
        tr.step(t, [m(*p) for p in pts] if i % 2 == 0 else None)
    players = tr.players()
    assert sorted(players) == ["P1", "P2", "P3", "P4"]
    assert players["P3"].xy[1] > 30 and players["P3"].xy[0] < 10


def test_referee_beside_the_net_never_gets_a_slot():
    # Three players plus someone standing 2.5 ft outside the sideline at the net, all session long.
    tr = run(CourtTracker(), lambda t: doubles(t)[:3] + [(22.5, 22)], 3.0)
    assert len(tr.players()) == 3
    assert all(p.xy[0] < 20 for p in tr.players().values())


def test_reacquired_players_keep_their_end_of_the_court():
    # Everyone vanishes for 3 s (a cut-away), then reappears having drifted 9-10 ft. Without the
    # net-side penalty the greedy choice can hand a near player a far player's slot.
    tr = run(CourtTracker(), doubles, 2.0)
    before = {s: p.xy[1] < 22 for s, p in tr.players().items()}
    shifted = lambda t: [(5, 6), (15, 6), (12, 36), (6, 36)]
    for i in range(int(2.0 * FPS), int(5.0 * FPS)):
        tr.step(i / FPS, [] if i % 2 == 0 else None)
    for i in range(int(5.0 * FPS), int(6.0 * FPS)):
        tr.step(i / FPS, [m(*p) for p in shifted(i / FPS)] if i % 2 == 0 else None)
    after = {s: p.xy[1] < 22 for s, p in tr.players().items()}
    assert after == before


def test_lone_far_player_reappearing_first_does_not_take_a_near_slot():
    tr = run(CourtTracker(), doubles, 2.0)
    for i in range(int(2.0 * FPS), int(5.0 * FPS)):
        tr.step(i / FPS, [] if i % 2 == 0 else None)
    # Only one far player is visible again, 15 ft from where either far slot was last seen.
    for i in range(int(5.0 * FPS), int(5.5 * FPS)):
        tr.step(i / FPS, [m(10, 30)] if i % 2 == 0 else None)
    (slot, p), = tr.players().items()
    assert slot in ("P3", "P4")


def test_distance_meter_rejects_teleports():
    meter = DistanceMeter()
    for k in range(30):
        meter.add(np.array([5.0 + 0.1 * k, 10.0]), t=k / 30)
    before = meter.total_ft
    meter.add(np.array([18.0, 40.0]), t=1.0)  # 33 ft in 1/30 s: an identity swap, not running
    for k in range(1, 10):
        meter.add(np.array([18.0, 40.0]), t=1.0 + k / 30)
    assert meter.rejected_jumps == 1
    assert meter.total_ft - before < 0.5


def test_filter_tracks_a_running_player():
    # 15 ft/s sprint across the court must not break the track.
    tr = run(CourtTracker(), lambda t: [(2 + 15 * min(t, 1.0), 10)], 2.0)
    players = tr.players()
    assert list(players) == ["P1"]
    assert abs(players["P1"].xy[0] - 17) < 0.5


def test_distance_meter_ignores_jitter_but_counts_walking():
    rng = np.random.default_rng(1)
    still = DistanceMeter()
    for _ in range(300):
        still.add(np.array([10.0, 15.0]) + rng.normal(0, 0.08, 2))
    assert still.total_ft < 1.0

    walk = DistanceMeter()
    for x in np.linspace(0, 20, 600):  # 20 ft walk sampled at 30 Hz with jitter
        walk.add(np.array([x, 15.0]) + rng.normal(0, 0.08, 2))
    assert 19.0 < walk.total_ft < 21.5


def test_ground_point_prefers_ankles_then_box():
    box = np.array([100, 100, 140, 300], dtype=float)
    kps = np.zeros((17, 2))
    kps[LEFT_ANKLE], kps[RIGHT_ANKLE] = (110, 290), (130, 294)
    scores = np.zeros(17)
    scores[LEFT_ANKLE] = scores[RIGHT_ANKLE] = 0.9
    uv, cue = PersonDetection(box, kps, scores).ground_point()
    assert cue == "ankles" and np.allclose(uv, (120, 292))
    scores[RIGHT_ANKLE] = 0.1
    assert PersonDetection(box, kps, scores).ground_point()[1] == "ankle"
    uv, cue = PersonDetection(box).ground_point()
    assert cue == "box" and np.allclose(uv, (120, 300))
