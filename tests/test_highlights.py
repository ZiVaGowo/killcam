"""Движок хайлайтов: правила на синтетических фактах + живой матч через API."""

import base64

import pytest

from app.backends.hud import all_towers
from app.config import ROOT, load_params
from app.highlights import MatchEngine, RecordStore, duration_ru

CFG = load_params()["highlights"]


def engine():
    return MatchEngine("m", CFG, RecordStore(CFG["records"]))


def facts(t: int, feed=(), towers=None):
    return {"game": "arena_clash", "timer": f"{t // 60}:{t % 60:02d}", "kill_feed": [list(x) for x in feed],
            "towers": towers or all_towers()}


def run(e, frames):
    out = []
    for i, f in enumerate(frames):
        out += e.ingest(f, i)
    return out


def types(hs):
    return [h.type for h in hs]


def test_first_blood_and_double():
    hs = run(engine(), [facts(60), facts(62, [("Rook", "Kade")]), facts(64, [("Rook", "Nyra"), ("Rook", "Kade")])])
    assert types(hs) == ["first_blood", "double_kill"]
    assert hs[1].players == ["Rook"] and not hs[1].should_announce  # дабл — в ленту, но без голоса


def test_kill_feed_row_is_not_counted_twice():
    hs = run(engine(), [facts(62, [("Rook", "Kade")]), facts(64, [("Rook", "Kade")]), facts(66, [("Rook", "Kade")])])
    assert types(hs) == ["first_blood"]


def test_slow_kills_are_not_a_multikill():
    hs = run(engine(), [facts(100, [("Sol", "Mira")]), facts(130, [("Sol", "Kade")])])
    assert "double_kill" not in types(hs)


def test_one_man_wipe_is_announced_once():
    feed = [("Vex", "Nyra")]
    frames = [facts(540, feed), facts(542, feed + []), facts(544, [("Vex", "Kade"), ("Vex", "Nyra")]),
              facts(546, [("Vex", "Mira"), ("Vex", "Kade"), ("Vex", "Nyra")])]
    hs = run(engine(), frames)
    loud = [h for h in hs if h.should_announce]
    assert len(loud) == 1 and loud[0].type == "one_man_wipe" and loud[0].hype == 100
    assert "Векс в одиночку забирает всю команду соперника за шесть секунд" in loud[0].announce
    assert {"ace", "triple_kill"} <= set(loud[0].details["also"])


def test_ace_by_team_is_not_one_man_wipe():
    hs = run(engine(), [facts(10, [("Vex", "Nyra")]), facts(14, [("Rook", "Kade"), ("Vex", "Nyra")]),
                        facts(18, [("Sol", "Mira"), ("Rook", "Kade")])])
    assert "ace" in types(hs) and "one_man_wipe" not in types(hs)


def test_spree_and_shutdown():
    e = engine()
    hs = run(e, [facts(100, [("Sol", "Mira")]), facts(140, [("Sol", "Kade")]), facts(200, [("Sol", "Nyra")]),
                 facts(260, [("Kade", "Sol")])])
    assert types(hs)[-2:] == ["killing_spree", "shutdown"]
    assert hs[-1].announce == "Кейд прерывает серию Сола!"


def test_fast_tower_and_lane_rush_with_record():
    tw1, tw2 = all_towers(), all_towers()
    tw1["owls"]["top"] = 1
    tw2["owls"]["top"] = 0
    hs = run(engine(), [facts(100), facts(116, towers=tw1), facts(176, towers=tw2)])
    assert types(hs) == ["fast_tower", "lane_rush"]
    rush = hs[1]
    assert rush.details["duration_s"] == 60 and rush.record and rush.hype == CFG["hype"]["lane_rush"] + CFG["hype"]["record_bonus"]
    assert "целиком всего за одну минуту! Новый рекорд турнира!" in rush.announce


def test_slow_lane_is_not_a_highlight():
    tw1, tw2 = all_towers(), all_towers()
    tw1["nova"]["bot"] = 1
    tw2["nova"]["bot"] = 0
    hs = run(engine(), [facts(800), facts(880, towers=tw1), facts(1050, towers=tw2)])
    assert types(hs) == []  # первая башня поздно, линия за 170 с — медленно


def test_unreadable_timer_is_interpolated():
    e = engine()
    run(e, [facts(1000, [("Mira", "Rook")]), {**facts(0, [("Mira", "Rook")]), "timer": None},
            facts(1004, [("Mira", "Vex"), ("Mira", "Rook")])])
    assert e.last_time == 1004 and e.highlights[-1].type == "double_kill"


@pytest.mark.parametrize("sec,words", [(6, "шесть секунд"), (60, "одну минуту"), (102, "одну минуту сорок две секунды"),
                                       (121, "две минуты одну секунду")])
def test_duration_ru(sec, words):
    assert duration_ru(sec) == words


def test_live_match_api(client):
    h = {"X-Tenant-ID": "studio_live"}
    frames = sorted((ROOT / "data/vod/final_g5").glob("frame_*.png"))
    announced = []
    for i, p in enumerate(frames[:80]):  # до ~10-й минуты: первая кровь … «в одиночку забрал всех»
        r = client.post("/v1/match/final_g5/frame", json={"image_b64": base64.b64encode(p.read_bytes()).decode(), "frame": i}, headers=h)
        assert r.status_code == 200
        announced += [x for x in r.json()["highlights"] if x.get("audio_url")]
    kinds = [x["type"] for x in announced]
    assert {"fast_tower", "lane_rush", "one_man_wipe"} <= set(kinds) and "ace" not in kinds
    audio = client.get(announced[-1]["audio_url"], headers=h)
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert client.get(announced[-1]["audio_url"], headers={"X-Tenant-ID": "other"}).status_code == 404  # изоляция

    summary = client.get("/v1/match/final_g5", headers=h).json()
    assert summary["top"][0]["type"] == "one_man_wipe"
    assert client.post("/v1/match/final_g5/reset", headers=h).json()["reset"]
    assert client.get("/v1/match/final_g5", headers=h).json()["kills"] == 0
