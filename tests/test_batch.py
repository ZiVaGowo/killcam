"""Batch: запись матча → лента хайлайтов (мультикиллы, эйс, снос линии, рекорды), cost, очередь, изоляция."""

import base64
import json
import time

from app.config import ROOT

VOD = ROOT / "data" / "vod" / "final_g5"


def _items():
    return [{"image_b64": base64.b64encode(p.read_bytes()).decode()} for p in sorted(VOD.glob("frame_*.png"))]


def _wait(client, job_id, headers, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/v1/batch/{job_id}", headers=headers).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_vod_batch_highlight_reel(client):
    items = _items()
    h = {"X-Tenant-ID": "studio_a"}
    r = client.post("/v1/batch", json={"match_id": "final_g5", "items": items, "make_thumbnails": True}, headers=h)
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"], h)
    assert job["status"] == "done", job.get("error")
    assert job["processed"] == len(items)

    expected = [json.loads(x) for x in (ROOT / "evals/golden/match_final_g5.jsonl").read_text(encoding="utf-8").splitlines()]
    found = {(x["type"], round(x["game_time"])) for x in job["highlights"]}
    for e in expected:
        assert (e["type"], e["game_time"]) in found, e
    assert len(job["highlights"]) == len(expected)  # и ничего лишнего (медленный снос линии — не хайлайт)

    wipe = next(x for x in job["highlights"] if x["type"] == "one_man_wipe")
    assert wipe["hype"] == 100 and wipe["players"] == ["Vex"] and wipe["should_announce"]
    assert set(wipe["details"]["also"]) == {"triple_kill", "killing_spree", "ace"}  # одна фраза вместо четырёх
    assert job["summary"]["kills"] == 13
    assert job["cost_usd"] > 0 and job["cost_per_highlight_usd"] > 0

    csv = client.get(f"/v1/batch/{job['job_id']}/highlights.csv", headers=h).text
    assert csv.splitlines()[0].startswith("frame,clock,type,hype,title")
    assert len(csv.strip().splitlines()) == len(expected) + 1

    # изоляция данных: другой тенант не видит задачу
    assert client.get(f"/v1/batch/{job['job_id']}", headers={"X-Tenant-ID": "studio_b"}).status_code == 404


def test_batch_thumbnail_cache_saves_cost(client):
    items = _items()[:60]  # первые минуты: первая кровь, быстрая башня, снос линии
    h = {"X-Tenant-ID": "cache_t"}
    j1 = _wait(client, client.post("/v1/batch", json={"match_id": "m1", "items": items, "make_thumbnails": True}, headers=h).json()["job_id"], h)
    j2 = _wait(client, client.post("/v1/batch", json={"match_id": "m1", "items": items, "make_thumbnails": True}, headers=h).json()["job_id"], h)
    loud1 = [x for x in j1["highlights"] if x["thumbnail_cached"] is not None]
    loud2 = [x for x in j2["highlights"] if x["thumbnail_cached"] is not None]
    assert loud1 and all(x["thumbnail_cached"] is False for x in loud1)
    assert loud2 and all(x["thumbnail_cached"] is True for x in loud2)
    assert j2["cost_usd"] < j1["cost_usd"]
