"""Batch: ночной VOD 100 скринов → таблица хайлайтов, cost, очередь, изоляция."""

import base64
import json
import time

from app.config import ROOT

VOD = ROOT / "data" / "vod" / "final_g5"


def _items():
    return [{"image_b64": base64.b64encode(p.read_bytes()).decode()} for p in sorted(VOD.glob("frame_*.png"))]


def _wait(client, job_id, headers, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/v1/batch/{job_id}", headers=headers).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_vod_batch_highlights(client):
    items = _items()
    assert len(items) == 100
    h = {"X-Tenant-ID": "studio_a"}
    r = client.post("/v1/batch", json={"match_id": "final_g5", "items": items, "make_thumbnails": True}, headers=h)
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"], h)
    assert job["status"] == "done", job.get("error")
    assert job["processed"] == 100

    manifest = [json.loads(x) for x in (VOD / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = [m for m in manifest if m["event"] in ("kill", "objective")]
    assert len(job["highlights"]) == len(expected)
    by_frame = {m["frame"]: m for m in manifest}
    for hl in job["highlights"]:
        assert hl["score"] == by_frame[hl["frame"]]["score"]  # ни одного выдуманного счёта
    assert job["cost_usd"] > 0 and job["cost_per_highlight_usd"] > 0

    csv = client.get(f"/v1/batch/{job['job_id']}/highlights.csv", headers=h).text
    assert csv.splitlines()[0].startswith("frame,timestamp,event,hero,score")
    assert len(csv.strip().splitlines()) == len(expected) + 1

    # изоляция данных: другой тенант не видит задачу
    assert client.get(f"/v1/batch/{job['job_id']}", headers={"X-Tenant-ID": "studio_b"}).status_code == 404


def test_batch_thumbnail_cache_saves_cost(client):
    item = {"image_b64": base64.b64encode((ROOT / "evals/fixtures/screens/ok_01.png").read_bytes()).decode()}
    h = {"X-Tenant-ID": "cache_t"}
    j1 = _wait(client, client.post("/v1/batch", json={"match_id": "m1", "items": [item], "make_thumbnails": True}, headers=h).json()["job_id"], h)
    j2 = _wait(client, client.post("/v1/batch", json={"match_id": "m1", "items": [item], "make_thumbnails": True}, headers=h).json()["job_id"], h)
    assert j1["highlights"][0]["thumbnail_cached"] is False
    assert j2["highlights"][0]["thumbnail_cached"] is True
    assert j2["cost_usd"] < j1["cost_usd"]
