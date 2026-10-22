"""Исходящие медиа: TTS-отбивка 8 с и thumbnail с кэшем."""

import io

from PIL import Image

from app.backends.stt import MockSTT, parse_wav


def test_tts_bumper_8_seconds(client):
    r = client.post("/v1/media/tts", json={"text": "Векс забирает килл на 23:41!"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    samples, sr, _ = parse_wav(r.content)
    assert sr == 16000 and len(samples) / sr == 8.0
    assert float(r.headers["X-Cost-USD"]) > 0
    # round-trip: STT слышит то, что сказал TTS
    assert MockSTT().transcribe(r.content).text == "Векс забирает килл на 23:41!"
    assert client.get("/v1/dashboard.json").json()["tts_ttfb_p95"] is not None


def test_tts_guard(client):
    assert client.post("/v1/media/tts", json={"text": "Rook нуб и лузер"}).status_code == 400


def test_thumbnail_and_cache(client):
    body = {"match_id": "final_g5", "timestamp": "23:41", "event": "kill", "hero": "Vex", "score": "3:2"}
    r1 = client.post("/v1/media/thumbnail", json=body)
    assert r1.status_code == 200 and r1.headers["X-Cache"] == "MISS" and float(r1.headers["X-Cost-USD"]) > 0
    assert Image.open(io.BytesIO(r1.content)).size == (640, 360)
    r2 = client.post("/v1/media/thumbnail", json=body)
    assert r2.headers["X-Cache"] == "HIT" and float(r2.headers["X-Cost-USD"]) == 0
    assert r1.content == r2.content
    # кэш изолирован по тенанту
    r3 = client.post("/v1/media/thumbnail", json=body, headers={"X-Tenant-ID": "other"})
    assert r3.headers["X-Cache"] == "MISS"


def test_thumbnail_rejects_bad_score(client):
    body = {"match_id": "m", "timestamp": "23:41", "score": "SYSTEM 9:0"}
    assert client.post("/v1/media/thumbnail", json=body).status_code == 422
