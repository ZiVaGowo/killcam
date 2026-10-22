"""Метрики: Prometheus (/metrics, Grafana) + rolling-окно для HTML-дашборда (/dashboard)."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)

REQUESTS = Counter("killcam_requests_total", "Requests", ["endpoint", "status"])
TTFT = Histogram("killcam_ttft_seconds", "Time to first token/field", ["endpoint"], buckets=BUCKETS)
LATENCY = Histogram("killcam_latency_seconds", "End-to-end latency", ["endpoint"], buckets=BUCKETS)
TTS_TTFB = Histogram("killcam_tts_ttfb_seconds", "TTS bumper time to first audio byte", buckets=BUCKETS)
COST = Counter("killcam_cost_usd_total", "Estimated cost in USD", ["component"])
GUARD_EVENTS = Counter("killcam_guard_events_total", "Guard events", ["kind"])
SCORE_HALLUC = Counter("killcam_score_hallucinations_total", "Model score != HUD score (caught by guard)")
CARDS = Counter("killcam_cards_total", "Cards produced", ["event"])
HIGHLIGHTS = Counter("killcam_highlights_total", "Highlights found in batch")
THUMB_CACHE = Counter("killcam_thumbnail_cache_total", "Thumbnail cache lookups", ["result"])
DRIFT = Gauge("killcam_drift_score", "Screen embedding drift vs UI reference")
QUEUE_DEPTH = Gauge("killcam_batch_queue_depth", "Batch jobs waiting")
HALLUC_RATE = Gauge("killcam_score_halluc_rate", "Rolling score hallucination rate")
FAITHFULNESS = Gauge("killcam_faithfulness", "Rolling faithfulness of card facts")


class RollingStore:
    def __init__(self, maxlen: int = 2000):
        self.records: deque[dict] = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.highlights = 0
        self.cost_total = 0.0
        self.drift_series: deque[tuple[float, float]] = deque(maxlen=500)

    def record(self, **kw) -> None:
        kw.setdefault("ts", time.time())
        with self.lock:
            self.records.append(kw)
            self.cost_total += kw.get("cost", 0.0)
            cards = [r for r in self.records if "halluc" in r]
            if cards:
                HALLUC_RATE.set(sum(r["halluc"] for r in cards) / len(cards))
                FAITHFULNESS.set(float(np.mean([r.get("faithfulness", 1.0) for r in cards])))

    def add_highlights(self, n: int, cost: float) -> None:
        with self.lock:
            self.highlights += n
            self.cost_total += cost
        HIGHLIGHTS.inc(n)

    def snapshot(self) -> dict:
        with self.lock:
            recs = list(self.records)
            drift = list(self.drift_series)
            highlights, cost_total = self.highlights, self.cost_total

        def pct(vals, q):
            return round(float(np.percentile(vals, q)), 4) if vals else None

        lat = [r["latency"] for r in recs if r.get("latency") is not None]
        ttft = [r["ttft"] for r in recs if r.get("ttft") is not None]
        tts = [r["tts_ttfb"] for r in recs if r.get("tts_ttfb") is not None]
        cards = [r for r in recs if "halluc" in r]
        kills = sum(1 for r in cards if r.get("event") == "kill")
        return {
            "requests": len(recs),
            "latency_p50": pct(lat, 50),
            "latency_p95": pct(lat, 95),
            "ttft_p50": pct(ttft, 50),
            "ttft_p95": pct(ttft, 95),
            "tts_ttfb_p95": pct(tts, 95),
            "score_halluc_rate": round(sum(r["halluc"] for r in cards) / len(cards), 4) if cards else 0.0,
            "faithfulness": round(float(np.mean([r.get("faithfulness", 1.0) for r in cards])), 4) if cards else 1.0,
            "cost_total_usd": round(cost_total, 4),
            "cost_per_highlight_usd": round(cost_total / max(1, highlights + kills), 4),
            "highlights": highlights + kills,
            "drift": [{"ts": t, "score": s} for t, s in drift[-100:]],
            "series": [
                {k: r.get(k) for k in ("ts", "endpoint", "latency", "ttft", "cost", "halluc")} for r in recs[-200:]
            ],
        }


STORE = RollingStore()
