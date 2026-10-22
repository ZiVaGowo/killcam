"""Batch ночных VOD: 100 скринов → таблица хайлайтов.

Очередь: Redis (docker-compose, отдельный контейнер-воркер, можно на GPU) или in-memory
(поток внутри API — для локального запуска и тестов). Задачи изолированы по тенанту.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import queue
import threading
import time
import uuid
from typing import Protocol

from app.media import ThumbnailCache
from app.monitoring.cost import CostMeter
from app.monitoring.metrics import QUEUE_DEPTH, STORE
from app.pipeline import MomentInput, Pipeline

HIGHLIGHT_COLUMNS = ["frame", "timestamp", "event", "hero", "score", "headline", "needs_review", "thumbnail_cached", "cost_usd"]


class JobQueue(Protocol):
    def submit(self, tenant: str, payload: dict) -> dict: ...
    def get(self, tenant: str, job_id: str) -> dict | None: ...
    def update(self, job: dict) -> None: ...
    def pop(self, timeout: float = 1.0) -> tuple[dict, dict] | None: ...
    def depth(self) -> int: ...


class MemoryQueue:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.payloads: dict[str, dict] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()

    def submit(self, tenant: str, payload: dict) -> dict:
        job = {"job_id": uuid.uuid4().hex[:12], "tenant": tenant, "status": "queued", "total": len(payload["items"]),
               "processed": 0, "highlights": [], "cost_usd": 0.0, "error": None, "created_at": time.time()}
        with self.lock:
            self.jobs[job["job_id"]] = job
            self.payloads[job["job_id"]] = payload
        self.q.put(job["job_id"])
        QUEUE_DEPTH.set(self.q.qsize())
        return dict(job)

    def get(self, tenant: str, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        return dict(job) if job and job["tenant"] == tenant else None

    def update(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["job_id"]] = dict(job)

    def pop(self, timeout: float = 1.0):
        try:
            jid = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        QUEUE_DEPTH.set(self.q.qsize())
        return dict(self.jobs[jid]), self.payloads.pop(jid)

    def depth(self) -> int:
        return self.q.qsize()


class RedisQueue:
    QUEUE = "killcam:batch:queue"

    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"killcam:batch:job:{job_id}"

    def submit(self, tenant: str, payload: dict) -> dict:
        job = {"job_id": uuid.uuid4().hex[:12], "tenant": tenant, "status": "queued", "total": len(payload["items"]),
               "processed": 0, "highlights": [], "cost_usd": 0.0, "error": None, "created_at": time.time()}
        self.r.set(self._key(job["job_id"]), json.dumps(job), ex=7 * 86400)
        self.r.set(self._key(job["job_id"]) + ":payload", json.dumps(payload), ex=86400)
        self.r.rpush(self.QUEUE, job["job_id"])
        QUEUE_DEPTH.set(self.depth())
        return job

    def get(self, tenant: str, job_id: str) -> dict | None:
        raw = self.r.get(self._key(job_id))
        job = json.loads(raw) if raw else None
        return job if job and job["tenant"] == tenant else None

    def update(self, job: dict) -> None:
        self.r.set(self._key(job["job_id"]), json.dumps(job), ex=7 * 86400)

    def pop(self, timeout: float = 1.0):
        item = self.r.blpop([self.QUEUE], timeout=max(1, int(timeout)))
        if not item:
            return None
        jid = item[1]
        job = json.loads(self.r.get(self._key(jid)) or "null")
        payload = json.loads(self.r.getdel(self._key(jid) + ":payload") or "null")
        QUEUE_DEPTH.set(self.depth())
        return (job, payload) if job and payload else None

    def depth(self) -> int:
        return int(self.r.llen(self.QUEUE))


def make_queue(redis_url: str) -> JobQueue:
    return RedisQueue(redis_url) if redis_url else MemoryQueue()


async def process_job(job: dict, payload: dict, pipeline: Pipeline, jq: JobQueue, thumbs: ThumbnailCache | None) -> dict:
    job["status"] = "running"
    jq.update(job)
    total_cost = 0.0
    try:
        for i, item in enumerate(payload["items"]):
            inp = MomentInput(
                match_id=payload["match_id"],
                image=base64.b64decode(item["image_b64"]),
                audio=base64.b64decode(item["audio_b64"]) if item.get("audio_b64") else None,
                highlights_only=True,
            )
            res = await pipeline.run(inp)
            frame_cost = res.cost_usd
            if res.card and not res.skipped:
                cached = None
                if payload.get("make_thumbnails") and thumbs is not None and res.card.get("timestamp"):
                    c = res.card
                    _, cached = thumbs.get_or_render(job["tenant"], c["match_id"], c["timestamp"], c["event"], c["hero"], c["score"])
                    if not cached:
                        frame_cost += CostMeter().image_gen()
                job["highlights"].append({
                    "frame": i, "timestamp": res.card["timestamp"] or item.get("timestamp_hint"), "event": res.card["event"],
                    "hero": res.card["hero"], "score": res.card["score"], "headline": res.card["headline"],
                    "needs_review": res.card["needs_review"], "thumbnail_cached": cached, "cost_usd": round(frame_cost, 6),
                })
            total_cost += frame_cost
            job["processed"] = i + 1
            job["cost_usd"] = round(total_cost, 6)
            if i % 10 == 0:
                jq.update(job)
        job["status"] = "done"
        job["cost_per_highlight_usd"] = round(total_cost / max(1, len(job["highlights"])), 6)
        STORE.add_highlights(len(job["highlights"]), total_cost)
    except Exception as e:  # noqa: BLE001 — статус задачи должен отразить любую ошибку
        job["status"] = "failed"
        job["error"] = repr(e)
    jq.update(job)
    return job


def highlights_csv(job: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HIGHLIGHT_COLUMNS)
    w.writeheader()
    for h in job["highlights"]:
        w.writerow({k: h.get(k) for k in HIGHLIGHT_COLUMNS})
    return buf.getvalue()


class InlineWorker(threading.Thread):
    """Воркер в фоне процесса API (когда нет Redis)."""

    def __init__(self, pipeline: Pipeline, jq: JobQueue, thumbs: ThumbnailCache | None):
        super().__init__(daemon=True, name="killcam-batch-worker")
        self.pipeline, self.jq, self.thumbs = pipeline, jq, thumbs
        self.stop_event = threading.Event()

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        while not self.stop_event.is_set():
            item = self.jq.pop(timeout=0.5)
            if item:
                loop.run_until_complete(process_job(item[0], item[1], self.pipeline, self.jq, self.thumbs))
        loop.close()
