"""Отдельный батч-воркер (docker-compose service `worker`, при наличии GPU — на GPU).

    python -m app.worker        # нужен REDIS_URL
"""

from __future__ import annotations

import asyncio
import logging
import os

from prometheus_client import start_http_server

from app.batch import RedisQueue, process_job
from app.config import get_settings
from app.media import ThumbnailCache
from app.monitoring.drift import DriftMonitor
from app.pipeline import Pipeline
from app.registry import Registry

log = logging.getLogger("killcam.worker")


async def main() -> None:
    s = get_settings()
    if not s.redis_url:
        raise SystemExit("REDIS_URL is required for the standalone worker")
    start_http_server(int(os.environ.get("KILLCAM_WORKER_METRICS_PORT", "9100")))
    drift_cfg = s.params["drift"]
    pipeline = Pipeline(s, Registry(s.registry_path), DriftMonitor(s.reference_path, drift_cfg["window"], drift_cfg["threshold"]))
    jq = RedisQueue(s.redis_url)
    thumbs = ThumbnailCache(s.runtime_dir)
    log.info("worker started device=%s", s.device)
    while True:
        item = await asyncio.to_thread(jq.pop, 5)
        if item:
            job = await process_job(item[0], item[1], pipeline, jq, thumbs)
            log.info("job %s %s highlights=%d cost=$%.4f", job["job_id"], job["status"], len(job["highlights"]), job["cost_usd"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
