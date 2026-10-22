"""Kill Cam API (FastAPI).

    /health                         смоук + версия в Production + устройство (cuda/cpu)
    /v1/chat/completions            OpenAI-совместимый, stream=true → SSE по-полевой стрим карточки
    /v1/batch                       ночные VOD: 100 скринов → таблица хайлайтов (+ .csv)
    /v1/media/tts                   8-секундная отбивка (стрим WAV)
    /v1/media/thumbnail             обложка «KILL · 23:41» (кэш по match_id+timestamp)
    /v1/tools/call                  allow-list; publish_social → HITL
    /v1/hitl/*                      одобрение оператором
    /v1/registry/*                  версии, promote (через eval-гейт), rollback
    /metrics, /dashboard            Prometheus и HTML+JSON дашборд
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.backends.stt import extract_mentions, get_stt
from app.backends.tts import get_tts
from app.batch import InlineWorker, highlights_csv, make_queue
from app.config import ROOT, Settings, get_settings
from app.highlights import MatchHub, RecordStore
from app.media import ThumbnailCache, stream_bytes, synth_bumper
from app.monitoring.cost import CostMeter
from app.monitoring.drift import DriftMonitor
from app.monitoring.metrics import LATENCY, REQUESTS, STORE, TTFT, TTS_TTFB
from app.pipeline import MomentInput, Pipeline, faithfulness
from app.registry import Registry, RegistryError
from app.schemas import (
    BatchRequest,
    ChatCompletionRequest,
    MatchFrameRequest,
    PromoteRequest,
    RegisterRequest,
    ThumbnailRequest,
    ToolCallRequest,
    TTSRequest,
)
from app.tools import ALLOW_LIST, HITLQueue, execute_tool

TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = None
        if not s.redis_url and s.inline_worker:
            worker = InlineWorker(app.state.pipeline, app.state.queue, app.state.thumbs)
            worker.start()
        # реальные речевые модели грузим в фоне сразу при старте — первый запрос из эфира не ждёт 30 с
        for engine in (get_stt(s.stt_backend, s.device), get_tts(s.tts_backend, s.device)):
            if hasattr(engine, "_load"):
                threading.Thread(target=engine._load, daemon=True, name=f"warmup-{engine.version}").start()
        yield
        if worker:
            worker.stop_event.set()

    app = FastAPI(title="Kill Cam", version=__version__, lifespan=lifespan)
    drift_cfg = s.params["drift"]
    app.state.settings = s
    app.state.registry = Registry(s.registry_path)
    app.state.drift = DriftMonitor(s.reference_path, drift_cfg["window"], drift_cfg["threshold"])
    app.state.pipeline = Pipeline(s, app.state.registry, app.state.drift)
    app.state.queue = make_queue(s.redis_url)
    app.state.thumbs = ThumbnailCache(s.runtime_dir)
    app.state.hitl = HITLQueue(s.runtime_dir / "hitl.json")
    hl_cfg = s.params["highlights"]
    app.state.matches = MatchHub(hl_cfg, RecordStore(hl_cfg["records"], s.runtime_dir / "records.json"))

    # ------------------------------------------------------------ deps
    def tenant(x_tenant_id: str = Header(default="default")) -> str:
        if not TENANT_RE.match(x_tenant_id):
            raise HTTPException(400, "bad X-Tenant-ID")
        return x_tenant_id

    def operator(x_operator_key: str = Header(default="")) -> str:
        if x_operator_key != s.operator_key:
            raise HTTPException(403, "operator key required (HITL)")
        return "operator"

    # ------------------------------------------------------------ health / metrics
    @app.get("/health")
    def health():
        try:
            prod = app.state.registry.production()
        except RegistryError as e:
            raise HTTPException(503, str(e)) from e
        return {
            "status": "ok",
            "version": __version__,
            "device": s.device,
            "llm_backend": s.llm_backend,
            "stt": getattr(get_stt(s.stt_backend, s.device), "version", s.stt_backend),
            "tts": getattr(get_tts(s.tts_backend, s.device), "version", s.tts_backend),
            "queue": "redis" if s.redis_url else "memory",
            "production": {k: prod[k] for k in ("version", "model_uri", "prompt_version", "vlm", "data_hash")},
            "drift": app.state.drift.status(),
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/dashboard.json")
    def dashboard_json():
        snap = STORE.snapshot()
        snap["drift_status"] = app.state.drift.status()
        snap["queue_depth"] = app.state.queue.depth()
        snap["production"] = app.state.registry.production()
        return snap

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui")

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def ui():
        return (STATIC / "ui.html").read_text(encoding="utf-8")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return RedirectResponse("/ui#monitor")

    # ------------------------------------------------------------ демо-данные для UI
    @app.get("/v1/demo/fixtures")
    def demo_fixtures():
        def rel(p: Path) -> str:
            return p.relative_to(ROOT).as_posix()

        def jsonl(name: str) -> list[dict]:
            f = ROOT / "evals" / "golden" / f"{name}.jsonl"
            return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []

        screens = {c["image"]: c.get("note", "") for c in jsonl("screens")}
        for p in sorted((ROOT / "evals" / "fixtures" / "screens").glob("*.png")):
            screens.setdefault(rel(p), "")
        audio = [{"path": c["audio"], "transcript": c["transcript"]} for c in jsonl("audio")]
        audio += [{"path": c["audio"], "transcript": ""} for c in jsonl("multimodal") if c.get("audio")]
        vod = [{"folder": rel(d), "frames": len(list(d.glob("*.png")))} for d in sorted((ROOT / "data" / "vod").glob("*")) if d.is_dir()]
        return {
            "screens": [{"path": k, "note": v} for k, v in screens.items()],
            "audio": audio,
            "vod": vod,
            "redteam": jsonl("redteam"),
        }

    @app.get("/v1/demo/file", include_in_schema=False)
    def demo_file(path: str):
        target = (ROOT / path).resolve()
        allowed = [(ROOT / "evals" / "fixtures").resolve(), (ROOT / "data" / "vod").resolve()]
        if not any(target.is_relative_to(a) for a in allowed) or not target.is_file() or target.suffix not in (".png", ".wav"):
            raise HTTPException(404, "file not found")
        return FileResponse(target)

    @app.post("/v1/demo/vod-batch", status_code=202)
    def demo_vod_batch(body: dict, t: str = Depends(tenant)):
        folder = (ROOT / str(body.get("folder", ""))).resolve()
        if not folder.is_relative_to((ROOT / "data" / "vod").resolve()) or not folder.is_dir():
            raise HTTPException(404, "VOD folder not found")
        items = [{"image_b64": base64.b64encode(p.read_bytes()).decode()} for p in sorted(folder.glob("*.png"))]
        if not items:
            raise HTTPException(422, "no frames")
        req = BatchRequest(match_id=str(body.get("match_id") or folder.name), items=items, make_thumbnails=bool(body.get("make_thumbnails")))
        return submit_batch(req, t)

    # ------------------------------------------------------------ chat completions
    def _parse_moment(req: ChatCompletionRequest) -> MomentInput:
        inp = MomentInput(match_id=req.match_id, chat=req.chat, overlay_text=req.overlay_text)
        texts = []
        for m in req.messages:
            if m.role != "user":  # клиентский system/assistant игнорируем: system-промпт только из registry
                continue
            parts = [{"type": "text", "text": m.content}] if isinstance(m.content, str) else m.content
            for p in parts:
                try:
                    if p.get("type") == "text":
                        texts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
                        inp.image = base64.b64decode(url.split(",", 1)[1] if url.startswith("data:") else url)
                    elif p.get("type") == "input_audio":
                        inp.audio = base64.b64decode(p["input_audio"]["data"])
                except (KeyError, IndexError, binascii.Error) as e:
                    raise HTTPException(422, f"bad content part: {p.get('type')}") from e
        inp.user_text = "\n".join(t for t in texts if t)
        return inp

    def _record(endpoint: str, res) -> None:
        LATENCY.labels(endpoint).observe(res.latency)
        if res.ttft is not None:
            TTFT.labels(endpoint).observe(res.ttft)
        rec = {"endpoint": endpoint, "latency": res.latency, "ttft": res.ttft, "cost": res.cost_usd}
        if res.card:
            rec.update(halluc=int(res.guard.get("hallucinated_score", False)), faithfulness=faithfulness(res.raw, res.facts), event=res.card["event"])
        STORE.record(**rec)
        REQUESTS.labels(endpoint, "refused" if res.refused else "ok").inc()

    def _killcam_ext(res) -> dict:
        return {
            "model": res.model_tag,
            "refused": res.refused,
            "refusal_reasons": res.refusal_reasons,
            "guard": res.guard,
            "facts": res.facts,
            "untrusted_data": res.data_report,
            "transcript": res.transcript,
            "schema_valid": res.schema_valid,
            "cost_usd": res.cost_usd,
            "ttft_s": res.ttft,
            "latency_s": round(res.latency, 4),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, _t: str = Depends(tenant)):
        inp = _parse_moment(req)
        if inp.image is None:
            raise HTTPException(422, "нужен кадр (image_url) — счёт берётся только с HUD")
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())
        pipe: Pipeline = app.state.pipeline

        if not req.stream:
            res = await pipe.run(inp, req.registry_version)
            _record("chat", res)
            content = json.dumps(res.card, ensure_ascii=False) if res.card else "Запрос отклонён guard'ом: " + ", ".join(res.refusal_reasons)
            return {
                "id": cid, "object": "chat.completion", "created": created, "model": res.model_tag,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "content_filter" if res.refused else "stop"}],
                "killcam": _killcam_ext(res),
            }

        async def sse():
            def chunk(delta: dict, finish: str | None = None, **extra) -> str:
                body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
                return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

            yield chunk({"role": "assistant"})
            async for ev in pipe.stream(inp, req.registry_version):
                if ev["type"] in ("delta", "refusal"):
                    yield chunk({"content": ev["text"]})
                elif ev["type"] == "final":
                    res = ev["result"]
                    _record("chat_stream", res)
                    yield chunk({}, "content_filter" if res.refused else "stop", killcam=_killcam_ext(res))
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ------------------------------------------------------------ batch
    @app.post("/v1/batch", status_code=202)
    def submit_batch(req: BatchRequest, t: str = Depends(tenant)):
        job = app.state.queue.submit(t, req.model_dump())
        REQUESTS.labels("batch", "queued").inc()
        return {"job_id": job["job_id"], "status": job["status"], "total": job["total"], "queue_depth": app.state.queue.depth()}

    @app.get("/v1/batch/{job_id}")
    def get_batch(job_id: str, t: str = Depends(tenant)):
        job = app.state.queue.get(t, job_id)
        if job is None:
            raise HTTPException(404, "job not found")  # чужой тенант тоже получает 404
        return job

    @app.get("/v1/batch/{job_id}/highlights.csv", response_class=PlainTextResponse)
    def get_batch_csv(job_id: str, t: str = Depends(tenant)):
        job = app.state.queue.get(t, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return PlainTextResponse(highlights_csv(job), media_type="text/csv")

    # ------------------------------------------------------------ media
    @app.post("/v1/media/tts")
    def tts(req: TTSRequest, _t: str = Depends(tenant)):
        from app.guards import check_instruction, is_toxic, redact_pii

        if is_toxic(req.text) or not check_instruction(req.text).allowed:
            raise HTTPException(400, "текст отбивки не прошёл guard")
        t0 = time.perf_counter()
        cfg = s.params["serving"]
        audio = synth_bumper(redact_pii(req.text), cfg["tts_seconds"], get_tts(s.tts_backend, s.device), req.voice or None)
        cost = CostMeter().tts(len(req.text))

        def gen():
            first = True
            for part in stream_bytes(audio):
                if first:
                    ttfb = time.perf_counter() - t0
                    TTS_TTFB.observe(ttfb)
                    STORE.record(endpoint="tts", tts_ttfb=ttfb, cost=cost)
                    first = False
                yield part

        return StreamingResponse(gen(), media_type="audio/wav", headers={"X-Cost-USD": f"{cost:.6f}", "X-Duration-S": str(cfg["tts_seconds"])})

    @app.get("/v1/media/voices")
    def voices():
        return get_tts(s.tts_backend, s.device).voices()

    @app.post("/v1/media/stt")
    async def stt(body: dict, _t: str = Depends(tenant)):
        """Распознать реплику кастера (WAV base64): текст + упоминания счёта/героя/события."""
        try:
            audio = base64.b64decode(body["audio_b64"])
        except (KeyError, binascii.Error) as e:
            raise HTTPException(422, "audio_b64 (WAV) required") from e
        t0 = time.perf_counter()
        engine = get_stt(s.stt_backend, s.device)
        try:
            tr = await asyncio.to_thread(engine.transcribe, audio)
        except ValueError as e:
            raise HTTPException(422, f"bad audio: {e}") from e
        cost = CostMeter().stt(tr.duration_s)
        latency = time.perf_counter() - t0
        STORE.record(endpoint="stt", latency=latency, cost=cost)
        return {"text": tr.text, "duration_s": round(tr.duration_s, 2), "stt": tr.stt_version,
                "mentions": extract_mentions(tr.text).to_dict(), "latency_s": round(latency, 3), "cost_usd": round(cost, 6)}

    @app.post("/v1/media/thumbnail")
    def thumbnail(req: ThumbnailRequest, t: str = Depends(tenant)):
        t0 = time.perf_counter()
        png, cached = app.state.thumbs.get_or_render(t, req.match_id, req.timestamp, req.event, req.hero, req.score)
        cost = 0.0 if cached else CostMeter().image_gen()
        STORE.record(endpoint="thumbnail", latency=time.perf_counter() - t0, cost=cost)
        return Response(png, media_type="image/png", headers={"X-Cache": "HIT" if cached else "MISS", "X-Cost-USD": f"{cost:.6f}"})

    # ------------------------------------------------------------ tools + HITL
    @app.get("/v1/tools")
    def list_tools():
        return ALLOW_LIST

    @app.post("/v1/tools/call")
    def call_tool(req: ToolCallRequest, t: str = Depends(tenant)):
        spec = ALLOW_LIST.get(req.tool)
        if spec is None:
            REQUESTS.labels("tool", "denied").inc()
            raise HTTPException(403, f"tool '{req.tool}' is not in allow-list")
        if spec["hitl"]:
            item = app.state.hitl.request(t, req.tool, req.args)
            return JSONResponse({"status": "pending_approval", "hitl_id": item["id"]}, status_code=202)
        return {"status": "done", "result": execute_tool(req.tool, req.args, tenant=t)}

    @app.get("/v1/hitl")
    def hitl_list(t: str = Depends(tenant)):
        return app.state.hitl.list(t)

    @app.post("/v1/hitl/{rid}/{decision}")
    def hitl_decide(rid: str, decision: str, t: str = Depends(tenant), op: str = Depends(operator)):
        if decision not in ("approve", "reject"):
            raise HTTPException(404)
        try:
            return app.state.hitl.decide(t, rid, decision == "approve", op)
        except KeyError as e:
            raise HTTPException(404, "hitl request not found") from e

    # ------------------------------------------------------------ registry
    @app.get("/v1/registry")
    def registry_list():
        return app.state.registry.load()

    @app.get("/v1/registry/prompts")
    def registry_prompts():
        return sorted(f"{d.name}:{f.stem}" for d in s.prompts_dir.iterdir() if d.is_dir() for f in d.glob("*.md"))

    @app.post("/v1/registry/register")
    def registry_register(req: RegisterRequest, _op: str = Depends(operator)):
        from scripts.data_lock import compute

        if req.prompt_version not in registry_prompts():
            raise HTTPException(422, f"unknown prompt {req.prompt_version}")
        prod = app.state.registry.production()
        return app.state.registry.register(req.model_uri or prod["model_uri"], req.prompt_version, req.vlm or prod["vlm"],
                                           compute()["data_hash"], req.notes)

    @app.post("/v1/registry/promote")
    async def registry_promote(req: PromoteRequest, _op: str = Depends(operator)):
        from evals.gate import gate_version

        if req.stage == "Production" and not req.skip_gate:
            passed, report = await gate_version(req.version, s)
            if not passed:
                return JSONResponse({"promoted": False, "gate": report}, status_code=409)
            return {"promoted": True, "version": app.state.registry.transition(req.version, "Production", report["metrics"], "gate passed"), "gate": report}
        return {"promoted": True, "version": app.state.registry.transition(req.version, req.stage, reason="manual")}

    @app.post("/v1/registry/rollback")
    def registry_rollback(_op: str = Depends(operator)):
        try:
            return app.state.registry.rollback()
        except RegistryError as e:
            raise HTTPException(409, str(e)) from e

    # ------------------------------------------------------------ живой матч: хайлайты + голос
    def match_id_ok(match_id: str) -> str:
        if not TENANT_RE.match(match_id):
            raise HTTPException(400, "bad match_id")
        return match_id

    @app.post("/v1/match/{match_id}/frame")
    async def match_frame(match_id: str, body: MatchFrameRequest, t: str = Depends(tenant)):
        """Очередной кадр трансляции -> факты с табло -> движок хайлайтов -> озвучка важных моментов."""
        match_id_ok(match_id)
        t0 = time.perf_counter()
        try:
            image = base64.b64decode(body.image_b64)
        except binascii.Error as e:
            raise HTTPException(422, "bad image_b64") from e
        pipe: Pipeline = app.state.pipeline
        a = await asyncio.to_thread(pipe.analyze, MomentInput(match_id=match_id, image=image), pipe.bundle())
        engine = app.state.matches.get(t, match_id)
        new = engine.ingest(a.facts, body.frame)
        cost = a.cost.total
        rows = []
        for h in new:
            row = h.to_dict()
            if h.should_announce and body.announce:
                tts = get_tts(s.tts_backend, s.device)
                wav = await asyncio.to_thread(synth_bumper, h.announce, s.params["serving"]["tts_seconds"], tts,
                                              body.voice or "v5_6_ru/romchik")
                cost += CostMeter().tts(len(h.announce))
                app.state.matches.store_audio(f"{t}:{h.id}", wav)
                row["audio_url"] = f"/v1/match/{match_id}/audio/{h.id}"
            rows.append(row)
        latency = time.perf_counter() - t0
        STORE.record(endpoint="match_frame", latency=latency, cost=cost)
        f = a.facts
        return {"match_id": match_id, "frame": body.frame, "game_time": engine.last_time, "clock": f.get("timer"),
                "score": f.get("score"), "kill_feed": f.get("kill_feed"), "towers": f.get("towers"),
                "highlights": rows, "kills": len(engine.kills), "latency_s": round(latency, 4), "cost_usd": round(cost, 6)}

    @app.get("/v1/match/{match_id}")
    def match_summary(match_id: str, t: str = Depends(tenant)):
        return app.state.matches.get(t, match_id_ok(match_id)).summary()

    @app.post("/v1/match/{match_id}/reset")
    def match_reset(match_id: str, t: str = Depends(tenant)):
        app.state.matches.reset(t, match_id_ok(match_id))
        return {"reset": True}

    @app.get("/v1/match/{match_id}/audio/{hid}")
    def match_audio(match_id: str, hid: str, t: str = Depends(tenant)):
        wav = app.state.matches.audio.get(f"{t}:{hid}")
        if wav is None:
            raise HTTPException(404, "audio not found")
        return Response(wav, media_type="audio/wav")

    @app.get("/v1/drift")
    def drift():
        return app.state.drift.status()

    @app.exception_handler(RegistryError)
    async def registry_error(_: Request, exc: RegistryError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    return app


app = create_app()
