"""Исходящие медиа: TTS-отбивка (8 с) и thumbnail «KILL · 23:41».

* TTS: mock-синтез на CPU или Silero (KILLCAM_TTS_BACKEND=silero), отдаётся стримом чанками WAV → меряем TTFB.
* Thumbnail: «генерация» фона (mock — процедурный арт по сиду момента; опционально
  OpenAI-совместимый /images/generations) + оверлей текста КОДОМ (Pillow) — текст никогда не
  доверяется генеративной модели, поэтому счёт на обложке не может «поплыть».
* Кэш thumbnail по (tenant, match_id, timestamp, ...): повторный килл не жжёт $.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.backends.hud import HEROES
from app.backends.stt import make_wav
from app.monitoring.metrics import THUMB_CACHE

TTS_CHUNK = 16000  # байт ≈ 0.5 с PCM16 @16 кГц


def synth_bumper(text: str, seconds: int = 8, tts=None, voice: str | None = None) -> bytes:
    """Ровно ``seconds`` секунд: начитка + тишина/обрезка. В ICMT — текст (удобно для round-trip теста)."""
    from app.backends.tts import MockTTS

    tts = tts or MockTTS()
    audio = tts.synth(text, seconds, voice)
    out = np.zeros(seconds * tts.sample_rate, dtype=np.float32)
    n = min(len(audio), len(out))
    out[:n] = audio[:n]
    return make_wav(out, tts.sample_rate, comment=text)


def stream_bytes(data: bytes, chunk: int = TTS_CHUNK) -> Iterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # старый Pillow
        return ImageFont.load_default()


def generate_background(prompt: str, size=(640, 360), hero: str | None = None) -> Image.Image:
    """Mock image-gen: детерминированный «арт» по сиду промпта."""
    if os.environ.get("KILLCAM_IMAGE_BACKEND") == "openai":
        return _openai_image(prompt, size)
    seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    w, h = size
    base = np.array(HEROES.get(hero or "", (200, 50, 50)), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w * 0.7 + yy / h * 0.3)[..., None]
    arr = (base * (1 - t) * 0.9 + np.array([10, 10, 25]) * t).clip(0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(14):  # «speed lines»
        x = int(rng.integers(-200, w))
        c = tuple(int(v) for v in np.clip(base + rng.integers(-40, 80, 3), 0, 255))
        d.line([(x, h), (x + 220, 0)], fill=c, width=int(rng.integers(2, 9)))
    return img


def _openai_image(prompt: str, size) -> Image.Image:
    import base64

    import httpx

    r = httpx.post(
        os.environ.get("KILLCAM_LLM_BASE_URL", "").rstrip("/") + "/images/generations",
        headers={"Authorization": f"Bearer {os.environ.get('KILLCAM_LLM_API_KEY', '')}"},
        json={"model": os.environ.get("KILLCAM_IMAGE_MODEL", "gpt-image-1"), "prompt": prompt, "size": "1536x1024"},
        timeout=120,
    )
    r.raise_for_status()
    img = Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB")
    return img.resize(size)


def render_thumbnail(match_id: str, timestamp: str, event: str = "kill", hero: str | None = None, score: str | None = None) -> bytes:
    prompt = f"esports {event} moment, hero {hero}, dramatic neon arena, no text"
    img = generate_background(f"{prompt}|{match_id}|{timestamp}", hero=hero)
    d = ImageDraw.Draw(img)
    label = {"kill": "KILL", "objective": "OBJECTIVE"}.get(event, "MOMENT")
    big, small = _font(64), _font(28)
    d.rectangle([0, 230, 640, 360], fill=(0, 0, 0))
    d.text((24, 244), f"{label} · {timestamp}", font=big, fill=(255, 255, 255))
    sub = " · ".join(x for x in [hero, f"score {score}" if score else None, match_id] if x)
    d.text((26, 318), sub, font=small, fill=(255, 200, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


class ThumbnailCache:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()

    def _path(self, tenant: str, key: tuple) -> Path:
        h = hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16]
        safe_tenant = "".join(c for c in tenant if c.isalnum() or c in "-_") or "default"
        return self.root / "thumbs" / safe_tenant / f"{h}.png"

    def get_or_render(self, tenant: str, match_id: str, timestamp: str, event: str, hero: str | None, score: str | None) -> tuple[bytes, bool]:
        p = self._path(tenant, (match_id, timestamp, event, hero, score))
        with self.lock:
            if p.exists():
                THUMB_CACHE.labels("hit").inc()
                return p.read_bytes(), True
        data = render_thumbnail(match_id, timestamp, event, hero, score)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        THUMB_CACHE.labels("miss").inc()
        return data, False
