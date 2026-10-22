"""Pydantic-схемы: карточка момента + OpenAI-совместимые запросы + batch/media/tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ALLOWED_EVENTS = ("kill", "objective", "none")


class SuggestedAction(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    requires_hitl: bool = False


class MomentCard(BaseModel):
    """Факт-карточка момента. Эталон проверяется по полям, а не «по красоте»."""

    match_id: str
    timestamp: str | None = Field(default=None, pattern=r"^\d{1,2}:\d{2}$")
    event: Literal["kill", "objective", "none"]
    hero: str | None = None
    score: str | None = Field(default=None, pattern=r"^\d{1,2}:\d{1,2}$")
    headline: str = Field(max_length=120)
    caption: str = Field(max_length=400)
    hashtags: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0, le=1)
    needs_review: bool
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)

    @field_validator("hashtags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.startswith("#") or " " in t:
                raise ValueError(f"bad hashtag {t!r}")
        return v


# ----------------------------------------------------------- OpenAI-compatible chat

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str = "killcam-card"
    messages: list[ChatMessage]
    stream: bool = False
    # расширения Kill Cam
    match_id: str = "unknown"
    chat: list[str] = Field(default_factory=list, description="сообщения чата трансляции (недоверенные)")
    overlay_text: str | None = Field(default=None, description="OCR оверлеев стрима (недоверенные)")
    registry_version: int | None = Field(default=None, description="явная версия из registry (по умолчанию Production)")


# ----------------------------------------------------------- batch

class BatchItem(BaseModel):
    image_b64: str
    timestamp_hint: str | None = None
    audio_b64: str | None = None


class BatchRequest(BaseModel):
    match_id: str
    items: list[BatchItem] = Field(min_length=1, max_length=500)
    make_thumbnails: bool = False


class BatchJobStatus(BaseModel):
    job_id: str
    tenant: str
    status: Literal["queued", "running", "done", "failed"]
    total: int
    processed: int = 0
    highlights: list[dict[str, Any]] = Field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None


# ----------------------------------------------------------- media / tools

class TTSRequest(BaseModel):
    text: str = Field(max_length=300)
    voice: str = Field(default="", max_length=40, description='модель/голос, напр. "v5_6_ru/romchik"; пусто — по умолчанию')


class ThumbnailRequest(BaseModel):
    match_id: str
    timestamp: str = Field(pattern=r"^\d{1,2}:\d{2}$")
    event: Literal["kill", "objective", "none"] = "kill"
    hero: str | None = None
    score: str | None = Field(default=None, pattern=r"^\d{1,2}:\d{1,2}$")


class ToolCallRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class PromoteRequest(BaseModel):
    version: int
    stage: Literal["Staging", "Production", "Archived"] = "Production"
    skip_gate: bool = False


class RegisterRequest(BaseModel):
    prompt_version: str
    model_uri: str | None = None
    vlm: str | None = None
    notes: str = ""


class MatchFrameRequest(BaseModel):
    image_b64: str
    frame: int | None = None
    announce: bool = True          # озвучивать моменты с hype ≥ порога
    voice: str = Field(default="", max_length=40)
