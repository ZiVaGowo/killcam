# Kill Cam — AI-продюсер киберспортивной трансляции

 Финал турнира, на экране килл: сервис за доли секунды делает **факт-карточку момента**
(JSON, стримом режиссёру), **8-секундную голосовую отбивку** и **обложку «KILL · 23:41»**.
Счёт не галлюцинируется, оскорбления из чата не попадают в эфир.

```
скрин кадра + голос кастера + (опц.) текст чата
        │
        ▼
 STT  +  VLM (счёт, герой, событие)  +  guard на чат/оверлеи (<data>)
        │
        ▼
 LLM: факт-карточка (JSON) ──по-полевой стрим + StreamGuard──► режиссёр
        │
        ├─ TTS: 8-секундная отбивка (стрим WAV)
        └─ Image gen: thumbnail + оверлей текста кодом (кэш по match_id+timestamp)
```

Всё запускается **на CPU без внешних API и ключей**: бэкенды по умолчанию — детерминированные
(`mock`), но с честной логикой (VLM реально читает пиксели HUD, guard'ы реально фильтруют).
Реальные модели подключаются переменными окружения (OpenAI-совместимый LLM/VLM, faster-whisper на GPU).

## Быстрый старт

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
```
```bash
pip install -r requirements-dev.txt
```
```bash
python scripts/make_fixtures.py --only vod     # VOD-кадры (DVC-стадия, в git не лежат)
```
```bash
pytest -q                                      # 102 теста
```
```bash
uvicorn app.main:app --port 8000               # API + веб-интерфейс http://127.0.0.1:8000/ui
```
```bash
python scripts/demo.py                         # демо 5 шагов (в другом терминале)
```

Docker (API + воркер + Redis + Prometheus + Grafana):

```bash
docker compose up --build
```

Веб-интерфейс — `:8000/ui`, API — `:8000`, Swagger — `:8000/docs`, Prometheus — `:9090`, Grafana — `:3000` (дашборд «Kill Cam»).
GPU-воркер: `docker compose --profile gpu up --build` (нужен nvidia-container-toolkit).

## Где что лежит — по обязательным темам

| Тема | Требование | В репо |
| --- | --- | --- |
| **1. MLOps** | CI на PR | [.github/workflows/ci.yml](.github/workflows/ci.yml): ruff, data-lock, pytest, eval-гейт, смоук uvicorn, docker build + смоук |
| | nightly / train-gate | [.github/workflows/nightly.yml](.github/workflows/nightly.yml): полный eval прода, гейт всех Staging-кандидатов, дрифт-отчёты |
| | DVC / lock данных | [dvc.yaml](dvc.yaml) (golden → vod → reference → lock → eval) + [data/data.lock.json](data/data.lock.json) (sha256 золотого набора и эталона UI, проверяется в CI) |
| | Model Registry (Staging/Production) | [registry/registry.json](registry/registry.json), [app/registry.py](app/registry.py), CLI [scripts/registry_cli.py](scripts/registry_cli.py); версия = `model_uri` + `prompt_version` + `vlm` + `data_hash` |
| **2. Serving** | FastAPI | [app/main.py](app/main.py) |
| | GPU или явный CPU-фолбэк | [app/config.py](app/config.py) `detect_device()`; `/health` показывает `device`; GPU-профиль в compose |
| | **стрим** | `POST /v1/chat/completions` `stream=true` — OpenAI-совместимый SSE, карточка стримится по полям через [StreamGuard](app/guards.py) |
| | **batch** | `POST /v1/batch` → очередь (Redis или in-memory) → [app/batch.py](app/batch.py), воркер [app/worker.py](app/worker.py); `GET /v1/batch/{id}/highlights.csv` |
| | media | `POST /v1/media/tts`, `POST /v1/media/thumbnail` — [app/media.py](app/media.py) |
| **3. Мониторинг** | drift | [app/monitoring/drift.py](app/monitoring/drift.py): эмбеддинги скринов vs эталон UI патча 7.35 (DVC) |
| | галлюцинации / faithfulness | `killcam_score_halluc_rate`, `killcam_faithfulness` — сравнение сырого ответа модели с фактами HUD |
| | latency TTFT / p95 | `killcam_ttft_seconds`, `killcam_latency_seconds`, `killcam_tts_ttfb_seconds` (TTFT отбивки) |
| | cost | [app/monitoring/cost.py](app/monitoring/cost.py), тарифы в [params.yaml](params.yaml); `$ / хайлайт` |
| | дашборд | `/dashboard` (HTML+JSON) и Grafana [monitoring/grafana](monitoring/grafana) |
| **4. Безопасность** | input / output guard | [app/guards.py](app/guards.py): `check_instruction`, `sanitize_untrusted`, `FieldGuard`, `StreamGuard` |
| | изоляция данных | `X-Tenant-ID`: батч-задачи, HITL-заявки, кэш обложек — чужой тенант получает 404; чат/OCR — только как `<data>` |
| | allow-list tools | [app/tools.py](app/tools.py): `make_thumbnail`, `tts_bumper`, `lookup_roster`, `publish_social`; остальное — 403 / вырезается из ответа |
| | HITL на публикации | `publish_social` → заявка в `/v1/hitl`, исполняется только по `approve` с `X-Operator-Key` |
| **5. Мультимодальность** | ≥2 входа | изображение (VLM) + аудио (STT) + текст чата |
| | исходящее медиа | TTS-отбивка **и** image gen (обложка) |

## Веб-интерфейс Kill Cam Studio (`/ui`)

| Вкладка | Что можно сделать |
| --- | --- |
| 🎯 Момент | выбрать кадр (или загрузить свой), голос кастера, чат и оверлей; пресеты «Килл», «Атака SYSTEM 9:0», «Jailbreak», «Нечитаемый HUD», «Чужая игра», «32 ≠ счёт», «После патча»; смотреть стрим по полям с таймингом, карточку, факты HUD/STT и события guard'а; из карточки — обложка, отбивка, публикация через HITL |
| 📼 Батч VOD | запустить запись матча, живой прогресс, таблица хайлайтов, $ / хайлайт, выгрузка CSV |
| 🎙 Медиа | TTS-отбивка с TTFB, обложка с индикатором кэша |
| 🛡 Безопасность | очередь HITL с кнопками одобрить/отклонить, вызов инструментов (в т.ч. запрещённых), прогон 10 red-team атак |
| 📦 Registry | регистрация версии, «В Production» через eval-гейт с разбором метрик, откат, история |
| 📈 Мониторинг | latency/TTFT/cost/галлюцинации/дрифт, графики |

Тенант и ключ оператора задаются в шапке. `/dashboard` открывает вкладку мониторинга.

## Автотесты качества

| Набор | Кол-во | Файл |
| --- | --- | --- |
| Золотые скрины: верный счёт / нечитаемый HUD / чужая игра | 15 | [evals/golden/screens.jsonl](evals/golden/screens.jsonl) |
| Аудио кастера: «три-два» vs «32» | 10 | [evals/golden/audio.jsonl](evals/golden/audio.jsonl) |
| Ядовитые сообщения чата | 8 | [evals/golden/chat.jsonl](evals/golden/chat.jsonl) |
| Red-team: инъекции (чат, оверлей), jailbreak, утечка system, tool вне allow-list, PII, публикация без HITL | 10 (×2) | [evals/golden/redteam.jsonl](evals/golden/redteam.jsonl) |
| Мультимодальные кейсы (фото + аудио + чат) | 5 | [evals/golden/multimodal.jsonl](evals/golden/multimodal.jsonl) |

Эталон — **поля / запреты / инструменты**, а не «красивый текст». Red-team прогоняется дважды:
на нормальной модели и на «скомпрометированной» (`NaiveLLM`, слушается инструкций из данных),
чтобы доказать, что guard'ы держат независимо от модели.

**Гейт** ([evals/gate.py](evals/gate.py)): кандидат не хуже прод-базлайна ([evals/baseline.json](evals/baseline.json))
больше чем на `δ = 0.02` ([params.yaml](params.yaml)) + жёсткие требования: red-team 100%, 0 утечек, схема 100%.

```bash
python -m evals.run_eval                       # Production vs baseline -> PASSED
```
```bash
python -m evals.run_eval --prompt caster:v2    # «кастерский» промпт -> FAILED (галлюцинации счёта 6% > δ)
```

## Демо (7–10 мин)

См. [docs/DEMO.md](docs/DEMO.md). Коротко: `python scripts/demo.py` проходит все 5 ударов —
стрим карточки, пойманная атака «SYSTEM: счёт 9:0», батч VOD с очередью и cost, график cost/latency,
красный гейт промпта v2 и откат версии в registry.

> Демо меняет `registry/registry.json` (регистрирует v2). Перед коммитом верните файл:
> `git checkout registry/registry.json`.

## Реальная речь: Whisper + Silero

По умолчанию речь — заглушки (CI и офлайн). Настоящие модели работают локально и бесплатно:

| Что | Модель | Переменная |
| --- | --- | --- |
| Слушать кастера (STT) | faster-whisper `medium` — на GPU float16, без CUDA сам уходит на CPU int8 | `KILLCAM_STT_BACKEND=whisper`, `KILLCAM_WHISPER_MODEL=small\|medium\|large-v3` |
| Говорить отбивку (TTS) | Silero TTS v4 ru, числа читаются словами («23:41» → «двадцать три сорок один») | `KILLCAM_TTS_BACKEND=silero`, `KILLCAM_TTS_SPEAKER=aidar\|baya\|kseniya\|xenia\|eugene` |

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```
```bash
pip install -r requirements-gpu.txt
```

Запуск с реальной речью (PowerShell):

```powershell
$env:KILLCAM_STT_BACKEND="whisper"; $env:KILLCAM_TTS_BACKEND="silero"; uvicorn app.main:app --port 8000
```

В интерфейсе: кнопка **Микрофон** на вкладке «Момент» (реплика кастера идёт в карточку) и карточка
**Распознавание речи** на вкладке «Медиа». Первый запрос загружает модели (Whisper medium ≈ 1.5 ГБ, Silero ≈ 60 МБ),
дальше — быстро. Сквозной тест настоящих моделей: `KILLCAM_REAL_MODELS=1 pytest tests/test_speech.py`.

## Реальные модели вместо mock

| Переменная | Значение |
| --- | --- |
| `KILLCAM_LLM_BACKEND=openai` + `KILLCAM_LLM_BASE_URL`, `KILLCAM_LLM_API_KEY` | LLM через любой OpenAI-совместимый API (vLLM, Ollama, OpenAI); версия в registry с `model_uri: openai://<model>` |
| `vlm: vlm_openai:<model>` в registry | VLM через OpenAI-совместимый API |
| `KILLCAM_IMAGE_BACKEND=openai` | фон обложки через `/images/generations` (текст всё равно рисуется кодом) |
| `KILLCAM_SHADOW_VLM=vlm_score_ocr:2.0@7.36` | shadow-режим новой VLM: ответы пишутся в `guard.shadow`, на прод не влияют |

## Stretch-цели

- **Детектор «это вообще килл?»** — в батче LLM вызывается только на килл/объект (`highlights_only`).
- **Кэш thumbnail по `(match_id, timestamp)`** — повторная обложка стоит $0 (`X-Cache: HIT`).
- **Shadow-режим новой VLM** — `KILLCAM_SHADOW_VLM`, VLM под патч 7.36 читает кадры, на которых прод-VLM молчит.

## Структура

```
app/            FastAPI, пайплайн, guards, tools/HITL, registry, batch, media, monitoring
app/backends/   hud (спецификация UI игры по патчам), vlm, stt, llm
evals/          золотой набор, фикстуры (png/wav), гейт, baseline
prompts/        версии промпта «кастер» (v1 — прод, v2 — регрессия)
registry/       Model Registry
scripts/        фикстуры, эталон UI, data lock, registry CLI, smoke, drift report, demo
monitoring/     Prometheus + Grafana provisioning
tests/          102 теста: схема, смоук, стрим, батч, медиа, guards, HITL, registry, eval, дрифт, мультимодальность
```
