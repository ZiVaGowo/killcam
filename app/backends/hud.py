"""Спецификация HUD игры «Arena Clash» по патчам + рендер синтетических кадров.

Один и тот же модуль используется:
  * генератором фикстур (scripts/make_fixtures.py) — рисует кадры;
  * VLM-бэкендом (vlm_score_ocr) — знает, где на кадре счёт/таймер/килфид.

Патч 7.36 двигает HUD и меняет палитру — это и есть visual/concept drift:
VLM, настроенный на 7.35, перестаёт читать счёт, а дрифт-монитор это ловит.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo

W, H = 320, 180

# 5x7 bitmap-шрифт для цифр HUD. Свой шрифт => рендер и OCR не зависят от версии Pillow.
GLYPHS: dict[str, list[str]] = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
}
GLYPH_ARR = {c: np.array([[int(b) for b in row] for row in rows], dtype=np.uint8) for c, rows in GLYPHS.items()}

SCALE = 2
PITCH = (5 + 1) * SCALE  # ширина ячейки символа с пробелом

HEROES: dict[str, tuple[int, int, int]] = {
    "Vex": (200, 60, 200),
    "Rook": (60, 160, 220),
    "Nyra": (240, 120, 160),
    "Kade": (80, 200, 120),
    "Sol": (250, 130, 40),
    "Mira": (150, 110, 60),
}
EVENTS: dict[str, tuple[int, int, int]] = {
    "kill": (220, 40, 40),
    "objective": (230, 200, 40),
    "none": (120, 120, 120),
}
GAME_MARKERS: dict[str, list[tuple[int, int, int]]] = {
    "arena_clash": [(255, 140, 0), (255, 255, 255), (255, 140, 0)],
    "hex_legends": [(0, 200, 80), (0, 90, 40), (0, 200, 80)],  # «чужая игра»
}


@dataclass(frozen=True)
class HudLayout:
    patch: str
    bar_box: tuple[int, int, int, int]  # x0, y0, x1, y1
    bar_color: tuple[int, int, int]
    marker_xy: tuple[int, int]
    score_xy: tuple[int, int]
    timer_xy: tuple[int, int]
    event_box: tuple[int, int, int, int]
    hero_box: tuple[int, int, int, int]


LAYOUTS: dict[str, HudLayout] = {
    "7.35": HudLayout(
        patch="7.35",
        bar_box=(0, 0, W, 22),
        bar_color=(18, 26, 52),
        marker_xy=(4, 5),
        score_xy=(140, 4),
        timer_xy=(250, 4),
        event_box=(8, 34, 24, 50),
        hero_box=(30, 34, 46, 50),
    ),
    # Патч: HUD уехал вниз, бар стал фиолетовым, килфид справа.
    "7.36": HudLayout(
        patch="7.36",
        bar_box=(0, H - 22, W, H),
        bar_color=(70, 20, 90),
        marker_xy=(4, H - 17),
        score_xy=(40, H - 18),
        timer_xy=(140, H - 18),
        event_box=(W - 46, 34, W - 30, 50),
        hero_box=(W - 24, 34, W - 8, 50),
    ),
}


@dataclass
class FrameSpec:
    """Описание кадра для генератора фикстур."""

    score: str = "3:2"
    timer: str = "23:41"
    hero: str = "Vex"
    event: str = "kill"
    game: str = "arena_clash"
    patch: str = "7.35"
    unreadable: bool = False      # нечитаемый HUD (блюр/шум/перекрытие)
    fake_overlay: str | None = None  # оверлей стримера с фейковым счётом/текстом
    seed: int = 0


def draw_text(img: Image.Image, text: str, xy: tuple[int, int], color=(255, 255, 255), scale: int = SCALE) -> None:
    px = img.load()
    x0, y0 = xy
    for i, ch in enumerate(text):
        g = GLYPH_ARR.get(ch)
        if g is None:
            continue
        cx = x0 + i * (5 + 1) * scale
        for gy in range(7):
            for gx in range(5):
                if g[gy, gx]:
                    for dy in range(scale):
                        for dx in range(scale):
                            x, y = cx + gx * scale + dx, y0 + gy * scale + dy
                            if 0 <= x < img.width and 0 <= y < img.height:
                                px[x, y] = color


def _background(seed: int, game: str) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array([40, 70, 50] if game == "arena_clash" else [70, 50, 40], dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = (xx / W)[..., None] * rng.uniform(10, 40, 3) + (yy / H)[..., None] * rng.uniform(10, 40, 3)
    arr = np.clip(base + grad, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(6):  # «арена»: пара силуэтов/стен
        x, y = int(rng.integers(0, W - 40)), int(rng.integers(30, H - 40))
        w, h = int(rng.integers(15, 60)), int(rng.integers(10, 40))
        c = tuple(int(v) for v in rng.integers(30, 110, 3))
        d.rectangle([x, y, x + w, y + h], fill=c)
    return img


def render_frame(spec: FrameSpec) -> Image.Image:
    lay = LAYOUTS[spec.patch]
    img = _background(spec.seed, spec.game)
    d = ImageDraw.Draw(img)
    d.rectangle(lay.bar_box, fill=lay.bar_color)
    mx, my = lay.marker_xy
    for i, c in enumerate(GAME_MARKERS[spec.game]):
        d.rectangle([mx + i * 7, my, mx + i * 7 + 5, my + 11], fill=c)
    draw_text(img, spec.score, lay.score_xy)
    draw_text(img, spec.timer, lay.timer_xy)
    d.rectangle(lay.event_box, fill=EVENTS[spec.event])
    d.rectangle(lay.hero_box, fill=HEROES[spec.hero])
    if spec.unreadable:
        # Смазываем зону счёта/таймера (маркер игры остаётся читаемым)
        _, y0, x1, y1 = lay.bar_box
        x0 = min(lay.score_xy[0], lay.timer_xy[0]) - 6
        region = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(3))
        rng = np.random.default_rng(spec.seed + 7)
        noisy = np.clip(np.asarray(region, dtype=np.int16) + rng.integers(-60, 60, (y1 - y0, x1 - x0, 3)), 0, 255)
        img.paste(Image.fromarray(noisy.astype(np.uint8)), (x0, y0))
    if spec.fake_overlay:
        # Оверлей стримера посреди кадра (не в зоне HUD): фейковый "9:0"
        d.rectangle([100, 80, 220, 120], fill=(10, 10, 10))
        digits = "".join(ch for ch in spec.fake_overlay if ch in GLYPHS)[:5]
        draw_text(img, digits, (118, 88), color=(255, 255, 255), scale=3)
    return img


def to_png_bytes(img: Image.Image, overlay_text: str | None = None) -> bytes:
    """PNG с tEXt-чанком overlay_text — симуляция OCR текста оверлеев стрима."""
    info = PngInfo()
    if overlay_text:
        info.add_text("overlay_text", overlay_text)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info, optimize=True)
    return buf.getvalue()
