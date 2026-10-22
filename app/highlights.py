"""Движок хайлайтов: следит за матчем во времени и находит моменты, которые стоит показать и озвучить.

Один кадр ничего не знает о прошлом. Движок получает факты с каждого кадра (таймер, лента убийств,
башни) и ведёт состояние матча:

    лента убийств  ──► новые убийства ──► первая кровь · дабл/трипл · серия · прерванная серия
                                         · эйс (вся команда соперника) · «в одиночку забрал всех»
    башни          ──► снесённые башни ──► быстрая первая башня · снос линии целиком за N секунд
                                         · новый рекорд турнира

У каждого момента есть «эпичность» (hype 0–110): голосом объявляются только моменты выше порога.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.backends.hud import TEAM_OF, TEAMS
from app.backends.tts import number_ru

HERO_RU = {"Vex": "Векс", "Rook": "Рук", "Sol": "Сол", "Nyra": "Найра", "Kade": "Кейд", "Mira": "Мира"}
HERO_RU_GEN = {"Vex": "Векса", "Rook": "Рука", "Sol": "Сола", "Nyra": "Найры", "Kade": "Кейда", "Mira": "Миры"}
TEAM_TITLE = {"nova": "Team Nova", "owls": "Iron Owls"}
TEAM_RU = {"nova": "Нова", "owls": "Совы"}
LANE_ACC = {"top": "верхнюю", "mid": "центральную", "bot": "нижнюю"}
LANE_PREP = {"top": "верхней", "mid": "центральной", "bot": "нижней"}
LANE_TITLE = {"top": "верхняя", "mid": "центральная", "bot": "нижняя"}
ORDINAL_PREP = ["первой", "второй", "третьей", "четвёртой", "пятой", "шестой", "седьмой", "восьмой", "девятой", "десятой"]
TIMER_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def _fem(words: str) -> str:
    """«один»/«два» в конце числа -> «одну»/«две» (минуту, секунды)."""
    return re.sub(r"(^|\s)один$", r"\1одну", re.sub(r"(^|\s)два$", r"\1две", words))


def duration_ru(seconds: float) -> str:
    """75 -> «одну минуту пятнадцать секунд» — для озвучки, без цифр."""
    m, s = divmod(int(round(seconds)), 60)
    parts = []
    if m:
        parts.append(f"{_fem(number_ru(m))} {_plural(m, 'минуту', 'минуты', 'минут')}")
    if s or not m:
        parts.append(f"{_fem(number_ru(s))} {_plural(s, 'секунду', 'секунды', 'секунд')}")
    return " ".join(parts)


def clock(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def parse_clock(timer: str | None) -> int | None:
    m = TIMER_RE.match(timer or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


@dataclass
class Highlight:
    id: str
    type: str
    title: str            # для экрана
    announce: str         # для голоса (без цифр и латиницы)
    hype: int
    game_time: float
    clock: str
    frame: int | None
    players: list[str] = field(default_factory=list)
    team: str | None = None
    details: dict = field(default_factory=dict)
    record: bool = False
    should_announce: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class RecordStore:
    """Рекорды турнира. В проде — общий файл/БД; для eval — в памяти, чтобы прогон был детерминированным."""

    def __init__(self, initial: dict, path: Path | None = None):
        self.path = path
        self.records = dict(initial)
        self.lock = threading.Lock()
        if path and path.exists():
            try:
                self.records.update(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass

    def beat(self, key: str, value: float) -> bool:
        """True, если value — новый рекорд (меньше — лучше)."""
        with self.lock:
            if value < self.records.get(key, float("inf")):
                self.records[key] = value
                if self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self.path.write_text(json.dumps(self.records), encoding="utf-8")
                return True
            return False


class MatchEngine:
    def __init__(self, match_id: str, cfg: dict, records: RecordStore):
        self.match_id = match_id
        self.cfg = cfg
        self.records = records
        self.hype = cfg["hype"]
        self.last_time: float | None = None
        self.step = 2.0
        self.prev_feed: set[tuple[str, str]] = set()
        self.kills: list[tuple[float, str, str]] = []
        self.streak: dict[str, int] = {}
        self.chain: dict[str, list[float]] = {}
        self.prev_towers: dict | None = None
        self.lane_first_down: dict[tuple[str, str], float] = {}
        self.first_tower_done = False
        self.last_ace: dict[str, float] = {}
        self.highlights: list[Highlight] = []
        self.frames = 0
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- helpers
    def _add(self, out: list[Highlight], type_: str, title: str, announce: str, t: float, frame: int | None,
             players: list[str], team: str | None = None, details: dict | None = None, record: bool = False) -> None:
        hype = self.hype[type_] + (self.hype["record_bonus"] if record else 0)
        h = Highlight(
            id=f"{self.match_id}-{next(self._ids)}", type=type_, title=title, announce=announce, hype=hype,
            game_time=t, clock=clock(t), frame=frame, players=players, team=team, details=details or {},
            record=record, should_announce=hype >= self.cfg["announce_threshold"],
        )
        out.append(h)
        self.highlights.append(h)

    def _time(self, timer: str | None) -> float:
        t = parse_clock(timer)
        if t is None:  # таймер не читается — интерполируем по шагу кадров
            return (self.last_time or 0) + self.step
        if self.last_time is not None and t > self.last_time:
            self.step = min(10.0, t - self.last_time)
        return float(t)

    # ---------------------------------------------------------------- main
    def ingest(self, facts: dict, frame: int | None = None) -> list[Highlight]:
        if facts.get("game") != "arena_clash" or facts.get("kill_feed") is None:
            return []
        t = self._time(facts.get("timer"))
        if self.last_time is not None and t < self.last_time:
            return []  # кадр из прошлого (перемотка) — состояние не трогаем
        self.last_time = t
        self.frames += 1
        out: list[Highlight] = []

        feed = [tuple(r) for r in facts["kill_feed"]]
        new_kills = [k for k in feed if k not in self.prev_feed]
        self.prev_feed = set(feed)
        for killer, victim in reversed(new_kills):  # в ленте свежие сверху — обрабатываем по порядку
            self._on_kill(out, t, frame, killer, victim)

        towers = facts.get("towers")
        if towers:
            if self.prev_towers:
                self._on_towers(out, t, frame, towers)
            self.prev_towers = towers
        self._merge_burst(out)
        return out

    @staticmethod
    def _merge_burst(out: list[Highlight]) -> None:
        """Несколько моментов в одном кадре (трипл + эйс + серия…) — комментатор говорит одну фразу:
        озвучиваем самый эпичный, остальные прикрепляем к нему тегами."""
        loud = [h for h in out if h.should_announce]
        if len(loud) < 2:
            return
        top = max(loud, key=lambda h: h.hype)
        for h in loud:
            if h is not top:
                h.should_announce = False
                h.details["merged_into"] = top.id
                top.details.setdefault("also", []).append(h.type)

    def _on_kill(self, out: list[Highlight], t: float, frame: int | None, killer: str, victim: str) -> None:
        c = self.cfg
        team, enemy = TEAM_OF.get(killer), TEAM_OF.get(victim)
        if team is None or enemy is None or team == enemy:
            return
        k_ru, v_gen = HERO_RU[killer], HERO_RU_GEN[victim]
        if not self.kills:
            self._add(out, "first_blood", f"Первая кровь: {killer} → {victim}",
                      f"Первая кровь! {k_ru} открывает счёт!", t, frame, [killer, victim], team)
        self.kills.append((t, killer, victim))

        if self.streak.get(victim, 0) >= c["spree_kills"]:
            self._add(out, "shutdown", f"{killer} прерывает серию {victim} ({self.streak[victim]} убийства)",
                      f"{k_ru} прерывает серию {v_gen}!", t, frame, [killer, victim], team, {"ended_streak": self.streak[victim]})
        self.streak[victim] = 0
        self.chain.pop(victim, None)

        self.streak[killer] = self.streak.get(killer, 0) + 1
        if self.streak[killer] == c["spree_kills"]:
            self._add(out, "killing_spree", f"{killer}: серия из {self.streak[killer]} убийств",
                      f"{k_ru} в ударе! {number_ru(self.streak[killer]).capitalize()} "
                      f"{_plural(self.streak[killer], 'убийство', 'убийства', 'убийств')} подряд!",
                      t, frame, [killer], team, {"streak": self.streak[killer]})

        chain = self.chain.get(killer, [])
        if chain and t - chain[-1] > c["chain_window_s"]:  # пауза больше окна — серия закончилась
            chain = []
        chain = chain + [t]
        self.chain[killer] = chain
        if len(chain) == 2:
            self._add(out, "double_kill", f"Двойное убийство: {killer}", f"Двойное убийство! {k_ru}!", t, frame, [killer], team)
        elif len(chain) == 3:
            self._add(out, "triple_kill", f"Тройное убийство: {killer}", f"Тройное убийство! {k_ru} неудержим!", t, frame, [killer], team)

        enemies = TEAMS[enemy]
        recent = [(kt, kk, kv) for kt, kk, kv in self.kills if t - kt <= c["ace_window_s"] and TEAM_OF[kk] == team]
        dead = {kv for _, _, kv in recent}
        if set(enemies) <= dead and t - self.last_ace.get(team, -1e9) > c["ace_window_s"]:
            self.last_ace[team] = t
            start = min(kt for kt, _, kv in recent if kv in enemies)
            self._add(out, "ace", f"Эйс! {TEAM_TITLE[team]} вырезает всю команду {TEAM_TITLE[enemy]}",
                      f"Эйс! {TEAM_RU[team]} не оставляет соперникам никого!", t, frame,
                      sorted({kk for _, kk, kv in recent if kv in enemies}), team, {"span_s": t - start})
            solo = [(kt, kv) for kt, kk, kv in recent if kk == killer and kv in enemies and t - kt <= c["one_man_window_s"]]
            if {kv for _, kv in solo} >= set(enemies):
                span = t - min(kt for kt, _ in solo)
                self._add(out, "one_man_wipe", f"{killer} в одиночку забирает всю команду за {span:.0f} с",
                          f"Невероятно! {k_ru} в одиночку забирает всю команду соперника за {duration_ru(max(span, 1))}!",
                          t, frame, [killer], team, {"span_s": span})

    def _on_towers(self, out: list[Highlight], t: float, frame: int | None, towers: dict) -> None:
        c = self.cfg
        for owner, lanes in towers.items():
            attacker = next(x for x in TEAMS if x != owner)
            for lane, alive in lanes.items():
                before = self.prev_towers.get(owner, {}).get(lane, alive)
                if alive >= before:
                    continue
                self.lane_first_down.setdefault((owner, lane), t)
                if not self.first_tower_done:
                    self.first_tower_done = True
                    if t <= c["fast_tower_s"]:
                        rec = self.records.beat("first_tower_s", t)
                        minute = ORDINAL_PREP[min(int(t // 60), 9)]
                        self._add(out, "fast_tower",
                                  f"{TEAM_TITLE[attacker]}: первая башня уже на {clock(t)} ({LANE_TITLE[lane]} линия)",
                                  f"{TEAM_RU[attacker]} сносит башню на {LANE_PREP[lane]} линии уже на {minute} минуте!"
                                  + (" Новый рекорд турнира!" if rec else ""),
                                  t, frame, [], attacker, {"lane": lane}, record=rec)
                if alive == 0:
                    dur = t - self.lane_first_down[(owner, lane)]
                    if 0 < dur <= c["lane_rush_s"]:
                        rec = self.records.beat("lane_rush_s", dur)
                        self._add(out, "lane_rush",
                                  f"{TEAM_TITLE[attacker]} сносит {LANE_ACC[lane]} линию за {dur:.0f} с",
                                  f"{TEAM_RU[attacker]} сносит {LANE_ACC[lane]} линию целиком всего за {duration_ru(dur)}!"
                                  + (" Новый рекорд турнира!" if rec else ""),
                                  t, frame, [], attacker, {"lane": lane, "duration_s": dur}, record=rec)

    def summary(self) -> dict:
        return {
            "match_id": self.match_id,
            "frames": self.frames,
            "game_time": self.last_time,
            "kills": len(self.kills),
            "highlights": [h.to_dict() for h in self.highlights],
            "top": [h.to_dict() for h in sorted(self.highlights, key=lambda h: -h.hype)[:5]],
        }


class MatchHub:
    """Состояние матчей по (tenant, match_id) — изоляция тенантов как у батча и HITL."""

    def __init__(self, cfg: dict, records: RecordStore):
        self.cfg, self.records = cfg, records
        self.matches: dict[tuple[str, str], MatchEngine] = {}
        self.audio: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def get(self, tenant: str, match_id: str) -> MatchEngine:
        with self.lock:
            key = (tenant, match_id)
            if key not in self.matches:
                self.matches[key] = MatchEngine(match_id, self.cfg, self.records)
            return self.matches[key]

    def store_audio(self, key: str, wav: bytes, limit: int = 200) -> None:
        with self.lock:
            self.audio[key] = wav
            while len(self.audio) > limit:  # память не растёт бесконечно
                self.audio.pop(next(iter(self.audio)))

    def reset(self, tenant: str, match_id: str) -> None:
        with self.lock:
            self.matches.pop((tenant, match_id), None)
