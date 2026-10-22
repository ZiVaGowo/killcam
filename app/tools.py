"""Allow-list инструментов + Human-in-the-loop для публикации.

``publish_social`` никогда не исполняется автоматически: вызов создаёт заявку в HITL-очереди,
публикация происходит только после ``approve`` оператором с ключом ``X-Operator-Key``.
Всё, чего нет в allow-list (send_dm, get_player_phone, ...), отклоняется.
Заявки изолированы по тенанту.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ALLOW_LIST: dict[str, dict[str, Any]] = {
    "make_thumbnail": {"hitl": False, "desc": "обложка KILL · mm:ss"},
    "tts_bumper": {"hitl": False, "desc": "8-секундная голосовая отбивка"},
    "lookup_roster": {"hitl": False, "desc": "публичный состав команд"},
    "publish_social": {"hitl": True, "desc": "публикация в соцсети — только после одобрения человеком"},
}

ROSTER = {"Vex": "Team Nova", "Rook": "Team Nova", "Nyra": "Iron Owls", "Kade": "Iron Owls", "Sol": "Team Nova", "Mira": "Iron Owls"}


class ToolDenied(Exception):
    pass


def filter_actions(actions: list[Any]) -> tuple[list[dict], list[dict]]:
    allowed, denied = [], []
    for a in actions:
        if not isinstance(a, dict) or "tool" not in a:
            continue
        name = a["tool"]
        if name in ALLOW_LIST:
            allowed.append({"tool": name, "args": a.get("args") or {}, "requires_hitl": ALLOW_LIST[name]["hitl"]})
        else:
            denied.append(a)
    return allowed, denied


class HITLQueue:
    def __init__(self, path: Path):
        self.path = path
        self.published_path = path.parent / "published.jsonl"
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        if path.exists():
            try:
                self._items = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._items = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, ensure_ascii=False, indent=1), encoding="utf-8")

    def request(self, tenant: str, tool: str, args: dict) -> dict:
        with self._lock:
            rid = uuid.uuid4().hex[:12]
            item = {"id": rid, "tenant": tenant, "tool": tool, "args": args, "status": "pending", "created_at": time.time()}
            self._items[rid] = item
            self._save()
            return item

    def list(self, tenant: str, status: str | None = "pending") -> list[dict]:
        return [i for i in self._items.values() if i["tenant"] == tenant and (status is None or i["status"] == status)]

    def get(self, tenant: str, rid: str) -> dict | None:
        item = self._items.get(rid)
        return item if item and item["tenant"] == tenant else None  # изоляция: чужие заявки не видны

    def decide(self, tenant: str, rid: str, approve: bool, operator: str) -> dict:
        with self._lock:
            item = self.get(tenant, rid)
            if item is None:
                raise KeyError(rid)
            if item["status"] != "pending":
                return item
            item["status"] = "approved" if approve else "rejected"
            item["decided_by"] = operator
            item["decided_at"] = time.time()
            if approve:
                item["result"] = execute_tool(item["tool"], item["args"], published_path=self.published_path, tenant=tenant)
            self._save()
            return item


def execute_tool(name: str, args: dict, published_path: Path | None = None, tenant: str = "default") -> dict:
    if name not in ALLOW_LIST:
        raise ToolDenied(name)
    if name == "lookup_roster":
        hero = args.get("hero")
        return {"hero": hero, "team": ROSTER.get(hero or "", "unknown")}
    if name == "publish_social":
        post = {"tenant": tenant, "text": str(args.get("text", ""))[:280], "ts": time.time(), "post_id": uuid.uuid4().hex[:8]}
        if published_path:
            published_path.parent.mkdir(parents=True, exist_ok=True)
            with published_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
        return {"published": True, "post_id": post["post_id"]}
    return {"ok": True, "note": f"{name}: используйте /v1/media/*"}
