"""Лёгкий Model Registry (файл в git, по семантике как MLflow: None → Staging → Production → Archived).

Версия = связка ``model_uri`` (LLM) + ``vlm`` (vlm_score_ocr) + ``prompt_version`` + ``data_hash``
(хэш lock-файла золотого набора/эталона UI). Продвижение в Production — только через eval-гейт
(см. ``eval/run_eval.py``), откат — одной командой, история переходов сохраняется.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

STAGES = ("None", "Staging", "Production", "Archived")


class RegistryError(Exception):
    pass


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- io
    def load(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------------- queries
    def versions(self) -> list[dict]:
        return self.load()["versions"]

    def get(self, version: int) -> dict:
        for v in self.versions():
            if v["version"] == version:
                return v
        raise RegistryError(f"version {version} not found")

    def by_stage(self, stage: str) -> dict | None:
        for v in self.versions():
            if v["stage"] == stage:
                return v
        return None

    def production(self) -> dict:
        v = self.by_stage("Production")
        if v is None:
            raise RegistryError("no Production version")
        return v

    # ---------------------------------------------------------------- mutations
    def register(self, model_uri: str, prompt_version: str, vlm: str, data_hash: str, notes: str = "") -> dict:
        with self._lock:
            data = self.load()
            new = {
                "version": max((v["version"] for v in data["versions"]), default=0) + 1,
                "model_uri": model_uri,
                "prompt_version": prompt_version,
                "vlm": vlm,
                "data_hash": data_hash,
                "stage": "Staging",
                "metrics": {},
                "notes": notes,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            data["versions"].append(new)
            data["history"].append({"ts": new["created_at"], "action": "register", "version": new["version"], "to": "Staging"})
            self.save(data)
            return new

    def transition(self, version: int, stage: str, metrics: dict | None = None, reason: str = "") -> dict:
        if stage not in STAGES:
            raise RegistryError(f"bad stage {stage}")
        with self._lock:
            data = self.load()
            target = next((v for v in data["versions"] if v["version"] == version), None)
            if target is None:
                raise RegistryError(f"version {version} not found")
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if stage == "Production":
                for v in data["versions"]:
                    if v["stage"] == "Production" and v["version"] != version:
                        v["stage"] = "Archived"
                        data["history"].append({"ts": ts, "action": "archive", "version": v["version"], "to": "Archived"})
            data["history"].append({"ts": ts, "action": "transition", "version": version, "from": target["stage"], "to": stage, "reason": reason})
            target["stage"] = stage
            if metrics:
                target["metrics"] = metrics
            self.save(data)
            return target

    def rollback(self) -> dict:
        """Вернуть в Production предыдущую версию, которая там была."""
        data = self.load()
        current = self.production()["version"]
        prod_history = [h["version"] for h in data["history"] if h.get("to") == "Production" and h["version"] != current]
        if not prod_history:
            raise RegistryError("nothing to roll back to")
        return self.transition(prod_history[-1], "Production", reason=f"rollback from v{current}")


def load_prompt(prompts_dir: Path, prompt_version: str) -> str:
    name, ver = prompt_version.split(":")
    return (Path(prompts_dir) / name / f"{ver}.md").read_text(encoding="utf-8")
