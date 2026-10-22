"""Allow-list инструментов, HITL на публикации, изоляция тенантов."""

from tests.conftest import OPERATOR

A = {"X-Tenant-ID": "studio_a"}
B = {"X-Tenant-ID": "studio_b"}


def test_tool_outside_allow_list_denied(client):
    for tool in ("send_dm", "get_player_phone", "exec_shell"):
        assert client.post("/v1/tools/call", json={"tool": tool}, headers=A).status_code == 403


def test_safe_tool_runs(client):
    r = client.post("/v1/tools/call", json={"tool": "lookup_roster", "args": {"hero": "Vex"}}, headers=A)
    assert r.status_code == 200 and r.json()["result"]["team"] == "Team Nova"


def test_publish_requires_human(client, settings):
    r = client.post("/v1/tools/call", json={"tool": "publish_social", "args": {"text": "Vex забирает килл!"}}, headers=A)
    assert r.status_code == 202 and r.json()["status"] == "pending_approval"
    rid = r.json()["hitl_id"]
    published = settings.runtime_dir / "published.jsonl"
    assert not published.exists()  # ничего не опубликовано без человека

    assert client.post(f"/v1/hitl/{rid}/approve", headers=A).status_code == 403  # нет ключа оператора
    assert client.post(f"/v1/hitl/{rid}/approve", headers={**B, **OPERATOR}).status_code == 404  # чужой тенант
    assert client.get("/v1/hitl", headers=B).json() == []

    ok = client.post(f"/v1/hitl/{rid}/approve", headers={**A, **OPERATOR})
    assert ok.status_code == 200 and ok.json()["status"] == "approved" and ok.json()["result"]["published"]
    assert "Vex забирает килл!" in published.read_text(encoding="utf-8")


def test_reject(client):
    rid = client.post("/v1/tools/call", json={"tool": "publish_social", "args": {"text": "x"}}, headers=A).json()["hitl_id"]
    r = client.post(f"/v1/hitl/{rid}/reject", headers={**A, **OPERATOR})
    assert r.json()["status"] == "rejected" and "result" not in r.json()
