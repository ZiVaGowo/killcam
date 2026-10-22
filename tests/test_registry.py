"""MLOps: registry Staging/Production, гейт при промоуте, откат (демо-удар №4)."""

import json

from app.registry import Registry
from tests.conftest import OPERATOR, chat_body


def test_bad_prompt_blocked_by_gate_then_good_one_promoted_and_rolled_back(client, settings):
    reg = Registry(settings.registry_path)
    v2 = reg.register("mock://llm/card-v1", "caster:v2", "vlm_score_ocr:1.0@7.35", "sha256:test", "кастерский промпт")
    assert v2["stage"] == "Staging"

    r = client.post("/v1/registry/promote", json={"version": v2["version"]}, headers=OPERATOR)
    assert r.status_code == 409
    gate = r.json()["gate"]
    assert gate["gate"]["passed"] is False
    assert any("raw_score_halluc_rate" in x for x in gate["gate"]["reasons"])
    assert reg.production()["version"] == 1  # не поехало в Production

    v3 = reg.register("mock://llm/card-v1", "caster:v1", "vlm_score_ocr:1.0@7.35", "sha256:test", "переупаковка")
    r = client.post("/v1/registry/promote", json={"version": v3["version"]}, headers=OPERATOR)
    assert r.status_code == 200 and r.json()["promoted"]
    assert reg.production()["version"] == v3["version"]
    assert reg.get(1)["stage"] == "Archived"
    assert client.get("/health").json()["production"]["version"] == v3["version"]

    r = client.post("/v1/registry/rollback", headers=OPERATOR)
    assert r.status_code == 200 and r.json()["version"] == 1
    assert reg.production()["version"] == 1


def test_promote_requires_operator(client):
    assert client.post("/v1/registry/promote", json={"version": 1}).status_code == 403


def test_explicit_registry_version_in_request(client, settings):
    reg = Registry(settings.registry_path)
    v2 = reg.register("mock://llm/card-v1", "caster:v2", "vlm_score_ocr:1.0@7.35", "sha256:test")
    body = chat_body("evals/fixtures/screens/ok_03.png", audio="evals/fixtures/audio/a07.wav", registry_version=v2["version"])
    out = client.post("/v1/chat/completions", json=body).json()
    card = json.loads(out["choices"][0]["message"]["content"])
    # v2 «услышала» 32 и выдала 3:2, но guard вернул счёт с HUD и зафиксировал галлюцинацию
    assert card["score"] == "12:10"
    assert out["killcam"]["guard"]["hallucinated_score"] is True


def test_register_via_api(client):
    assert client.post("/v1/registry/register", json={"prompt_version": "caster:v2"}).status_code == 403
    assert "caster:v2" in client.get("/v1/registry/prompts").json()
    r = client.post("/v1/registry/register", json={"prompt_version": "caster:v2", "notes": "из UI"}, headers=OPERATOR)
    assert r.status_code == 200 and r.json()["stage"] == "Staging" and r.json()["version"] == 2
    assert client.post("/v1/registry/register", json={"prompt_version": "caster:v9"}, headers=OPERATOR).status_code == 422
