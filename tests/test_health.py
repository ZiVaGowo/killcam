"""Смоук: /health, /metrics, дашборд."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] in ("cpu", "cuda")
    assert body["production"]["version"] == 1
    assert body["production"]["prompt_version"] == "caster:v1"
    assert body["production"]["data_hash"].startswith("sha256:")


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "killcam_ttft_seconds" in r.text
    assert "killcam_cost_usd_total" in r.text
    assert "killcam_drift_score" in r.text


def test_ui_and_dashboard(client):
    assert "Kill Cam Studio" in client.get("/ui").text
    assert "Kill Cam Studio" in client.get("/").text            # редирект на /ui
    assert "Kill Cam Studio" in client.get("/dashboard").text   # редирект на /ui#monitor
    d = client.get("/v1/dashboard.json").json()
    for key in ("latency_p95", "ttft_p95", "score_halluc_rate", "cost_per_highlight_usd", "drift_status"):
        assert key in d


def test_bad_tenant_rejected(client):
    assert client.get("/v1/hitl", headers={"X-Tenant-ID": "../etc"}).status_code == 400


def test_demo_fixtures(client):
    d = client.get("/v1/demo/fixtures").json()
    assert len(d["screens"]) >= 15 and len(d["audio"]) >= 10 and len(d["redteam"]) == 10
    r = client.get("/v1/demo/file", params={"path": d["screens"][0]["path"]})
    assert r.status_code == 200 and r.content[1:4] == b"PNG"


def test_demo_file_no_traversal(client):
    for bad in ("registry/registry.json", "../../etc/passwd", "evals/fixtures/../../params.yaml", "app/main.py"):
        assert client.get("/v1/demo/file", params={"path": bad}).status_code == 404


def test_demo_vod_batch_rejects_outside_folder(client):
    assert client.post("/v1/demo/vod-batch", json={"folder": "evals/fixtures/screens"}).status_code == 404
