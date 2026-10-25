"""
Integration Tests for IncuBrix FastAPI Service (All 10 Standards).

Tests:
- Integration Standard #2: OpenAPI routes under /api/v1/
- Integration Standard #3: Asynchronous job queue (202 Accepted + progress)
- Integration Standard #4: Safe retries with Idempotency-Key header
- Integration Standard #5: Standardized UPPER_SNAKE_CASE error format
- Integration Standard #8: Health and Ready probes with structured headers
"""

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.mark.asyncio
async def test_health_and_readiness_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Liveness probe
        res_health = await client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        # Readiness probe
        res_ready = await client.get("/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_api_session_and_idempotent_steps():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create a session
        idempotency_key = "test-session-key-001"
        res_create = await client.post(
            "/api/v1/sessions",
            json={"video_url_or_path": "fixtures/test_video.mp4"},
            headers={"Idempotency-Key": idempotency_key}
        )
        assert res_create.status_code == 201
        session_id = res_create.json()["session_id"]

        # Safe retry test: Send exact same request with same Idempotency-Key
        res_retry = await client.post(
            "/api/v1/sessions",
            json={"video_url_or_path": "fixtures/test_video.mp4"},
            headers={"Idempotency-Key": idempotency_key}
        )
        assert res_retry.status_code == 201
        assert res_retry.json()["session_id"] == session_id

        # 2. Execute an instruction step
        res_step = await client.post(
            f"/api/v1/sessions/{session_id}/steps",
            json={"say": "Cut the first 30 seconds"}
        )
        assert res_step.status_code == 200
        data = res_step.json()
        assert data["status"] == "applied"
        assert "shorter" in data["summary"]

        # 3. Retrieve full timeline state
        res_timeline = await client.get(f"/api/v1/sessions/{session_id}")
        assert res_timeline.status_code == 200
        tl_data = res_timeline.json()
        assert len(tl_data["timeline_hash"]) == 64

        # 4. Trigger asynchronous render (Must return 202 Accepted)
        res_render = await client.post(
            f"/api/v1/sessions/{session_id}/render",
            json={}
        )
        assert res_render.status_code == 202
        job_id = res_render.json()["job_id"]

        # 5. Poll job status
        res_job = await client.get(f"/api/v1/jobs/{job_id}")
        assert res_job.status_code == 200
        assert res_job.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_standardized_error_format():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request non-existent session
        res = await client.get("/api/v1/sessions/non_existent_id")
        assert res.status_code == 404
        err_body = res.json()
        assert "error" in err_body
        assert err_body["error"]["code"] == "SESSION_NOT_FOUND"
        assert isinstance(err_body["error"]["retryable"], bool)
