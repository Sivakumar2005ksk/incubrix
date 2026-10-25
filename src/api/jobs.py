"""
Asynchronous Job Execution & Webhook Dispatcher (Integration Standard #3).

Handles long-running tasks (> 2 seconds) asynchronously:
- Generates job_id and returns HTTP 202 Accepted.
- Tracks status (queued, running, succeeded, failed) and progress (0-100%).
- Emits signed HMAC-SHA256 webhooks to callback_url via 'X-IncuBrix-Signature' header.
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field

from src.config import settings


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100, default=0)
    created_at: float
    completed_at: Optional[float] = None
    result_links: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    callback_url: Optional[str] = None


class JobManager:
    """Manages async job state and background execution."""
    def __init__(self):
        self._jobs: Dict[str, JobResult] = {}

    def create_job(self, callback_url: Optional[str] = None) -> JobResult:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = JobResult(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress=0,
            created_at=time.time(),
            callback_url=callback_url
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[JobResult]:
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        progress: int,
        result_links: Optional[Dict[str, str]] = None,
        error: Optional[str] = None
    ) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = status
        job.progress = progress
        if result_links:
            job.result_links.update(result_links)
        if error:
            job.error = error
        if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            job.completed_at = time.time()
            if job.callback_url:
                asyncio.create_task(self._dispatch_webhook(job))

    async def _dispatch_webhook(self, job: JobResult) -> None:
        """Sends an HMAC-SHA256 signed HTTP POST payload to the callback_url."""
        payload_bytes = job.model_dump_json().encode("utf-8")
        signature = hmac.new(
            settings.WEBHOOK_SECRET_KEY.encode("utf-8"),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-IncuBrix-Signature": f"sha256={signature}"
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(job.callback_url, content=payload_bytes, headers=headers)
        except Exception:
            pass  # Best effort delivery


job_manager = JobManager()
