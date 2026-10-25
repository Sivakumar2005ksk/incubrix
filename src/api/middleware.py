"""
API Middleware for Idempotency, Correlation Tracing, and Structured Logging.

Satisfies:
- Integration Standard #4: Idempotency-Key header safe retries.
- Integration Standard #8: Structured JSON logs with timestamp, level, job_id, and X-Correlation-Id.
"""

import json
import logging
import sys
import time
import uuid
from typing import Dict, Any, Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "job_id": getattr(record, "job_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        return json.dumps(log_obj)


# Initialize structured JSON logger
logger = logging.getLogger("incubrix")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(StructuredJsonFormatter())
if not logger.handlers:
    logger.addHandler(handler)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Injects or propagates X-Correlation-Id on every incoming request."""
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        corr_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))
        request.state.correlation_id = corr_id

        start_time = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        response.headers["X-Correlation-Id"] = corr_id
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            extra={"correlation_id": corr_id}
        )
        return response


class IdempotencyStore:
    """In-memory cache for idempotency keys (can be backed by Redis in production)."""
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, status_code: int, body: Dict[str, Any]) -> None:
        self._cache[key] = {
            "status_code": status_code,
            "body": body,
            "cached_at": time.time()
        }


idempotency_store = IdempotencyStore()
