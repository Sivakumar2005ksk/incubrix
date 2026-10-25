"""
Standardized JSON Error Handling (Integration Standard #5).

All error responses strictly follow the schema:
{
  "error": {
    "code": "UPPER_SNAKE_CASE",
    "message": "Human readable message",
    "retryable": bool,
    "details": {}
  }
}
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException
from pydantic import BaseModel


class ErrorPayload(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: Dict[str, Any] = {}


class StandardErrorResponse(BaseModel):
    error: ErrorPayload


class IncuBrixAPIException(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
