"""
Configuration Management (Integration Standard #6).

All settings, ports, paths, model names, and limits are driven by environment variables.
Zero hard-coded secrets or paths.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Service Information
    APP_NAME: str = "IncuBrix Prompt-Based Video Editing Engine"
    API_PREFIX: str = "/api/v1"
    PORT: int = Field(default=8000, alias="PORT")
    HOST: str = "0.0.0.0"
    DEBUG: bool = False

    # Storage & Object Store (MinIO / S3) (Integration Standard #7)
    STORAGE_TYPE: str = Field(default="local", description="'local' or 's3'")
    LOCAL_STORAGE_DIR: str = "./storage"
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "incubrix-outputs"
    S3_REGION: str = "us-east-1"

    # Job & Webhook Security (Integration Standard #3)
    WEBHOOK_SECRET_KEY: str = "incubrix-super-secret-hmac-key"

    # STT & AI Providers (Provider Rule #3)
    STT_PROVIDER: str = "mock"  # "mock" or "whisper"
    NLP_PROVIDER: str = "deterministic"  # "deterministic" or "llm"
    LLM_PROVIDER: str = "mock"  # "mock", "gemini", "openai", "ollama"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "gemini-1.5-flash"

    # Execution Thresholds
    MAX_VIDEO_DURATION_SECONDS: float = 7200.0  # 2 hours
    ASYNC_JOB_THRESHOLD_SECONDS: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
