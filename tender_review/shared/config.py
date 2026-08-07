from __future__ import annotations

import os
import hashlib
import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from tender_review import __version__


class AppSettings(BaseModel):
    """Process settings loaded once by bootstrap, never by business modules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_name: str = "tender-review"
    version: str = __version__
    environment: str = "local"
    adapter_mode: Literal["fake", "production"] = "fake"
    log_level: str = "INFO"
    log_json: bool = True
    api_prefix: str = "/api/v1"
    readiness_timeout_seconds: float = Field(default=2.0, gt=0)
    database_url: str = ""
    database_echo: bool = False
    database_connect_timeout_seconds: int = Field(default=5, gt=0)
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_bucket: str = "tender-review-artifacts"
    minio_region: str | None = None
    minio_timeout_seconds: float = Field(default=5.0, gt=0)
    minio_max_attempts: int = Field(default=3, ge=1)
    llm_base_url: str = ""
    llm_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    model_config_id: str = "openai-compatible-review-v1"
    model_prompt_version: str = "structured-extraction-v1"
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""), repr=False
    )
    embedding_model: str = ""
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_max_attempts: int = Field(default=3, ge=1)
    embedding_dimensions: int = Field(default=1536, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1)
    worker_poll_interval_seconds: float = Field(default=1.0, ge=0)
    worker_lease_seconds: int = Field(default=30, gt=0)
    worker_id: str | None = None
    document_max_upload_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    workbench_demo_enabled: bool = False
    a7_admission_report_path: str = ""
    a7_attestation_key_id: str = ""
    a7_attestation_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""), repr=False
    )

    @property
    def model_config_hash(self) -> str:
        """Hash only canonical, non-secret settings that affect review execution."""

        payload = {
            "schema_version": 1,
            "provider": "openai-compatible",
            "prompt_version": self.model_prompt_version,
            "llm": {
                "base_url": self.llm_base_url.rstrip("/"),
                "model": self.llm_model,
                "timeout_seconds": self.llm_timeout_seconds,
                "max_attempts": self.llm_max_attempts,
                "temperature": self.llm_temperature,
            },
            "embedding": {
                "base_url": self.embedding_base_url.rstrip("/"),
                "model": self.embedding_model,
                "timeout_seconds": self.embedding_timeout_seconds,
                "max_attempts": self.embedding_max_attempts,
                "dimensions": self.embedding_dimensions,
                "batch_size": self.embedding_batch_size,
            },
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        normalized = "/" + value.strip().strip("/")
        if normalized == "/":
            raise ValueError("api_prefix must not be the root path")
        return normalized

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        prefix: str = "TENDER_REVIEW_",
    ) -> "AppSettings":
        source = os.environ if environ is None else environ
        values: dict[str, object] = {}
        mapping = {
            "SERVICE_NAME": "service_name",
            "VERSION": "version",
            "ENVIRONMENT": "environment",
            "ADAPTER_MODE": "adapter_mode",
            "LOG_LEVEL": "log_level",
            "LOG_JSON": "log_json",
            "API_PREFIX": "api_prefix",
            "READINESS_TIMEOUT_SECONDS": "readiness_timeout_seconds",
            "DATABASE_URL": "database_url",
            "DATABASE_ECHO": "database_echo",
            "DATABASE_CONNECT_TIMEOUT_SECONDS": "database_connect_timeout_seconds",
            "MINIO_ENDPOINT": "minio_endpoint",
            "MINIO_ACCESS_KEY": "minio_access_key",
            "MINIO_SECRET_KEY": "minio_secret_key",
            "MINIO_SECURE": "minio_secure",
            "MINIO_BUCKET": "minio_bucket",
            "MINIO_REGION": "minio_region",
            "MINIO_TIMEOUT_SECONDS": "minio_timeout_seconds",
            "MINIO_MAX_ATTEMPTS": "minio_max_attempts",
            "LLM_BASE_URL": "llm_base_url",
            "LLM_API_KEY": "llm_api_key",
            "LLM_MODEL": "llm_model",
            "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
            "LLM_MAX_ATTEMPTS": "llm_max_attempts",
            "LLM_TEMPERATURE": "llm_temperature",
            "MODEL_CONFIG_ID": "model_config_id",
            "MODEL_PROMPT_VERSION": "model_prompt_version",
            "EMBEDDING_BASE_URL": "embedding_base_url",
            "EMBEDDING_API_KEY": "embedding_api_key",
            "EMBEDDING_MODEL": "embedding_model",
            "EMBEDDING_TIMEOUT_SECONDS": "embedding_timeout_seconds",
            "EMBEDDING_MAX_ATTEMPTS": "embedding_max_attempts",
            "EMBEDDING_DIMENSIONS": "embedding_dimensions",
            "EMBEDDING_BATCH_SIZE": "embedding_batch_size",
            "WORKER_POLL_INTERVAL_SECONDS": "worker_poll_interval_seconds",
            "WORKER_LEASE_SECONDS": "worker_lease_seconds",
            "WORKER_ID": "worker_id",
            "DOCUMENT_MAX_UPLOAD_BYTES": "document_max_upload_bytes",
            "WORKBENCH_DEMO_ENABLED": "workbench_demo_enabled",
            "A7_ADMISSION_REPORT_PATH": "a7_admission_report_path",
            "A7_ATTESTATION_KEY_ID": "a7_attestation_key_id",
            "A7_ATTESTATION_KEY": "a7_attestation_key",
        }
        unprefixed = {
            "DATABASE_URL",
            "MINIO_ENDPOINT",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
            "MINIO_SECURE",
            "MINIO_BUCKET",
            "MINIO_REGION",
        }
        for env_suffix, field_name in mapping.items():
            prefixed_name = prefix + env_suffix
            if prefixed_name in source:
                values[field_name] = source[prefixed_name]
            elif env_suffix in unprefixed and env_suffix in source:
                values[field_name] = source[env_suffix]
        for field_name, env_suffix in (
            ("log_json", "LOG_JSON"),
            ("database_echo", "DATABASE_ECHO"),
            ("minio_secure", "MINIO_SECURE"),
            ("workbench_demo_enabled", "WORKBENCH_DEMO_ENABLED"),
        ):
            if field_name in values:
                values[field_name] = _parse_bool(
                    str(values[field_name]), prefix + env_suffix
                )
        return cls.model_validate(values)


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")
