"""Configuration settings for Text-to-SQL Analyst using pydantic-settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE_PATH.exists():
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)


class Settings(BaseSettings):
    """Application runtime settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database Configuration (PostgreSQL 16 + pgvector)
    POSTGRES_HOST: str = Field(default="127.0.0.1", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field(default="text2sql", description="Database name")
    POSTGRES_USER: str = Field(default="owner", description="Superuser/owner role")
    POSTGRES_PASSWORD: str = Field(
        default="change_this_owner_password", description="Owner password"
    )
    ANALYST_RO_PASSWORD: str = Field(
        default="change_this_analyst_ro_password", description="Read-only analyst role password"
    )
    ANALYST_PII_RO_PASSWORD: str = Field(
        default="change_this_analyst_pii_ro_password",
        description="Analyst role with PII access password",
    )
    APP_RW_PASSWORD: str = Field(
        default="change_this_app_rw_password",
        description="Application memory and RAG read-write role password",
    )

    # OpenAI API Configuration
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL_FALLBACK: str = Field(
        default="",
        description="Optional fallback OpenAI model name to use on rate-limits or transient unavailability",
    )
    OPENAI_TIMEOUT_SECONDS: float = Field(
        default=30.0, description="HTTP request timeout in seconds for OpenAI calls"
    )
    OPENAI_MAX_RETRIES: int = Field(
        default=3, description="Maximum retry attempts with exponential backoff on transient errors"
    )

    # Model Selection (Configurable per OpenAI documentation)
    OPENAI_CHAT_MODEL: str | None = Field(
        default=None, description="Optional override for all chat agents"
    )
    OPENAI_EMBEDDING_MODEL: str | None = Field(
        default=None, description="Optional override for embedding model"
    )
    GEN_MODEL: str = Field(
        default="gpt-4o-mini", description="Model for SQL generation and explanations"
    )
    FAST_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model for planning, rewriting, grading, reranking, and input classification",
    )
    JUDGE_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model for DeepEval LLM evaluation judge",
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="Model for document/query embeddings"
    )
    EMBEDDING_DIMENSIONS: int = Field(default=1536, description="Embedding vector dimensionality")

    @field_validator("GEN_MODEL", "FAST_MODEL", "JUDGE_MODEL", mode="before")
    @classmethod
    def validate_openai_chat_model(cls, v: str | None) -> str:
        """Sanitizes legacy or unconfigured model names from environment."""
        if not v or not any(x in str(v).lower() for x in ("gpt", "o1", "o3", "chatgpt")):
            return "gpt-4o-mini"
        return v

    @field_validator("EMBEDDING_MODEL", mode="before")
    @classmethod
    def validate_openai_embedding_model(cls, v: str | None) -> str:
        """Sanitizes legacy or unconfigured embedding model names from environment."""
        if not v or "text-embedding" not in str(v).lower():
            return "text-embedding-3-small"
        return v

    @field_validator("EMBEDDING_DIMENSIONS", mode="before")
    @classmethod
    def validate_dimensions(cls, v: int | str | None) -> int:
        """Ensures 1536 vector dimensionality for text-embedding-3-small."""
        if v in (768, "768", None, ""):
            return 1536
        return int(v)

    @model_validator(mode="after")
    def apply_model_aliases(self) -> "Settings":
        """Applies OPENAI_CHAT_MODEL and OPENAI_EMBEDDING_MODEL overrides if configured."""
        if self.OPENAI_CHAT_MODEL:
            chat_m = self.OPENAI_CHAT_MODEL.strip()
            if chat_m and any(x in chat_m.lower() for x in ("gpt", "o1", "o3", "chatgpt")):
                self.GEN_MODEL = chat_m
                self.FAST_MODEL = chat_m
        if self.OPENAI_EMBEDDING_MODEL:
            emb_m = self.OPENAI_EMBEDDING_MODEL.strip()
            if emb_m and "text-embedding" in emb_m.lower():
                self.EMBEDDING_MODEL = emb_m
        return self

    # Business Date Literal Rule (Ensures reproducible evaluation)
    AS_OF_DATE: str = Field(
        default="2026-11-02", description="Reference business as-of date for queries"
    )

    # Runtime & Guardrail Controls
    ALLOW_PII: bool = Field(
        default=False, description="Whether to allow access to PII fields (email, phone)"
    )
    SCHEMA_MODE: str = Field(default="rag", description="Schema injection mode: 'rag' or 'full'")
    RERANK_ENABLED: bool = Field(default=True, description="Enable 2nd-stage LLM reranking")
    GRADING_ENABLED: bool = Field(
        default=True, description="Enable retrieval coverage grading and optional re-query"
    )
    COMBINE_PLAN_REWRITE: bool = Field(
        default=False,
        description="Optionally combine intent classification and rewrite in single call",
    )
    MAX_ATTEMPTS: int = Field(
        default=3, description="Maximum self-correction SQL generation attempts"
    )
    MAX_ROWS: int = Field(default=100, description="Maximum rows returned to the user")
    MAX_PLAN_COST: float = Field(
        default=50000.0, description="Maximum EXPLAIN total cost allowed before rejecting query"
    )
    RATE_LIMIT_RPM: int = Field(default=60, description="Client-side request-per-minute throttle")


# Global cached settings instance
settings = Settings()
