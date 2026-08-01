from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from env / backend/.env."""

    model_config = SettingsConfigDict(env_file=os.getenv("LMS_ENV_FILE", ".env"), extra="ignore")

    APP_NAME: str = "Praxos LMS API"
    VERSION: str = "0.2.0"

    # SQLite by default so the app runs with no external services.
    # For Supabase: postgresql+psycopg://postgres.<ref>:<pwd>@<host>:5432/postgres
    DATABASE_URL: str = "sqlite:///./praxos_lms.db"

    SEED_ON_STARTUP: bool = False  # demo seed only when explicitly enabled (never in prod)
    CORS_ORIGINS: str = "http://localhost:5173"

    # Where invited users land after accepting a Clerk invitation.
    APP_BASE_URL: str = "http://localhost:5173"

    # Postgres schema to isolate LMS tables from any existing app tables.
    # Ignored on SQLite (which has no real schema support).
    DB_SCHEMA: str = "praxos_lms"

    @property
    def db_schema(self) -> Optional[str]:
        return None if self.DATABASE_URL.startswith("sqlite") else self.DB_SCHEMA

    # Clerk auth. When CLERK_JWKS_URL is unset, auth is not enforced (review mode).
    CLERK_JWKS_URL: Optional[str] = None
    CLERK_ISSUER: Optional[str] = None
    CLERK_SECRET_KEY: Optional[str] = None

    # ---- MeldOS gateway (see lms_app/meldos.py) ------------------------------
    # When both are set, chat completions go through MeldOS instead of straight to
    # a model vendor, so spend is metered per application and per person. The
    # application key is a SECRET: server-side only, never in the client bundle,
    # never committed (it belongs in .env, which is gitignored).
    MELDOS_API_BASE_URL: Optional[str] = None
    MELDOS_APPLICATION_KEY: Optional[str] = None
    MELDOS_MODEL: str = "company-chat-model"
    MELDOS_TIMEOUT: float = 60.0

    @property
    def meldos_enabled(self) -> bool:
        return bool(self.MELDOS_API_BASE_URL and self.MELDOS_APPLICATION_KEY)

    # ---- AI provider (see lms_app/llm.py) ------------------------------------
    # The provider is a deployment choice, not a code constant. `openai_compatible`
    # covers any endpoint speaking /v1/chat/completions (Groq, Together, vLLM,
    # OpenRouter, LiteLLM, an in-house gateway): set LLM_BASE_URL + LLM_MODEL.
    LLM_PROVIDER: str = "openai"  # openai | openai_compatible | poke | none
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: Optional[str] = None  # falls back to OPENAI_API_KEY
    LLM_MODEL: str = "gpt-4o"

    # Embeddings can come from a different provider than chat.
    EMBED_BASE_URL: Optional[str] = None  # falls back to LLM_BASE_URL
    EMBED_API_KEY: Optional[str] = None  # falls back to LLM_API_KEY / OPENAI_API_KEY
    EMBED_MODEL: Optional[str] = None  # falls back to OPENAI_EMBED_MODEL
    EMBED_DIM: Optional[int] = None  # falls back to OPENAI_EMBED_DIM

    # Legacy OpenAI-specific names, kept so existing deploys keep working.
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBED_DIM: int = 1536

    @property
    def llm_api_key(self) -> Optional[str]:
        return self.LLM_API_KEY or self.OPENAI_API_KEY

    @property
    def embed_base_url(self) -> str:
        return self.EMBED_BASE_URL or self.LLM_BASE_URL

    @property
    def embed_api_key(self) -> Optional[str]:
        return self.EMBED_API_KEY or self.llm_api_key

    @property
    def embed_model(self) -> str:
        return self.EMBED_MODEL or self.OPENAI_EMBED_MODEL

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM or self.OPENAI_EMBED_DIM

    @property
    def openai_enabled(self) -> bool:
        """Back-compat alias: is text generation available at all?"""
        if self.meldos_enabled:
            return True
        return bool(self.llm_api_key) and self.LLM_PROVIDER.lower() not in ("none", "poke")

    # ---- Voice: LiveKit + Deepgram (STT) + Cartesia (TTS) --------------------
    # The browser joins a LiveKit room; the agent worker runs
    # Deepgram -> LLM -> Cartesia and publishes the tutor's audio back.
    LIVEKIT_URL: Optional[str] = None
    LIVEKIT_API_KEY: Optional[str] = None
    LIVEKIT_API_SECRET: Optional[str] = None
    LIVEKIT_ROOM_TTL_MINUTES: int = 60

    DEEPGRAM_API_KEY: Optional[str] = None
    DEEPGRAM_MODEL: str = "nova-3"
    DEEPGRAM_LANGUAGE: str = "multi"

    CARTESIA_API_KEY: Optional[str] = None
    CARTESIA_MODEL: str = "sonic-2"
    # Cartesia voice id for the tutor. Override per deployment.
    CARTESIA_VOICE: str = "638efaaa-4d0c-442e-b701-3fae16aad012"

    # Shared secret the agent worker presents on /api/sessions/agent/* — it is not
    # a learner and has no Clerk session. Unset => those routes are closed.
    AGENT_SHARED_SECRET: Optional[str] = None

    @property
    def livekit_enabled(self) -> bool:
        return bool(self.LIVEKIT_URL and self.LIVEKIT_API_KEY and self.LIVEKIT_API_SECRET)

    @property
    def voice_enabled(self) -> bool:
        return self.livekit_enabled and bool(self.DEEPGRAM_API_KEY) and bool(self.CARTESIA_API_KEY)

    # ---- Memory: mem0 --------------------------------------------------------
    MEM0_API_KEY: Optional[str] = None
    MEMORY_RECAP_FACTS: int = 6  # how many recalled facts ride into the tutor prompt

    @property
    def memory_enabled(self) -> bool:
        return bool(self.MEM0_API_KEY)

    # ---- Poke (notifications only; see lms_app/poke.py) ----------------------
    POKE_API_KEY: Optional[str] = None
    POKE_TIMEOUT: float = 10.0
    POKE_NOTIFY_AT_RISK: bool = False  # opt-in: pushes a message to the admin's Poke

    # ---- Scoring -------------------------------------------------------------
    # A document's score is the weighted mean of its section scores, so one weak
    # section can't erase a strong document and one strong section can't hide gaps.
    AT_RISK_THRESHOLD: int = 55
    MASTERY_THRESHOLD: int = 70  # a section must reach this to count as understood

    @property
    def auth_enabled(self) -> bool:
        return bool(self.CLERK_JWKS_URL)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
