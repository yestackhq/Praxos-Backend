from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from env / backend/.env."""

    model_config = SettingsConfigDict(env_file=os.getenv("LMS_ENV_FILE", ".env"), extra="ignore")

    APP_NAME: str = "Praxos LMS API"
    VERSION: str = "0.1.0"

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

    # OpenAI — powers document embeddings, the realtime voice session, and scoring.
    # When OPENAI_API_KEY is unset, indexing still extracts/chunks text (no vectors)
    # and the voice session is unavailable (the UI degrades gracefully).
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBED_DIM: int = 1536
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_REALTIME_MODEL: str = "gpt-realtime-2"
    OPENAI_REALTIME_VOICE: str = "marin"
    # gpt-4o transcribe models hallucinate far less on silence/noise than whisper-1
    # (which invents "Thank you" etc. from quiet audio). Tunable without a code change.
    OPENAI_TRANSCRIBE_MODEL: str = "gpt-4o-mini-transcribe"

    # NOTE: original-PDF storage is handled CLIENT-SIDE — the browser uploads
    # straight to Supabase Storage with the publishable key (see web/src/lib/
    # docStorage.ts). The backend never needs a Supabase service/secret key.

    # HyperMem (helix) — conversation memory. The voice tutor writes each
    # session's transcript here and reads back a recap at the start of the next
    # session so it remembers the learner across chapters of a document. When
    # HELIX_URL / HELIX_API_KEY are unset, memory degrades gracefully (the tutor
    # just teaches without recall). Auth is a bearer token; origin check is off.
    HELIX_URL: Optional[str] = None
    HELIX_API_KEY: Optional[str] = None
    HELIX_TENANT: str = "praxos"
    HELIX_READ_TIMEOUT: float = 12.0   # /start blocks on this to build the recap
    HELIX_WRITE_TIMEOUT: float = 10.0  # /score fires an async ingest (returns fast)

    @property
    def helix_enabled(self) -> bool:
        return bool(self.HELIX_URL and self.HELIX_API_KEY)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.CLERK_JWKS_URL)

    @property
    def openai_enabled(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
