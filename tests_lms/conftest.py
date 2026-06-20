from __future__ import annotations

import os
import tempfile

import pytest

# Point the app at a throwaway SQLite DB and a non-existent env file BEFORE
# importing the app, so settings never pick up a real backend/.env.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["LMS_ENV_FILE"] = "/nonexistent.env"
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp.name}"
os.environ["SEED_ON_STARTUP"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from lms_app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # triggers lifespan -> create_all + seed
        yield c
