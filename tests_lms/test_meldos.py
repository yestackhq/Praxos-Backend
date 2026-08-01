from __future__ import annotations

"""MeldOS gateway integration.

Proves the four things that must hold for every model call:
  1. it goes to $MELDOS_API_BASE_URL/v1/chat/completions with
     `Authorization: Bearer $MELDOS_APPLICATION_KEY` and model `company-chat-model`;
  2. exactly one attribution header is attached, and a user's token never leaves
     for any host other than MeldOS;
  3. 401 / 403 / 429 / 5xx are each handled, not collapsed;
  4. the application key and the user's token appear in no log record and in
     nothing the client can see.
"""

import io
import json
import logging

import httpx
import pytest

from lms_app import llm, meldos
from lms_app.config import settings

FAKE_KEY = "mk_test_0123456789abcdef.SUPERSECRETVALUE"
FAKE_HOST = "control-api-testing.up.railway.app"
FAKE_TOKEN = "eyJhbGciOiJSUzI1NiJ9.USER-TOKEN-DO-NOT-LEAK.sig"

MODEL_ALIAS = "company-chat-model"


class Captured:
    """The last request meldos.chat_completion tried to send."""

    def __init__(self):
        self.url = None
        self.headers = {}
        self.json = None


@pytest.fixture
def meldos_on(monkeypatch):
    monkeypatch.setattr(settings, "MELDOS_API_BASE_URL", FAKE_HOST, raising=False)
    monkeypatch.setattr(settings, "MELDOS_APPLICATION_KEY", FAKE_KEY, raising=False)
    monkeypatch.setattr(settings, "MELDOS_MODEL", MODEL_ALIAS, raising=False)
    llm._chat_client.cache_clear()
    yield
    llm._chat_client.cache_clear()


@pytest.fixture
def capture(monkeypatch, meldos_on):
    """Intercept the outbound call and reply with a canned completion."""
    cap = Captured()

    def fake_post(url, *, headers=None, json=None, timeout=None):  # noqa: A002
        cap.url = url
        cap.headers = dict(headers or {})
        cap.json = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(meldos.httpx, "post", fake_post)
    return cap


def _respond(monkeypatch, status: int, *, body: str = "", headers: dict | None = None):
    """Make the next MeldOS call return ``status``."""
    resp_headers = headers or {}

    def fake_post(url, **_kwargs):
        return httpx.Response(
            status, text=body, headers=resp_headers, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(meldos.httpx, "post", fake_post)


# ---- 1. the request contract -------------------------------------------------


def test_sends_key_model_and_url(capture):
    llm.chat_json("system prompt", "user prompt")

    assert capture.url == f"https://{FAKE_HOST}/v1/chat/completions"
    assert capture.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert capture.headers["Content-Type"] == "application/json"
    assert capture.json["model"] == MODEL_ALIAS
    assert capture.json["messages"][0]["role"] == "system"
    assert capture.json["messages"][-1]["content"] == "user prompt"


def test_session_id_is_always_sent(capture):
    """MeldOS rejects a model request without X-Session-ID (VALIDATION_ERROR), so
    an explicit id is passed through and a fallback is generated when there is
    none — a caller must never be able to omit it."""
    llm.chat_json("s", "u", session_id="praxos-u9-d4-s2")
    assert capture.headers["X-Session-ID"] == "praxos-u9-d4-s2"

    llm.chat_json("s", "u")
    assert capture.headers["X-Session-ID"].startswith("praxos-")


def test_base_url_accepts_a_bare_host(meldos_on):
    """The configured value is documented as a bare host, so it must not need a
    scheme to produce a valid URL."""
    assert meldos.base_url().startswith("https://")
    assert meldos.chat_url().endswith("/v1/chat/completions")
    assert "//v1" not in meldos.chat_url()


# ---- 2. attribution ----------------------------------------------------------


def test_verified_attribution_sends_only_the_token(capture):
    llm.chat_json("s", "u", end_user=llm.EndUser.verified(FAKE_TOKEN))
    assert capture.headers["X-End-User-Token"] == FAKE_TOKEN
    assert "X-End-User-Id" not in capture.headers


def test_claimed_attribution_sends_the_persons_name(capture):
    llm.chat_json("s", "u", end_user=llm.EndUser.claimed("Kiran Varma"))
    assert capture.headers["X-End-User-Id"] == "Kiran Varma"
    assert "X-End-User-Token" not in capture.headers


def test_only_one_attribution_header_is_ever_sent(capture):
    """Both supplied → the verified one wins; sending both would be ambiguous."""
    llm.chat_json("s", "u", end_user=meldos.EndUser(token=FAKE_TOKEN, name="Kiran Varma"))
    assert "X-End-User-Token" in capture.headers
    assert "X-End-User-Id" not in capture.headers


def test_no_attribution_header_when_there_is_nothing_to_attribute(capture):
    llm.chat_json("s", "u", end_user=llm.EndUser.claimed(""))
    assert "X-End-User-Token" not in capture.headers
    assert "X-End-User-Id" not in capture.headers


def test_user_token_is_never_sent_to_another_host(meldos_on):
    """The hard rule: a person's sign-in credential goes to MeldOS and nowhere
    else. Repointing a base URL must not turn into a token leak."""
    end_user = llm.EndUser.verified(FAKE_TOKEN)
    for foreign in (
        "https://api.openai.com/v1/chat/completions",
        "https://evil.example.com/v1/chat/completions",
        f"http://{FAKE_HOST}/v1/chat/completions",  # downgraded scheme
        f"https://{FAKE_HOST}.evil.com/v1/chat/completions",  # suffix attack
    ):
        assert meldos.attribution_headers(end_user, url=foreign) == {}

    assert meldos.attribution_headers(end_user, url=meldos.chat_url()) == {
        "X-End-User-Token": FAKE_TOKEN
    }


# ---- 3. failure modes --------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [(401, 401), (403, 403), (429, 429), (500, 500), (502, 502), (503, 503)],
)
def test_documented_failures_surface_distinctly(monkeypatch, meldos_on, status, expected):
    _respond(monkeypatch, status, body="upstream said no")
    with pytest.raises(meldos.MeldOSError) as err:
        llm.chat_json("s", "u")
    assert err.value.status == expected
    assert err.value.detail  # actionable, not empty


def test_429_carries_retry_after(monkeypatch, meldos_on):
    _respond(monkeypatch, 429, headers={"Retry-After": "12"})
    with pytest.raises(meldos.MeldOSError) as err:
        llm.chat_json("s", "u")
    assert err.value.retry_after == "12"
    assert "12" in err.value.detail


def test_transport_failure_is_not_mistaken_for_a_rejection(monkeypatch, meldos_on):
    def boom(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(meldos.httpx, "post", boom)
    with pytest.raises(meldos.MeldOSError) as err:
        llm.chat_json("s", "u")
    assert err.value.status == 0


# ---- 4. secret hygiene -------------------------------------------------------


def test_no_secret_in_any_log_record(monkeypatch, meldos_on, caplog):
    """Capture EVERY logger at DEBUG across a success, a rejection and a rate
    limit, then assert neither secret reached the log stream."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG):
            # An upstream error that echoes the Authorization header back at us —
            # the nastiest realistic way a key ends up in a log.
            _respond(monkeypatch, 500, body=f"failed for Authorization: Bearer {FAKE_KEY}")
            with pytest.raises(meldos.MeldOSError) as err:
                llm.chat_json("s", "u", end_user=llm.EndUser.verified(FAKE_TOKEN))
            assert FAKE_KEY not in err.value.detail
            assert FAKE_TOKEN not in err.value.detail

            _respond(monkeypatch, 429)
            with pytest.raises(meldos.MeldOSError):
                llm.chat_json("s", "u", end_user=llm.EndUser.verified(FAKE_TOKEN))
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)

    logged = stream.getvalue() + caplog.text
    assert FAKE_KEY not in logged
    assert FAKE_TOKEN not in logged


def test_scrubber_redacts_an_echoed_key(meldos_on):
    assert FAKE_KEY not in meldos._safe(f"Bearer {FAKE_KEY} rejected")


# ---- end to end through the API ---------------------------------------------


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import bearer_token, optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)
    app.dependency_overrides.pop(bearer_token, None)


def test_scoring_request_attributes_to_the_signed_in_learner(client, monkeypatch, capture):
    """End to end: a learner scoring their own session forwards THEIR token, the
    call carries the application key and the model alias, and neither secret
    appears in the response the browser receives."""
    from lms_app.auth import bearer_token
    from lms_app.main import app
    from tests_lms.test_indexing import _minimal_pdf

    app.dependency_overrides[bearer_token] = lambda: FAKE_TOKEN

    def scored(url, *, headers=None, json=None, timeout=None):  # noqa: A002
        capture.url, capture.headers, capture.json = url, dict(headers or {}), json
        payload = {
            "score": 78,
            "covered": 100,
            "summary": "Explained the reporting window unprompted.",
            "topics": [{"name": "Reporting", "score": 78, "evidence": "within 24 hours"}],
            "strengths": ["Knew the window"],
            "gaps": [],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json_dumps(payload)}}]},
            request=httpx.Request("POST", url),
        )

    json_dumps = json.dumps
    monkeypatch.setattr(meldos.httpx, "post", scored)

    try:
        _as({"sub": "meldos_learner"})
        client.post("/api/bootstrap", json={"name": "Kiran Varma", "email": "kiran@meldos.dev"})
        doc = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "policy.pdf",
                    io.BytesIO(_minimal_pdf("Report incidents to IT within 24 hours.")),
                    "application/pdf",
                )
            },
        ).json()

        resp = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "tutor", "text": "When must an incident be reported?"},
                    {
                        "role": "learner",
                        "text": "Within twenty four hours, straight to the IT security team.",
                    },
                ],
            },
        )
        assert resp.status_code == 200, resp.text

        # The contract, as actually sent by the running app.
        assert capture.url == f"https://{FAKE_HOST}/v1/chat/completions"
        assert capture.headers["Authorization"] == f"Bearer {FAKE_KEY}"
        assert capture.json["model"] == MODEL_ALIAS
        assert capture.headers["X-Session-ID"]  # required by MeldOS
        # Verified attribution: the learner's own token, not their Clerk id.
        assert capture.headers["X-End-User-Token"] == FAKE_TOKEN
        assert "X-End-User-Id" not in capture.headers

        # Nothing the browser receives contains either secret.
        body = resp.text
        assert FAKE_KEY not in body
        assert FAKE_TOKEN not in body
        assert resp.json()["score"] == 78
    finally:
        _clear()


def test_rate_limit_reaches_the_client_as_429_without_secrets(client, monkeypatch, meldos_on):
    from lms_app.auth import bearer_token
    from lms_app.main import app
    from tests_lms.test_indexing import _minimal_pdf

    app.dependency_overrides[bearer_token] = lambda: FAKE_TOKEN
    try:
        _as({"sub": "meldos_limited"})
        client.post("/api/bootstrap", json={"name": "Rate Limited", "email": "rl@meldos.dev"})
        doc = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "p.pdf",
                    io.BytesIO(_minimal_pdf("Some indexed policy text about reporting.")),
                    "application/pdf",
                )
            },
        ).json()

        _respond(monkeypatch, 429, headers={"Retry-After": "30"})
        resp = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "learner", "text": "You report an incident within twenty four hours."}
                ],
            },
        )
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "30"
        assert FAKE_KEY not in resp.text
        assert FAKE_TOKEN not in resp.text
    finally:
        _clear()


def test_rejected_key_does_not_leak_through_the_api(client, monkeypatch, meldos_on):
    """A 401 from MeldOS is a server misconfiguration, so the learner gets a 503 —
    and the response never contains the key that was rejected."""
    from lms_app.auth import bearer_token
    from lms_app.main import app
    from tests_lms.test_indexing import _minimal_pdf

    app.dependency_overrides[bearer_token] = lambda: FAKE_TOKEN
    try:
        _as({"sub": "meldos_401"})
        client.post("/api/bootstrap", json={"name": "Bad Key", "email": "bk@meldos.dev"})
        doc = client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "p.pdf",
                    io.BytesIO(_minimal_pdf("Indexed policy text for the 401 case.")),
                    "application/pdf",
                )
            },
        ).json()

        _respond(monkeypatch, 401, body=f"invalid key {FAKE_KEY}")
        resp = client.post(
            "/api/sessions/score",
            json={
                "documentId": doc["id"],
                "transcript": [
                    {"role": "learner", "text": "Incidents go to IT within twenty four hours."}
                ],
            },
        )
        assert resp.status_code == 503
        assert FAKE_KEY not in resp.text
        assert FAKE_TOKEN not in resp.text
    finally:
        _clear()


def test_health_reports_the_gateway_but_never_the_key(client, meldos_on):
    body = client.get("/api/health").json()
    assert body["llm"]["provider"] == "meldos"
    assert body["llm"]["model"] == MODEL_ALIAS
    assert FAKE_KEY not in json.dumps(body)


# ---- embeddings degrade rather than failing an upload ------------------------


def test_embedding_failure_does_not_break_indexing(monkeypatch, caplog):
    """A key with an exhausted balance (429 insufficient_quota) must leave the
    document indexed WITHOUT vectors, not fail the upload. Raising here would
    make a bad key worse than no key."""

    class Boom:
        class embeddings:
            @staticmethod
            def create(**_kw):
                raise RuntimeError("You have no credits remaining. sk-proj-SECRETKEY")

    monkeypatch.setattr(llm, "_embed_client", lambda: Boom())
    with caplog.at_level(logging.WARNING):
        assert llm.embed_texts(["chunk one", "chunk two"]) is None
        assert llm.embed_one("chunk") is None
    assert "keyword overlap" in caplog.text
