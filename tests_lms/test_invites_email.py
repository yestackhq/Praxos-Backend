from __future__ import annotations


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)


def test_invite_triggers_clerk_email_when_configured(client, monkeypatch):
    """When Clerk is configured, creating an invite sends a Clerk invitation and
    stores its id; a later revoke calls Clerk's revoke."""
    sent: list[tuple[str, str]] = []
    revoked: list[str] = []
    from lms_app.routers import bootstrap

    monkeypatch.setattr(
        bootstrap.clerk_api, "create_invitation", lambda email, role, redirect_url=None: sent.append((email, role)) or "inv_clerk_1"
    )
    monkeypatch.setattr(bootstrap.clerk_api, "revoke_invitation", lambda cid: revoked.append(cid))
    try:
        _as({"sub": "mail_owner"})
        client.post("/api/bootstrap", json={"name": "Mail Owner", "email": "mo@x.dev"})
        r = client.post("/api/team/invites", json={"email": "Newbie@x.dev", "role": "Manager"})
        assert r.status_code == 201
        assert r.json()["emailSent"] is True
        assert sent == [("newbie@x.dev", "Manager")]  # email lowercased, role forwarded

        # Revoke should call Clerk with the stored invitation id.
        b = client.post("/api/bootstrap", json={"name": "Mail Owner", "email": "mo@x.dev"}).json()
        inv_id = next(i["id"] for i in b["admin"]["pendingInvites"] if i["email"] == "newbie@x.dev")
        assert client.delete(f"/api/team/invites/{inv_id}").status_code == 204
        assert revoked == ["inv_clerk_1"]
    finally:
        _clear()


def test_invite_works_without_clerk_key(client):
    """No Clerk key (the default in tests) → invite still stored, emailSent False."""
    try:
        _as({"sub": "nokey_owner"})
        client.post("/api/bootstrap", json={"name": "NoKey", "email": "nk@x.dev"})
        r = client.post("/api/team/invites", json={"email": "pending@x.dev", "role": "Learner"})
        assert r.status_code == 201
        assert r.json()["emailSent"] is False
    finally:
        _clear()
