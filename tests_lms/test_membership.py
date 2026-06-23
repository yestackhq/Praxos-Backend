from __future__ import annotations

"""Multi-workspace membership + switcher: a person can own one workspace and be a
learner in another, switch between them via the X-Workspace-Id header, and accepted
invitees show as members (not pending). Reproduces the originally-reported bug."""


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)


def test_owner_of_one_is_learner_in_another_and_can_switch(client):
    try:
        # Alice signs up first → her own personal workspace, as its Admin.
        _as({"sub": "subA"})
        a0 = client.post("/api/bootstrap", json={"name": "Alice", "email": "alice@a.dev"}).json()
        a_id = a0["activeWorkspaceId"]
        assert a0["role"] == "Admin"

        # Bob creates Beta and invites Alice as a Learner.
        _as({"sub": "subB"})
        client.post("/api/bootstrap", json={"name": "Bob", "email": "bob@b.dev"})
        client.post("/api/onboarding/complete", json={"workspaceName": "Beta"})
        b_id = client.post("/api/bootstrap", json={"name": "Bob", "email": "bob@b.dev"}).json()["activeWorkspaceId"]
        assert client.post("/api/team/invites", json={"email": "alice@a.dev", "role": "Learner"}).status_code == 201

        # Alice bootstraps again → now a member of BOTH; her own stays the default active.
        _as({"sub": "subA"})
        a = client.post("/api/bootstrap", json={"name": "Alice", "email": "alice@a.dev"}).json()
        names = {w["name"]: w["role"] for w in a["workspaces"]}
        assert names.get("Beta") == "Learner", names
        assert a["activeWorkspaceId"] == a_id and a["role"] == "Admin"

        # Switch to Beta via the header → learner view, no onboarding.
        sw = client.post(
            "/api/bootstrap",
            json={"name": "Alice", "email": "alice@a.dev"},
            headers={"X-Workspace-Id": str(b_id)},
        ).json()
        assert sw["activeWorkspaceId"] == b_id and sw["role"] == "Learner"
        assert sw["needsOnboarding"] is False

        # A forged/stale header (a workspace Alice isn't in) can never reach it — falls back.
        bogus = client.post(
            "/api/bootstrap",
            json={"name": "Alice", "email": "alice@a.dev"},
            headers={"X-Workspace-Id": "99999"},
        ).json()
        assert bogus["activeWorkspaceId"] in (a_id, b_id)

        # From Beta's side, Alice is a confirmed Learner — not a pending invite.
        _as({"sub": "subB"})
        beta = client.post(
            "/api/bootstrap",
            json={"name": "Bob", "email": "bob@b.dev"},
            headers={"X-Workspace-Id": str(b_id)},
        ).json()["admin"]
        assert "alice@a.dev" not in [i["email"] for i in beta["pendingInvites"]]
        assert any(p["email"] == "alice@a.dev" and p["role"] == "Learner" for p in beta["people"])
    finally:
        _clear()


def test_create_workspace_adds_to_switcher(client):
    try:
        _as({"sub": "subC"})
        first = client.post("/api/bootstrap", json={"name": "Cara", "email": "cara@c.dev"}).json()["activeWorkspaceId"]
        new = client.post("/api/workspaces", json={"name": "Second WS"}).json()
        assert new["role"] == "Admin"
        after = client.post(
            "/api/bootstrap",
            json={"name": "Cara", "email": "cara@c.dev"},
            headers={"X-Workspace-Id": str(new["id"])},
        ).json()
        assert after["activeWorkspaceId"] == new["id"]
        assert {first, new["id"]} <= {w["id"] for w in after["workspaces"]}
    finally:
        _clear()
