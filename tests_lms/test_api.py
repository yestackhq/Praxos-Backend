from __future__ import annotations


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["auth_enabled"] is False  # no Clerk key in tests


def test_workspace(client):
    r = client.get("/api/workspace")
    assert r.status_code == 200
    assert r.json()["name"] == "Meridian Health"


def test_learner_home(client):
    r = client.get("/api/learner/home")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Daniel Acheampong"
    assert body["understanding"] == 74
    assert body["path_progress"] == "1 / 5"
    assert len(body["path"]) == 5
    # path is ordered; first item mastered, second in progress
    assert body["path"][0]["status"] == "mastered"
    assert body["path"][1]["status"] == "in_progress"
    assert body["path"][1]["progress"] == 62


def test_learner_sessions(client):
    r = client.get("/api/learner/sessions")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 4
    assert rows[0]["doc"] == "Code of conduct"
    assert all("score" in row for row in rows)


def test_admin_overview(client):
    r = client.get("/api/admin/overview")
    assert r.status_code == 200
    body = r.json()
    assert len(body["kpis"]) == 5
    assert len(body["cohort_health"]) == 4
    # at-risk learners have understanding < 55
    assert all(p["understanding"] < 55 for p in body["people_at_risk"])
    assert any(p["name"] == "Grace Mwangi" for p in body["people_at_risk"])


def test_admin_people(client):
    r = client.get("/api/admin/people")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 8
    roles = {row["role"] for row in rows}
    assert {"Learner", "Manager", "Admin"} <= roles


def test_admin_teams_and_cohorts(client):
    teams = client.get("/api/admin/teams").json()
    cohorts = client.get("/api/admin/cohorts").json()
    assert len(teams) == 6
    assert len(cohorts) == 4
    eng = next(t for t in teams if t["name"] == "Engineering")
    assert eng["lead"] == "Marcus Lindqvist"
    assert eng["avg"] == 82


def test_documents_and_plan(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 6
    gdpr = next(d for d in docs if d["name"] == "Data protection & GDPR")
    plan = client.get(f"/api/documents/{gdpr['id']}/plan").json()
    assert plan["doc"] == "Data protection & GDPR"
    assert len(plan["modules"]) == 5
    assert plan["modules"][0]["title"] == "What personal data means"
    assert "Definitions" in plan["modules"][0]["topics"]


def test_document_plan_404(client):
    r = client.get("/api/documents/99999/plan")
    assert r.status_code == 404


def test_bootstrap_requires_identity(client):
    # No verified token → no user identity → 401 (never serves demo here).
    r = client.post("/api/bootstrap", json={"name": "Nobody"})
    assert r.status_code == 401


def test_bootstrap_new_user_starts_empty(client):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: {"sub": "user_test_ada"}
    try:
        r = client.post("/api/bootstrap", json={"name": "Ada Lovelace", "email": "ada@analytical.dev"})
        assert r.status_code == 200
        b = r.json()
        assert b["mode"] == "user"
        # the new user's own identity, NOT the Meridian/Daniel demo
        assert b["learner"]["name"] == "Ada Lovelace"
        assert b["learner"]["firstName"] == "Ada"
        assert b["workspace"]["name"] == "Ada's workspace"
        # everything empty/zeroed
        assert b["learner"]["understanding"] == 0
        assert b["learner"]["sessions"] == 0
        assert b["continueLearning"] is None
        assert b["learningPath"] == []
        assert b["pastSessions"] == []
        assert b["myDocuments"] == []
        assert b["admin"]["cohorts"] == []
        assert b["admin"]["teams"] == []
        assert b["admin"]["documents"] == []
        assert b["admin"]["understandingTrend"] == []
        # only themselves in their workspace
        assert [p["name"] for p in b["admin"]["people"]] == ["Ada Lovelace"]
        # idempotent: a second call returns the same workspace, no demo bleed-in
        r2 = client.post("/api/bootstrap", json={"name": "Ada Lovelace", "email": "ada@analytical.dev"})
        assert r2.json()["workspace"]["name"] == "Ada's workspace"
    finally:
        app.dependency_overrides.pop(optional_claims, None)


def _as(claims):
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides[optional_claims] = lambda: claims


def _clear():
    from lms_app.auth import optional_claims
    from lms_app.main import app

    app.dependency_overrides.pop(optional_claims, None)


def test_invite_creates_admin_who_joins_on_signup(client):
    try:
        # Owner signs up → personal workspace, Admin.
        _as({"sub": "owner_1"})
        owner = client.post("/api/bootstrap", json={"name": "Grace Hopper", "email": "grace@navy.mil"}).json()
        assert owner["role"] == "Admin"
        ws_name = owner["workspace"]["name"]

        # Owner invites a teammate AS AN ADMIN.
        r = client.post("/api/team/invites", json={"email": "Margaret@nasa.gov", "role": "Admin"})
        assert r.status_code == 201
        assert r.json()["role"] == "Admin"

        # Invite shows up as pending on the owner's bundle.
        owner2 = client.post("/api/bootstrap", json={"name": "Grace Hopper", "email": "grace@navy.mil"}).json()
        assert any(i["email"] == "margaret@nasa.gov" for i in owner2["admin"]["pendingInvites"])

        # Invited person signs up → joins the SAME workspace as Admin (not a new personal one).
        _as({"sub": "invitee_1"})
        invitee = client.post("/api/bootstrap", json={"name": "Margaret Hamilton", "email": "margaret@nasa.gov"}).json()
        assert invitee["workspace"]["name"] == ws_name
        assert invitee["role"] == "Admin"
        names = sorted(p["name"] for p in invitee["admin"]["people"])
        assert names == ["Grace Hopper", "Margaret Hamilton"]
        # Invite is consumed.
        assert invitee["admin"]["pendingInvites"] == []
    finally:
        _clear()


def test_invite_requires_admin_and_role_change(client):
    try:
        # A learner (non-admin) cannot invite.
        _as({"sub": "owner_2"})
        client.post("/api/bootstrap", json={"name": "Owner Two", "email": "o2@x.dev"})
        client.post("/api/team/invites", json={"email": "lena@x.dev", "role": "Learner"})
        _as({"sub": "learner_2"})
        client.post("/api/bootstrap", json={"name": "Lena", "email": "lena@x.dev"})  # joins as Learner
        forbidden = client.post("/api/team/invites", json={"email": "x@y.dev", "role": "Admin"})
        assert forbidden.status_code == 403

        # Owner promotes the learner to Admin.
        _as({"sub": "owner_2"})
        people = client.post("/api/bootstrap", json={"name": "Owner Two", "email": "o2@x.dev"}).json()["admin"]["people"]
        lena = next(p for p in people if p["email"] == "lena@x.dev")
        promo = client.patch(f"/api/team/members/{lena['id']}/role", json={"role": "Admin"})
        assert promo.status_code == 200
        assert promo.json()["role"] == "Admin"
    finally:
        _clear()


def test_onboarding_flow(client):
    try:
        # A fresh owner needs onboarding (regardless of how they authenticated).
        _as({"sub": "onb_owner"})
        b = client.post("/api/bootstrap", json={"name": "Owner", "email": "owner@onb.dev"}).json()
        assert b["needsOnboarding"] is True
        # Completing it renames the workspace and clears the flag.
        done = client.post("/api/onboarding/complete", json={"workspaceName": "Acme Inc."}).json()
        assert done["needsOnboarding"] is False
        assert done["workspace"]["name"] == "Acme Inc."
        # Stays onboarded on next login.
        again = client.post("/api/bootstrap", json={"name": "Owner", "email": "owner@onb.dev"}).json()
        assert again["needsOnboarding"] is False
    finally:
        _clear()


def test_upload_document_appears_in_workspace(client):
    try:
        _as({"sub": "doc_owner"})
        b0 = client.post("/api/bootstrap", json={"name": "Doc Owner", "email": "do@x.dev"}).json()
        assert b0["admin"]["documents"] == []
        r = client.post("/api/documents", json={"name": "Security policy.pdf", "sections": 4})
        assert r.status_code == 201
        assert r.json()["status"] == "Indexing"
        b1 = client.post("/api/bootstrap", json={"name": "Doc Owner", "email": "do@x.dev"}).json()
        names = [d["name"] for d in b1["admin"]["documents"]]
        assert "Security policy.pdf" in names
    finally:
        _clear()


def test_invited_member_skips_onboarding(client):
    try:
        _as({"sub": "onb_owner2"})
        client.post("/api/bootstrap", json={"name": "Owner Two", "email": "owner2@onb.dev"})
        client.post("/api/onboarding/complete", json={"workspaceName": "Globex"})
        client.post("/api/team/invites", json={"email": "newadmin@onb.dev", "role": "Admin"})
        # An invited admin joins an onboarded workspace → no onboarding.
        _as({"sub": "onb_invitee"})
        inv = client.post("/api/bootstrap", json={"name": "New Admin", "email": "newadmin@onb.dev"}).json()
        assert inv["workspace"]["name"] == "Globex"
        assert inv["needsOnboarding"] is False
    finally:
        _clear()
