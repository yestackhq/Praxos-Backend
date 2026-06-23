from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import clerk_api, indexing, models, workspace
from ..auth import active_membership, optional_claims
from ..config import settings
from ..db import get_db

router = APIRouter(prefix="/api", tags=["workspace"])

VALID_ROLES = {"Learner", "Manager", "Admin"}


class BootstrapIn(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class InviteIn(BaseModel):
    email: str
    role: str = "Learner"


class RoleIn(BaseModel):
    role: str


class OnboardingIn(BaseModel):
    workspaceName: Optional[str] = None
    slug: Optional[str] = None


class DocumentIn(BaseModel):
    name: str
    sections: int = 0


def _active_ws_id(x_workspace_id: Optional[str]) -> Optional[int]:
    return int(x_workspace_id) if (x_workspace_id and x_workspace_id.isdigit()) else None


@router.post("/bootstrap")
def bootstrap(
    body: BootstrapIn,
    x_workspace_id: Optional[str] = Header(default=None, alias="X-Workspace-Id"),
    claims: Optional[dict] = Depends(optional_claims),
    db: Session = Depends(get_db),
) -> dict:
    """Resolve the signed-in person's active membership (creating/joining workspaces,
    applying pending invites) and return their bundle plus the workspaces they can
    switch between."""
    sub = claims.get("sub") if claims else None
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
    try:
        user = workspace.resolve_active_membership(
            db, sub, _active_ws_id(x_workspace_id), body.name or None, body.email or None
        )
    except ValueError:
        # Email not ready yet — tell the client to retry (its bootstrap loop handles it).
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Account not ready, retry")
    return workspace.build_bundle(db, user, body.name or user.name)


@router.post("/onboarding/complete")
def complete_onboarding(
    body: OnboardingIn,
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> dict:
    """Mark the workspace onboarded (optionally renaming it) and return the bundle.
    Triggered the same way regardless of how the user authenticated (Google or email)."""
    ws = db.get(models.Workspace, user.workspace_id)
    name = (body.workspaceName or "").strip()
    if name:
        ws.name = name
    # Workspace link: an explicit slug, else derived from the name.
    ws.slug = workspace.slugify(body.slug or name or ws.name)
    ws.onboarded = True
    db.commit()
    return workspace.build_bundle(db, user, user.name)


class WorkspaceIn(BaseModel):
    name: Optional[str] = None
    slug: Optional[str] = None


@router.post("/workspaces", status_code=status.HTTP_201_CREATED)
def create_workspace(
    body: WorkspaceIn,
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> dict:
    """Create a new workspace and make the caller its Admin, so it appears in their
    switcher and they can switch into it immediately."""
    name = (body.name or "").strip() or f"{workspace.first_name(user.name)}'s workspace"
    ws = models.Workspace(name=name, plan="Admin workspace", onboarded=True)
    ws.slug = workspace.slugify(body.slug or name)
    db.add(ws)
    db.flush()
    db.add(
        models.User(
            clerk_id=user.clerk_id,
            workspace_id=ws.id,
            name=user.name,
            email=user.email,
            role="Admin",
            cohort="—",
        )
    )
    db.commit()
    db.refresh(ws)
    return {"id": ws.id, "name": ws.name, "slug": ws.slug, "role": "Admin"}


@router.post("/documents", status_code=status.HTTP_201_CREATED)
def add_document(body: DocumentIn, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> dict:
    """Add a document to the workspace. Real text extraction/indexing is not wired
    yet, so it lands in 'Indexing' state and shows up in the documents list."""
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can add documents")
    name = body.name.strip() or "Untitled document"
    doc = models.Document(
        workspace_id=user.workspace_id,
        name=name,
        sections=max(0, body.sections),
        assigned=0,
        status="Indexing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"id": doc.id, "name": doc.name, "sections": doc.sections, "assigned": doc.assigned, "status": doc.status}


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB, matches the UI copy


@router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    storage_path: Optional[str] = Form(None),
    user: models.User = Depends(active_membership),
    db: Session = Depends(get_db),
) -> dict:
    """Upload a real PDF: extract + chunk + embed the text and mark it Indexed.
    The original file is persisted to Supabase Storage by the browser (publishable
    key, no server secret); the resulting path is recorded here. Admins only."""
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can add documents")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds 50MB")

    name = (file.filename or "Untitled document").strip()
    doc = models.Document(
        workspace_id=user.workspace_id,
        name=name,
        sections=0,
        assigned=0,
        status="Indexing",
        storage_path=(storage_path or None),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    indexing.index_document(db, doc, data)
    db.refresh(doc)
    return {
        "id": doc.id,
        "name": doc.name,
        "sections": doc.sections,
        "assigned": doc.assigned,
        "status": doc.status,
    }


@router.get("/team/invites")
def list_invites(user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> list[dict]:
    return workspace._pending(db, user.workspace_id)


@router.post("/team/invites", status_code=status.HTTP_201_CREATED)
def create_invite(body: InviteIn, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> dict:
    """Invite a teammate to this workspace with a role (Admin makes them an admin)."""
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can invite people")
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")

    email = str(body.email).lower()
    # Already a member?
    existing = db.scalar(
        select(models.User).where(models.User.workspace_id == user.workspace_id, models.User.email == email)
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="That person is already a member")
    # Re-use a pending invite if present.
    invite = db.scalar(
        select(models.Invite).where(
            models.Invite.workspace_id == user.workspace_id,
            models.Invite.email == email,
            models.Invite.status == "pending",
        )
    )
    if invite is None:
        invite = models.Invite(workspace_id=user.workspace_id, email=email, role=body.role, invited_by=user.name)
        db.add(invite)
    else:
        invite.role = body.role
    db.commit()
    db.refresh(invite)

    # Best-effort: send the actual invitation email via Clerk (no-op without a key).
    invite.clerk_invite_id = clerk_api.create_invitation(
        email, body.role, redirect_url=f"{settings.APP_BASE_URL.rstrip('/')}/sign-up"
    )
    db.commit()
    return {"id": invite.id, "email": invite.email, "role": invite.role, "emailSent": bool(invite.clerk_invite_id)}


@router.delete("/team/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_invite(invite_id: int, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> None:
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can revoke invites")
    invite = db.get(models.Invite, invite_id)
    if invite and invite.workspace_id == user.workspace_id:
        if invite.clerk_invite_id:
            clerk_api.revoke_invitation(invite.clerk_invite_id)
        db.delete(invite)
        db.commit()


@router.patch("/team/members/{member_id}/role")
def set_member_role(member_id: int, body: RoleIn, user: models.User = Depends(active_membership), db: Session = Depends(get_db)) -> dict:
    """Change a member's role (e.g. promote to Admin). Admins only."""
    if not workspace.is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can change roles")
    if body.role not in VALID_ROLES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
    member = db.get(models.User, member_id)
    if member is None or member.workspace_id != user.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    member.role = body.role
    db.commit()
    return {"id": member.id, "role": member.role}
