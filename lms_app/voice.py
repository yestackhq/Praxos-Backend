from __future__ import annotations

"""LiveKit room + access-token minting for the voice tutor.

Topology
--------
    browser  ──audio──►  LiveKit room  ◄──audio──  agent worker (voice_agent/)
                                                     │
                                    Deepgram STT ──► LLM ──► Cartesia TTS

The browser gets a short-lived join token from ``/api/sessions/start`` and never
sees a provider key. The agent worker is dispatched into the room by name and
reads the room's metadata to learn which learner, document and section it is
teaching; it then calls back to this API (with the shared agent secret) for the
grounded instructions and, at the end, posts the transcript for grading.

Room metadata is the ONLY thing that crosses to the worker, so it carries ids —
never the material, and never a learner-facing token.
"""

import json
import logging
from datetime import timedelta
from typing import Optional

from .config import settings

logger = logging.getLogger("praxos.voice")

AGENT_IDENTITY = "praxos-tutor"


def room_name(*, user_id: int, document_id: int, module_idx: int, session_nonce: str) -> str:
    return f"praxos-u{user_id}-d{document_id}-s{module_idx}-{session_nonce}"


def room_metadata(*, workspace_id: int, user_id: int, document_id: int, module_idx: int) -> str:
    return json.dumps(
        {
            "workspaceId": workspace_id,
            "userId": user_id,
            "documentId": document_id,
            "moduleIdx": module_idx,
        }
    )


def mint_join_token(
    *,
    room: str,
    identity: str,
    name: str,
    metadata: str = "",
    ttl_minutes: Optional[int] = None,
) -> Optional[str]:
    """A JWT the browser uses to join ``room``. None when LiveKit is unconfigured."""
    if not settings.livekit_enabled:
        return None
    from livekit import api

    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    token = (
        api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(name)
        .with_grants(grants)
        .with_ttl(timedelta(minutes=ttl_minutes or settings.LIVEKIT_ROOM_TTL_MINUTES))
        # Explicitly dispatch the tutor into this room.
        #
        # The worker registers under an agent_name, which turns OFF LiveKit's
        # automatic dispatch — a named worker joins only rooms it is explicitly
        # asked to. That is what we want: this LiveKit project hosts other
        # products' agents too, and automatic dispatch would have our tutor join
        # their rooms (and theirs join ours). The cost is that a room with no
        # dispatch request gets no agent at all: the learner connects, hears
        # silence, and nothing in the logs looks wrong.
        #
        # Carrying the request on the learner's own token means the dispatch
        # happens exactly when they join, with no extra API call to fail.
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=AGENT_IDENTITY, metadata=metadata or "")]
            )
        )
    )
    if metadata:
        token = token.with_metadata(metadata)
    return token.to_jwt()


async def create_room(*, room: str, metadata: str) -> bool:
    """Pre-create the room carrying its metadata, so the agent worker knows what
    to teach the moment it is dispatched. Best-effort: if this fails the room is
    still auto-created on join and the worker falls back to the participant's
    metadata."""
    if not settings.livekit_enabled:
        return False
    from livekit import api

    lkapi = api.LiveKitAPI(
        url=settings.LIVEKIT_URL,
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
    )
    try:
        await lkapi.room.create_room(
            api.CreateRoomRequest(
                name=room,
                metadata=metadata,
                empty_timeout=300,
                max_participants=2,
            )
        )
        return True
    except Exception as exc:
        logger.warning("livekit create_room failed: %s", exc)
        return False
    finally:
        await lkapi.aclose()
