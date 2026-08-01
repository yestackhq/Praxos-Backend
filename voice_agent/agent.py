from __future__ import annotations

"""The Praxos voice tutor, as a LiveKit agent worker.

    learner's mic ──► Deepgram (STT) ──► LLM ──► Cartesia (TTS) ──► learner

Run it alongside the API:

    python -m voice_agent.agent dev        # local, hot-reload
    python -m voice_agent.agent start      # production worker

The worker is stateless. When it is dispatched into a room it reads the room's
metadata (workspace/user/document/section ids, written by ``/api/sessions/start``),
calls the API for that section's grounded instructions, and teaches. It never
talks to the database and never holds a learner's credentials.

Two things the browser needs from here, both sent as LiveKit data messages:
  • ``section_ready``  — the tutor called ``mark_section_understood``; the UI
    reveals the "Next section" button. It carries the evidence the model had to
    supply, so a section can never be passed on "yeah, got it".
  • ``caption``        — the tutor's line, word-aligned to the audio actually
    being spoken (Cartesia timestamps via ``use_tts_aligned_transcript``), which
    is what lets the subtitle highlight the word currently being said.

And one it accepts:
  • ``advance``        — the learner tapped the button; fetch the next section's
    instructions and switch onto them without reconnecting.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

import httpx
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("praxos.agent")

API_BASE = os.getenv("PRAXOS_API_BASE", "http://localhost:8000").rstrip("/")
AGENT_SECRET = os.getenv("AGENT_SHARED_SECRET", "")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


class SessionContext:
    """What this worker is teaching, and what was said."""

    def __init__(self, meta: dict):
        self.workspace_id: int = int(meta.get("workspaceId", 0))
        self.user_id: int = int(meta.get("userId", 0))
        self.document_id: int = int(meta.get("documentId", 0))
        self.module_idx: int = int(meta.get("moduleIdx", 0))
        self.total_modules: int = 0
        self.is_last: bool = True
        self.transcript: list[dict] = []
        self.scored_upto: int = 0  # transcript index where the current section began
        self.posted: bool = False

    @property
    def valid(self) -> bool:
        return bool(self.user_id and self.document_id)

    def section_turns(self) -> list[dict]:
        return self.transcript[self.scored_upto :]


def _headers() -> dict:
    return {"X-Agent-Secret": AGENT_SECRET, "Content-Type": "application/json"}


async def fetch_context(ctx: SessionContext, *, advancing: bool = False) -> Optional[dict]:
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                f"{API_BASE}/api/sessions/agent/context",
                headers=_headers(),
                json={
                    "workspaceId": ctx.workspace_id,
                    "userId": ctx.user_id,
                    "documentId": ctx.document_id,
                    "moduleIdx": ctx.module_idx,
                    "advancing": advancing,
                },
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("could not fetch agent context: %s", exc)
            return None


async def post_score(ctx: SessionContext, *, paused: bool) -> None:
    """Grade whatever was said in the current section. The browser normally does
    this; the worker is the safety net for a closed tab or a dropped network, so
    a real conversation is never silently lost."""
    turns = ctx.section_turns()
    if ctx.posted or not turns:
        return
    ctx.posted = True
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await client.post(
                f"{API_BASE}/api/sessions/agent/score",
                headers=_headers(),
                json={
                    "workspaceId": ctx.workspace_id,
                    "userId": ctx.user_id,
                    "documentId": ctx.document_id,
                    "moduleIdx": ctx.module_idx,
                    "transcript": turns,
                    "paused": paused,
                },
            )
        except Exception as exc:
            logger.warning("agent score post failed: %s", exc)


def _room_meta(room: rtc.Room) -> dict:
    for raw in (room.metadata or "", ):
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
    # Fall back to the learner participant's metadata (set on their join token),
    # so a room created implicitly on join still resolves.
    for p in room.remote_participants.values():
        if p.metadata:
            try:
                return json.loads(p.metadata)
            except json.JSONDecodeError:
                continue
    return {}


async def _publish(room: rtc.Room, payload: dict) -> None:
    try:
        await room.local_participant.publish_data(
            json.dumps(payload).encode(), reliable=True, topic="praxos"
        )
    except Exception as exc:
        logger.debug("publish_data failed: %s", exc)


class TutorAgent(Agent):
    """The tutor. Its only tool is the advancement gate."""

    def __init__(self, instructions: str, ctx: SessionContext, room: rtc.Room):
        super().__init__(instructions=instructions)
        self._ctx = ctx
        self._room = room
        self._caption_seq = 0

    async def transcription_node(self, text, model_settings):  # type: ignore[override]
        """Forward the tutor's line to the browser WITH per-word timings.

        Cartesia returns word-level timestamps and ``use_tts_aligned_transcript``
        surfaces them here as ``TimedString``. Publishing them is what lets the
        subtitle highlight the word actually being spoken — the previous UI
        advanced the highlight on a fixed 3.3-words-per-second guess, so it
        drifted out of sync within a sentence.
        """
        self._caption_seq += 1
        seq = self._caption_seq
        started = False

        async def _stream():
            nonlocal started
            async for chunk in text:
                start = getattr(chunk, "start_time", None)
                end = getattr(chunk, "end_time", None)
                word = {"t": str(chunk)}
                if isinstance(start, (int, float)):
                    word["s"] = float(start)
                if isinstance(end, (int, float)):
                    word["e"] = float(end)
                asyncio.create_task(
                    _publish(
                        self._room,
                        {
                            "type": "caption",
                            "seq": seq,
                            "first": not started,
                            "word": word,
                        },
                    )
                )
                started = True
                yield chunk

        async for out in Agent.default.transcription_node(self, _stream(), model_settings):
            yield out

    @function_tool(
        name="mark_section_understood",
        description=(
            "Call ONLY when the learner has explained this section's key points in their OWN "
            "words and answered a follow-up probe testing whether they can APPLY the idea, not "
            "repeat it. Never call it after an acknowledgement, a one-word answer, silence, or "
            "the learner merely agreeing with you. If you cannot quote the learner's actual "
            "explanation, do not call this."
        ),
    )
    async def mark_section_understood(
        self,
        learner_explanation: str,
        key_points_covered: list[str],
        confidence: int,
    ) -> str:
        """Reveal the learner's 'Next section' button."""
        # The gate: a model that cannot produce the learner's own words has not
        # heard an explanation. This is what stops "yeah, got it" from passing.
        if len(learner_explanation.split()) < 6 or confidence < 60:
            return (
                "Not enough evidence to advance. Ask the learner to explain the idea in their "
                "own words, then probe with a fresh example before calling this again."
            )
        await _publish(
            self._room,
            {
                "type": "section_ready",
                "moduleIdx": self._ctx.module_idx,
                "isLast": self._ctx.is_last,
                "evidence": {
                    "explanation": learner_explanation,
                    "keyPoints": key_points_covered,
                    "confidence": confidence,
                },
            },
        )
        return (
            "The learner's 'Next section' button is now visible. Tell them in ONE short sentence "
            "that they have finished this section and can tap it when ready, then stop and wait."
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    room = ctx.room

    # The learner may still be joining — wait briefly for metadata to resolve.
    meta = _room_meta(room)
    for _ in range(20):
        if meta:
            break
        await asyncio.sleep(0.25)
        meta = _room_meta(room)

    sctx = SessionContext(meta)
    if not sctx.valid:
        logger.error("room %s has no usable praxos metadata; leaving", room.name)
        return

    bootstrap = await fetch_context(sctx)
    if not bootstrap:
        await _publish(room, {"type": "error", "message": "Could not load the lesson."})
        return
    sctx.total_modules = int(bootstrap.get("totalModules") or 0)
    sctx.is_last = bool(bootstrap.get("isLast", True))
    sctx.module_idx = int(bootstrap.get("moduleIdx", sctx.module_idx))

    stt_cfg = bootstrap.get("stt") or {}
    tts_cfg = bootstrap.get("tts") or {}

    session: AgentSession = AgentSession(
        stt=deepgram.STT(
            model=stt_cfg.get("model", "nova-3"),
            language=stt_cfg.get("language", "multi"),
            interim_results=True,
            smart_format=True,
            punctuate=True,
            # Do not transcribe silence into "Thank you." — the artifact that
            # made empty sittings look like real answers and score 10.
            filler_words=False,
        ),
        llm=openai.LLM(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL),
        tts=cartesia.TTS(
            model=tts_cfg.get("model", "sonic-2"),
            voice=tts_cfg.get("voice"),
            api_key=os.getenv("CARTESIA_API_KEY"),
        ),
        vad=silero.VAD.load(),
        # Word timings from Cartesia ride along with the transcript, so the
        # browser can highlight the word being spoken instead of guessing.
        use_tts_aligned_transcript=True,
    )

    agent = TutorAgent(bootstrap["instructions"], sctx, room)

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        item = getattr(event, "item", None)
        if item is None:
            return
        role = getattr(item, "role", "")
        text = (getattr(item, "text_content", None) or "").strip()
        if not text or role not in ("user", "assistant"):
            return
        sctx.transcript.append({"role": "learner" if role == "user" else "tutor", "text": text})

    @session.on("user_input_transcribed")
    def _on_user_text(event) -> None:
        if getattr(event, "is_final", False):
            return
        asyncio.create_task(
            _publish(room, {"type": "learner_partial", "text": getattr(event, "transcript", "")})
        )

    @session.on("agent_state_changed")
    def _on_state(event) -> None:
        asyncio.create_task(
            _publish(room, {"type": "agent_state", "state": str(getattr(event, "new_state", ""))})
        )

    async def _on_data(packet: rtc.DataPacket) -> None:
        try:
            msg = json.loads(packet.data.decode())
        except Exception:
            return
        if msg.get("type") != "advance":
            return
        # Grade the section we're leaving, then switch onto the next one in place.
        await post_score(sctx, paused=False)
        sctx.posted = False
        sctx.module_idx = int(msg.get("moduleIdx", sctx.module_idx + 1))
        nxt = await fetch_context(sctx, advancing=True)
        if not nxt:
            return
        sctx.module_idx = int(nxt.get("moduleIdx", sctx.module_idx))
        sctx.is_last = bool(nxt.get("isLast", True))
        sctx.scored_upto = len(sctx.transcript)
        await agent.update_instructions(nxt["instructions"])
        await _publish(
            room,
            {
                "type": "section_changed",
                "moduleIdx": sctx.module_idx,
                "moduleTitle": nxt.get("moduleTitle"),
                "isLast": sctx.is_last,
            },
        )
        session.generate_reply()

    @room.on("data_received")
    def _data(packet: rtc.DataPacket) -> None:
        asyncio.create_task(_on_data(packet))

    async def _flush(paused: bool = True) -> None:
        await post_score(sctx, paused=paused)

    ctx.add_shutdown_callback(lambda: _flush(paused=True))

    await session.start(
        room=room,
        agent=agent,
        room_input_options=RoomInputOptions(close_on_disconnect=True),
    )
    # Teach first — never wait for the learner to speak.
    session.generate_reply()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="praxos-tutor"))
