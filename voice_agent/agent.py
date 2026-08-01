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
import uuid
from typing import Any, Optional

import httpx
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, RoomInputOptions, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("praxos.agent")

API_BASE = os.getenv("PRAXOS_API_BASE", "http://localhost:8000").rstrip("/")
AGENT_SECRET = os.getenv("AGENT_SHARED_SECRET", "")

# Model access. When MeldOS is configured the tutor's turn-by-turn reasoning —
# by far the largest share of model spend — goes through the gateway, so it is
# metered per application and per person like every other call.
MELDOS_API_BASE_URL = (os.getenv("MELDOS_API_BASE_URL") or "").strip().rstrip("/")
if MELDOS_API_BASE_URL and "://" not in MELDOS_API_BASE_URL:
    MELDOS_API_BASE_URL = f"https://{MELDOS_API_BASE_URL}"
MELDOS_APPLICATION_KEY = os.getenv("MELDOS_APPLICATION_KEY", "")
MELDOS_MODEL = os.getenv("MELDOS_MODEL", "company-chat-model")
MELDOS_ENABLED = bool(MELDOS_API_BASE_URL and MELDOS_APPLICATION_KEY)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


def _llm(learner_name: str, session_id: str = ""):
    """The tutor's LLM, pointed at MeldOS when configured.

    Attribution is CLAIMED (``X-End-User-Id``), by the learner's name: the worker
    authenticates to the API with a service secret and never holds the learner's
    own sign-in token, so it has nothing to make a verified claim with. The
    header is only ever set on the MeldOS client — never on the direct-provider
    fallback below.
    """
    if MELDOS_ENABLED:
        # X-Session-ID is required by MeldOS; the LiveKit room is exactly the
        # unit of work, so every turn of one lesson groups under one session.
        headers = {"X-Session-ID": session_id or f"praxos-{uuid.uuid4()}"}
        if learner_name:
            headers["X-End-User-Id"] = learner_name
        kwargs: dict[str, Any] = {"extra_headers": headers}
        return openai.LLM(
            model=MELDOS_MODEL,
            api_key=MELDOS_APPLICATION_KEY,
            base_url=f"{MELDOS_API_BASE_URL}/v1",
            **kwargs,
        )
    return openai.LLM(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


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
        # Serialises advances: two taps, or a tap arriving while the previous
        # swap is still in flight, must not interleave and leave the worker
        # teaching one section while the UI shows another.
        self.advancing: asyncio.Lock = asyncio.Lock()
        # True once the BROWSER has taken responsibility for grading everything
        # up to `scored_upto`. The worker only grades what the browser did not,
        # so a section is never scored twice (and never billed twice).
        self.client_scored: bool = False

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


def _room_meta(room: rtc.Room, job_metadata: str = "") -> dict:
    # Preferred source: the dispatch request that put this worker in the room.
    # It is set by the API on the learner's token and arrives before the room or
    # participant metadata has necessarily replicated.
    for raw in (job_metadata, room.metadata or ""):
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


async def advance_section(
    target: int,
    *,
    sctx: SessionContext,
    session,
    agent,
    room,
    fetch=None,
    publish=None,
) -> None:
    """Move the live session onto section ``target``.

    Ordering matters and is the whole fix. Previously this swapped
    instructions and called generate_reply() without stopping the turn
    already in progress: update_instructions only affects the NEXT inference,
    so the tutor carried on speaking about the section the learner had just
    left, and the new turn queued behind it. Whether that happened at all
    depended on whether the tutor happened to be mid-sentence when the button
    was tapped — which is exactly why it was intermittent.
    """
    async with sctx.advancing:
        if target <= sctx.module_idx:
            return  # stale or duplicate tap
        if sctx.total_modules and target >= sctx.total_modules:
            await (publish or _publish)(room, {"type": "course_complete", "moduleIdx": sctx.module_idx})
            return

        # 1. Stop the current turn FIRST, so nothing from the old section is
        #    still being spoken over the new one.
        try:
            await session.interrupt()
        except Exception as exc:
            logger.debug("interrupt before advance failed: %s", exc)

        # 2. The browser grades the section being left (it holds the
        #    learner's token, so that call is attributed to them as
        #    verified). Record that so the worker's safety net does not
        #    grade the same turns a second time.
        sctx.client_scored = True

        previous_idx = sctx.module_idx
        sctx.module_idx = target
        nxt = await (fetch or fetch_context)(sctx, advancing=True)

        if nxt is None or nxt.get("complete"):
            # Roll back and TELL the browser. Returning silently left the UI
            # waiting on a section change that would never arrive, with the
            # button already hidden — stuck on the previous section with no
            # way out.
            sctx.module_idx = previous_idx
            await (publish or _publish)(
                room,
                {
                    "type": "course_complete" if (nxt or {}).get("complete") else "advance_failed",
                    "moduleIdx": previous_idx,
                },
            )
            return

        sctx.module_idx = int(nxt.get("moduleIdx", target))
        sctx.is_last = bool(nxt.get("isLast", True))
        sctx.total_modules = int(nxt.get("totalModules") or sctx.total_modules)
        sctx.scored_upto = len(sctx.transcript)
        sctx.posted = False

        await agent.update_instructions(nxt["instructions"])
        await (publish or _publish)(
            room,
            {
                "type": "section_changed",
                "moduleIdx": sctx.module_idx,
                "moduleTitle": nxt.get("moduleTitle"),
                "isLast": sctx.is_last,
            },
        )
        # 3. Only now speak, with the new section's instructions in place.
        session.generate_reply()


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
    job_meta = getattr(getattr(ctx, "job", None), "metadata", "") or ""
    meta = _room_meta(room, job_meta)
    for _ in range(20):
        if meta:
            break
        await asyncio.sleep(0.25)
        meta = _room_meta(room, job_meta)

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
            # Deepgram finalises a segment after this much silence. The default
            # of 25ms splits ordinary speech mid-sentence: a learner saying
            # "...talking about. Doesn't make sense." was recorded as two turns,
            # "Doesn't" and "sense." That inflates the answer count and hands the
            # grader broken fragments instead of what the learner actually said.
            endpointing_ms=400,
        ),
        llm=_llm(str(bootstrap.get("learnerName") or ""), session_id=room.name),
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
        kind = msg.get("type")
        if kind == "advance":
            await advance_section(
                int(msg.get("moduleIdx", sctx.module_idx + 1)),
                sctx=sctx,
                session=session,
                agent=agent,
                room=room,
            )
        elif kind == "ending":
            # The browser is grading the final section itself and then leaving;
            # the shutdown safety net must not grade it again.
            sctx.client_scored = True

    @room.on("data_received")
    def _data(packet: rtc.DataPacket) -> None:
        asyncio.create_task(_on_data(packet))

    async def _flush(paused: bool = True) -> None:
        """Grade the current section if — and only if — the browser did not.

        Without this check the browser's score and the worker's safety net both
        graded the same turns: two sittings recorded, and two billed model calls
        for one conversation."""
        if sctx.client_scored:
            return
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
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="praxos-tutor",
            # The worker serves its own health endpoint at "/" on this port. Bind
            # it to the platform's PORT so the deployment healthcheck has
            # something to hit — a worker that dies otherwise looks healthy.
            port=int(os.getenv("PORT", "8081")),
        )
    )
