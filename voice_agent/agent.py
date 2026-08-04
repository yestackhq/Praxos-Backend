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
from livekit.agents import (
    APIConnectOptions,
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
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
        # Sections the learner advanced past whose grade the browser has NOT
        # yet confirmed ({"module_idx", "turns"}). The browser's advance-time
        # grade post is best-effort; twice now it silently failed and a real
        # conversation vanished. The worker keeps the segment until a "scored"
        # confirmation arrives, and grades whatever is still here at shutdown.
        self.unconfirmed: list[dict] = []

    @property
    def valid(self) -> bool:
        return bool(self.user_id and self.document_id)

    def section_turns(self) -> list[dict]:
        return self.transcript[self.scored_upto :]


def _headers() -> dict:
    return {"X-Agent-Secret": AGENT_SECRET, "Content-Type": "application/json"}


async def fetch_context(
    ctx: SessionContext, *, advancing: bool = False, locate: bool = False
) -> Optional[dict]:
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
                    # A replacement worker (dispatched by a reconnect after the
                    # previous one died) carries the section index the SITTING
                    # started on, not where the learner actually is. `locate`
                    # asks the API to resolve the real position from the DB.
                    "locate": locate,
                },
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("could not fetch agent context: %s", exc)
            return None


async def post_turns(
    ctx: SessionContext, *, module_idx: int, turns: list[dict], paused: bool
) -> None:
    """Grade one section's turns via the agent-authenticated endpoint."""
    if not turns:
        return
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await client.post(
                f"{API_BASE}/api/sessions/agent/score",
                headers=_headers(),
                json={
                    "workspaceId": ctx.workspace_id,
                    "userId": ctx.user_id,
                    "documentId": ctx.document_id,
                    "moduleIdx": module_idx,
                    "transcript": turns,
                    "paused": paused,
                },
            )
        except Exception as exc:
            logger.warning("agent score post failed: %s", exc)


async def post_score(ctx: SessionContext, *, paused: bool) -> None:
    """Grade whatever was said in the current section, plus any earlier section
    whose browser-side grade was never confirmed. The browser normally grades;
    the worker is the safety net for a closed tab, a dropped network — or a
    grade request that silently died, so a real conversation is never lost."""
    for seg in ctx.unconfirmed:
        logger.warning(
            "grading section %s from the safety net — the browser never confirmed it",
            seg["module_idx"],
        )
        await post_turns(ctx, module_idx=seg["module_idx"], turns=seg["turns"], paused=False)
    ctx.unconfirmed = []
    # The current section: skipped when the browser said it is grading it
    # itself (the "ending" message) — but the unconfirmed drain above runs
    # regardless, because those segments are exactly the ones the browser
    # failed to grade.
    turns = ctx.section_turns()
    if ctx.client_scored or ctx.posted or not turns:
        return
    ctx.posted = True
    await post_turns(ctx, module_idx=ctx.module_idx, turns=turns, paused=paused)


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


def _handoff_block(turns: list[dict], max_chars: int = 2000) -> str:
    """The LEARNER's words from the finished section, carried into the next
    section's instructions. The recap fetched from the memory service cannot
    contain them — grading and memory ingestion are still running when the swap
    happens — so the worker, which holds the turns, hands them over directly.

    Learner turns ONLY. Including the tutor's side put its own closing line
    ("you have finished this section, tap the button") at the very end of the
    new instructions — and the model opened the NEXT section by parroting it,
    which looked exactly like being stuck on the finished section."""
    lines = [
        f"LEARNER: {t.get('text', '').strip()}"
        for t in turns
        if t.get("role") == "learner" and t.get("text", "").strip()
    ]
    if not lines:
        return ""
    tail: list[str] = []
    used = 0
    for ln in reversed(lines):
        if used + len(ln) > max_chars:
            break
        tail.append(ln)
        used += len(ln) + 1
    if not tail:
        return ""
    return (
        "\n\n--- WHAT THE LEARNER SAID IN THE PREVIOUS SECTION (carry it forward) ---\n"
        + "\n".join(reversed(tail))
        + "\nEverything demonstrated above is settled — build on it, never ask for it again. "
        "The previous section is CLOSED: do not say it is finished, do not mention any "
        "button. Open by teaching THIS section."
    )


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

        # 2. The browser grades the section being left (it holds the learner's
        #    token, so that call is verified). But its post is not trusted until
        #    a "scored" confirmation arrives — the leaving segment is captured
        #    below into `unconfirmed`, and the safety net grades it at shutdown
        #    if the confirmation never came. Trusting the browser at THIS point
        #    is how two real conversations vanished when its post silently died.
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

        # The section being left, captured BEFORE the boundary moves: handed to
        # the next section's instructions so cross-section carry-over does not
        # depend on the memory service having ingested a grade that is still
        # being computed — and kept in `unconfirmed` until the browser confirms
        # its grade post actually succeeded.
        leaving_turns = sctx.section_turns()
        handoff = _handoff_block(leaving_turns)
        if leaving_turns:
            sctx.unconfirmed.append({"module_idx": previous_idx, "turns": leaving_turns})
            del sctx.unconfirmed[:-4]  # bound the buffer; confirmations arrive in seconds

        sctx.module_idx = int(nxt.get("moduleIdx", target))
        sctx.is_last = bool(nxt.get("isLast", True))
        sctx.total_modules = int(nxt.get("totalModules") or sctx.total_modules)
        sctx.scored_upto = len(sctx.transcript)
        sctx.posted = False
        # Re-arm the shutdown safety net for the NEW section. client_scored
        # covers only the turns the browser graded when it asked to advance;
        # leaving it True meant everything said after the first advance was
        # discarded if the sitting ended without a clean goodbye.
        sctx.client_scored = False

        await agent.update_instructions(nxt["instructions"] + handoff)
        await (publish or _publish)(
            room,
            {
                "type": "section_changed",
                "moduleIdx": sctx.module_idx,
                "moduleTitle": nxt.get("moduleTitle"),
                "isLast": sctx.is_last,
            },
        )
        # 3. Only now speak, with the new section's instructions in place. The
        #    per-reply directive outweighs the tail of the chat history — the
        #    last thing there is the OLD section's "tap the button" closing
        #    line, and without this the model has opened the new section by
        #    repeating it.
        #    tool_choice="none": the opener must be SPEECH. A model that spent
        #    this reply on a silent mark_section_understood call left the
        #    learner in a dead room saying "hello?" — a section cannot be
        #    understood before it has been taught.
        opener_instructions = (
            "The learner just moved to the new section. In one sentence recap what "
            "the previous section established, then teach this section's first key "
            "point and ask one question. Do not say any section is finished and do "
            "not mention any button."
        )
        swap_len = len(sctx.transcript)
        session.generate_reply(instructions=opener_instructions, tool_choice="none")

        # 4. If the opener never becomes audible — a playout glitch or a lost
        #    model reply has left learners in a silent room more than once —
        #    retry it once rather than sitting idle while they say "hello?".
        async def _opener_watchdog() -> None:
            await asyncio.sleep(10)
            spoke = any(
                t.get("role") == "tutor" for t in sctx.transcript[swap_len:]
            ) or getattr(session, "current_speech", None) is not None
            if not spoke:
                logger.warning("section opener silent 10s after swap; retrying generate_reply")
                try:
                    session.generate_reply(
                        instructions=opener_instructions, tool_choice="none"
                    )
                except Exception as exc:
                    logger.warning("opener retry failed: %s", exc)

        asyncio.create_task(_opener_watchdog())


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
        confidence: float,
    ) -> str:
        """Reveal the learner's 'Next section' button."""
        # Typed float, not int. The parameter was documented as 0-100 but typed
        # int, and the model answered 0.95 — the natural reading of "confidence".
        # Pydantic rejected the call, so the tool never ran, section_ready was
        # never published, and the learner sat looking at "End session" while the
        # tutor told them to move on. Accept both conventions.
        score = float(confidence)
        if score <= 1.0:
            score *= 100.0
        # The gate: a model that cannot produce the learner's own words has not
        # heard an explanation. This is what stops "yeah, got it" from passing.
        # A low floor on purpose. This exists only to stop the tool firing on
        # "yeah, ok" — not to second-guess the tutor. Set high it became the thing
        # that kept learners pinned to section one, which costs far more than
        # advancing someone a little early.
        if len(learner_explanation.split()) < 4 or score < 40:
            return (
                "Not enough evidence yet. Ask them to say the idea in their own words, then "
                "call this again — one clear answer is enough."
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
                    "confidence": round(score),
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

    # locate=True: this worker may be a REPLACEMENT — dispatched by a browser
    # reconnecting after the previous worker died mid-sitting (laptop sleep).
    # The metadata's moduleIdx is where the sitting STARTED; the API resolves
    # where the learner actually is now.
    bootstrap = await fetch_context(sctx, locate=True)
    if not bootstrap:
        await _publish(room, {"type": "error", "message": "Could not load the lesson."})
        return
    sctx.total_modules = int(bootstrap.get("totalModules") or 0)
    sctx.is_last = bool(bootstrap.get("isLast", True))
    sctx.module_idx = int(bootstrap.get("moduleIdx", sctx.module_idx))

    stt_cfg = bootstrap.get("stt") or {}
    tts_cfg = bootstrap.get("tts") or {}

    # The gateway intermittently answers 502 UPSTREAM_UNAVAILABLE. The defaults
    # here are tuned for a fast chat model: three tries at 0.1s/2s/2s, and a 10s
    # connect timeout that a reasoning model routinely exceeds on a long turn.
    # Both were making the tutor fall silent mid-lesson. Widen the window rather
    # than let a transient upstream blip end the conversation.
    conn = SessionConnectOptions(
        llm_conn_options=APIConnectOptions(max_retry=5, retry_interval=2.0, timeout=90.0),
        tts_conn_options=APIConnectOptions(max_retry=4, retry_interval=1.0, timeout=30.0),
        stt_conn_options=APIConnectOptions(max_retry=4, retry_interval=1.0, timeout=30.0),
    )

    session: AgentSession = AgentSession(
        conn_options=conn,
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
        # Learners give long spoken answers. The default VAD buffer keeps 60s of
        # one continuous speech segment and then DROPS the rest ("max_buffered_
        # speech reached") — a learner explaining at length was truncated
        # mid-answer and the turn went unresponsive. Three minutes of 16kHz
        # mono is ~11MB; memory is the cheap side of this trade.
        vad=silero.VAD.load(max_buffered_speech=180.0),
        # NO turn_handling override. Setting {"endpointing": {"min_delay": 0.8}}
        # here stopped live transcription reaching the room: the browser's
        # transcript stayed empty for a whole sitting, so the section grade it
        # posts on advance had nothing in it and a real conversation was lost
        # unscored. The library's late-final warning is cosmetic by comparison.
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
        elif kind == "scored":
            # The browser's advance-time grade post SUCCEEDED for this section —
            # only now does the worker drop its copy of those turns.
            confirmed = int(msg.get("moduleIdx", -1))
            sctx.unconfirmed = [s for s in sctx.unconfirmed if s["module_idx"] != confirmed]

    @room.on("data_received")
    def _data(packet: rtc.DataPacket) -> None:
        asyncio.create_task(_on_data(packet))

    async def _flush(paused: bool = True) -> None:
        """Grade whatever the browser did not: every unconfirmed earlier
        section, and the current one unless the browser claimed it ("ending").
        The double-grading guard lives inside post_score, so an unconfirmed
        segment is never skipped just because the final section was claimed."""
        await post_score(sctx, paused=paused)

    ctx.add_shutdown_callback(lambda: _flush(paused=True))

    await session.start(
        room=room,
        agent=agent,
        room_input_options=RoomInputOptions(close_on_disconnect=True),
    )
    # Announce which section this worker is teaching. The browser uses it to
    # notice a REPLACEMENT agent (reconnect after the previous worker died) and
    # re-sync its section state instead of assuming continuity.
    await _publish(
        room,
        {
            "type": "agent_ready",
            "moduleIdx": sctx.module_idx,
            "moduleTitle": bootstrap.get("moduleTitle"),
            "totalModules": sctx.total_modules,
            "isLast": sctx.is_last,
        },
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
