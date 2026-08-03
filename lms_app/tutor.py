from __future__ import annotations

"""The voice tutor's instructions.

Built in one place and used by both the LiveKit agent worker (which teaches) and
the assessor (which grades), so what the tutor was told to check and what the
grader checks against can never drift apart.

The previous prompt asked the tutor to be "warm but RIGOROUS" and left the rest
to the model's judgement. In practice it accepted the first plausible-sounding
sentence, said "exactly!", and moved on — which is why a learner could talk for
thirty seconds and be marked as understanding a section. The rules below replace
exhortation with structure: named key points, a required probe per point, an
explicit ban on crediting the learner for the tutor's own words, and an
advancement tool that cannot be called without citing evidence.
"""

from typing import Optional

MAX_CONTEXT_CHARS = 6000


def section_material(chunks: list[str]) -> str:
    return "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]


def replay_tail(transcript: list[dict], max_chars: int = 2500) -> str:
    """The tail of an interrupted sitting's conversation, formatted for the
    tutor's instructions. Keeps the most recent turns that fit — the end of the
    conversation is where 'continue from where you left off' lives."""
    lines = [
        f"{'LEARNER' if t.get('role') == 'learner' else 'TUTOR'}: {t.get('text', '').strip()}"
        for t in transcript
        if t.get("text", "").strip()
    ]
    tail: list[str] = []
    used = 0
    for ln in reversed(lines):
        if used + len(ln) > max_chars:
            break
        tail.append(ln)
        used += len(ln) + 1
    return "\n".join(reversed(tail))


# The tutor calls this ONLY once the learner has demonstrated the section. It takes
# arguments on purpose: a model that must quote the learner's own explanation and
# name which key points were covered cannot fire the tool on "yeah, got it" — and
# the arguments are stored as the audit trail for why a section was passed.
ADVANCE_TOOL = {
    "type": "function",
    "name": "mark_section_understood",
    "description": (
        "Call this ONLY when the learner has explained this section's key points in their "
        "OWN WORDS and answered at least one follow-up probe that tested whether they can "
        "apply the idea rather than repeat it. Never call it after an acknowledgement, a "
        "one-word answer, silence, or the learner simply agreeing with something you said. "
        "If you cannot fill in `learner_explanation` with the learner's actual words, you "
        "must not call this."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "learner_explanation": {
                "type": "string",
                "description": "The learner's own explanation, quoted or closely paraphrased.",
            },
            "key_points_covered": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Which of this section's key points the learner demonstrated.",
            },
            "confidence": {
                # number, not integer: models answer 0.95 as readily as 95, and a
                # strict integer type made the whole tool call fail validation.
                # The worker accepts either and normalises.
                "type": "number",
                "description": (
                    "How confident you are that they genuinely understand it. "
                    "Either 0-100 or 0-1; both are accepted."
                ),
            },
        },
        "required": ["learner_explanation", "key_points_covered", "confidence"],
    },
}


_CORE = """You are Praxos, a voice tutor. You are teaching '{doc}'.

YOU ARE SPEAKING OUT LOUD — read this first
Everything you produce is converted straight to speech and played to the learner.
There is no screen. So:
  • Write plain spoken sentences, the way a person talks. Nothing else.
  • NO markdown, NO bullet points, NO numbered lists, NO headings, NO bold or quotes.
  • NO brackets, parentheses, arrows, colons-as-labels, code, URLs, emoji or symbols.
    A line like "Engineering: the artifact -> value" is read aloud literally and is
    gibberish to the ear.
  • Say numbers and short forms as you would speak them: "twenty four hours", not "24h".
  • If you would normally structure something as a list, say it as a sentence instead:
    "There are two parts here — what it does, and who it's for."

KEEP IT SHORT. Two or three sentences, then stop and let them talk. That is a hard
limit, not a guideline. A long turn is unlistenable: by the time you reach your point
the learner has lost the beginning. Never deliver a paragraph, an essay, a worked
example with steps, or a summary of everything so far. One idea, one question, hand back.

HOW YOU TEACH
Explain one idea in your own words, then ask something that makes the learner put it in
their own words. Never lecture for several turns in a row.

Ground everything in the SECTION MATERIAL below; use general knowledge only for everyday
analogies, never to add facts that contradict it. You always have the material you need —
never say a section is missing, and never ask the learner to paste, type, upload or share
anything. This is a voice call; they cannot send you text.

HOW YOU CHECK — be generous, not exhaustive
Your job is to leave the learner able to explain the idea, then MOVE ON. You are not
examining them. Erring towards advancing costs almost nothing; trapping someone on a
section they already understand loses them entirely.

1. Filler is not an answer. "Yeah", "ok", "got it", silence, a single word, or stray
   background speech: don't say "exactly", don't advance, just ask them to say it in
   their own words.
2. ONE good answer is enough. If they explain the gist of this section in their own
   words — even roughly, even partially — that is the bar. Credit every key point their
   answer touches, all at once. Small imprecision is fine; you are not marking an exam.
3. Two or three exchanges should finish a section. If you are about to ask a fourth
   question, stop and advance instead.
4. NEVER re-ask something they already answered, here or in what you know about them
   below. Say you remember it and move on.
5. If they say they have already covered it, ask to move on, or sound impatient — accept
   that immediately and advance. Do not argue, do not re-test, do not explain why you
   were asking.
6. When in doubt, ADVANCE. They can always revisit the section.
7. When they are plainly wrong, correct it warmly in one sentence and give them one more
   go. If they are still off, explain it simply and advance anyway.
8. To advance, call `mark_section_understood` and in the same turn say one short sentence
   telling them they have finished this section and can tap the button. Never call it
   silently.

"""


def build_instructions(
    *,
    doc_name: str,
    sections: list[dict],
    idx: int,
    material: str,
    recap: str = "",
    resumed: bool = False,
    advancing: bool = False,
    replay: str = "",
) -> str:
    """The full instruction block for one section.

    ``sections`` is the plan: [{title, description, topics, key_points,
    check_questions}]. ``material`` is this section's source text.
    """
    total = len(sections)
    cur: Optional[dict] = sections[idx] if 0 <= idx < total else None

    # The recap is rendered independently of the opening. It used to be attached
    # only to the "returning learner" opening, so advancing a section mid-session
    # dropped it entirely — the tutor started the next section knowing nothing
    # about what the learner had just demonstrated, and asked for it again.
    recap_block = f"\n{recap}\n" if recap else ""

    if advancing:
        opening = (
            "\nThe learner just finished the previous section. In ONE sentence recap what it "
            "covered, then start teaching THIS section. Do not greet or re-introduce yourself. "
            "Carry forward everything they have already demonstrated — anything they explained "
            "well a moment ago is settled, and asking for it again wastes their time.\n"
        )
    elif recap:
        opening = (
            "\nYou have taught this learner before. Do NOT introduce yourself. Give a ONE-line "
            "recap of where they left off, then continue teaching this section.\n"
        )
    else:
        opening = (
            "\nStart teaching THIS section directly — no self-introduction, no summary of the "
            "document. Open with the first key point and a question.\n"
        )

    outline = ""
    if total:
        lines = "\n".join(
            f"  {i + 1}. {s.get('title', '')}" + ("   <- teaching now" if i == idx else "")
            for i, s in enumerate(sections)
        )
        outline = f"\n--- COURSE OUTLINE ({total} sections) ---\n{lines}\n"

    section_block = ""
    if cur is not None:
        section_block = f"\n--- SECTION {idx + 1} OF {total}: {cur.get('title', '')} ---\n"
        if cur.get("description"):
            section_block += f"Aim: {cur['description']}\n"
        key_points = cur.get("key_points") or []
        if key_points:
            section_block += (
                "The learner must be able to state each of these, in their own words, before "
                "you may advance:\n"
                + "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(key_points))
                + "\n"
            )
        elif cur.get("topics"):
            section_block += "Make sure to cover: " + ", ".join(str(t) for t in cur["topics"]) + "\n"
        checks = cur.get("check_questions") or []
        if checks:
            section_block += (
                "Use these checks (rephrase them naturally; at least one must be a transfer "
                "probe):\n" + "\n".join(f"  - {q}" for q in checks) + "\n"
            )
        section_block += "Teach ONLY this section now.\n"
        if idx >= total - 1 and total:
            section_block += (
                "This is the FINAL section: once they have demonstrated it, wrap the whole "
                "document up in a sentence or two, then call `mark_section_understood`.\n"
            )
        if resumed and replay:
            section_block += (
                "\n--- THE INTERRUPTED CONVERSATION (this section, where you left off) ---\n"
                f"{replay}\n"
                "That conversation was cut off — continue it exactly where it stopped. "
                "Acknowledge the interruption in one short sentence, then pick up from the last "
                "unanswered point. Anything the learner already answered above is settled; never "
                "ask for it again.\n"
            )
        elif resumed:
            section_block += (
                "The learner PAUSED partway through this section last time. Recall from the "
                "recap where you left off, briefly reorient them, and CONTINUE from there — "
                "do not restart the section.\n"
            )

    return (
        _CORE.format(doc=doc_name)
        + "\n"
        + recap_block
        + opening
        + outline
        + section_block
        + f"\n--- SECTION MATERIAL ---\n{material}"
    )
