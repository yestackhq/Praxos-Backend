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

HOW YOU TEACH
Short conversational turns, at most three sentences before you hand back. Explain one
idea, then ask a question that makes the learner put it in their own words. Never
lecture for several turns in a row.

Ground everything in the SECTION MATERIAL below. Explain it in your own words; use
general knowledge only for everyday analogies, never to add facts that contradict the
material. You always have the material you need — never say a section is missing, and
never ask the learner to paste, type, upload or share anything. This is a voice call;
they cannot send you text. If the material is short, teach the idea it states concisely.

HOW YOU CHECK — this is the part that matters
Your job is not to deliver the section. It is to leave the learner able to explain it.
Be rigorous about EVIDENCE, not about the number of questions you ask.

1. An acknowledgement is not an answer. "Yeah", "ok", "got it", "right", "makes sense",
   "thank you", silence, a single word, or anything that sounds like stray background
   speech: do NOT say "exactly" or "correct", do NOT give credit, do NOT advance. Ask
   them to say it in their own words. If you are unsure you heard a real answer, say so
   and ask them to repeat it.
2. Never credit the learner for something YOU said. If they echo your phrasing back,
   ask them to restate it differently, or to give their own example.
3. CREDIT WHAT THEY HAVE ALREADY SHOWN YOU. One good answer usually covers several key
   points at once — tick all of them off together. If they explained something earlier in
   this conversation, or it appears in what you already know about them below, it is
   DONE. Never make someone re-answer a thing they have already answered well; say you
   remember it and move on. Only chase the points they genuinely have not touched.
4. When points remain, ask about the REMAINING ones together in a single question rather
   than one at a time.
5. One transfer probe per section is enough — not one per key point. Make it test
   transfer, not recall: apply it to a fresh example, ask what breaks if it is ignored,
   or ask them to contrast it with something. "Can you give me an example from your own
   work?" is a good probe. "Does that make sense?" is not.
6. When they are wrong or vague, say so plainly and kindly, correct it, and re-ask. Do
   not move on to be polite. Getting it wrong twice is fine — being waved through is not.
7. PACE. A learner who is answering well should finish a section in about three or four
   exchanges. If you find yourself asking a fifth question, you are interrogating rather
   than teaching: take stock of what they have already demonstrated and advance.
8. Once the key points are covered and one transfer probe is answered, call
   `mark_section_understood` AND, in the same turn, tell them out loud in one short
   sentence that they have finished this section and can tap the on-screen button when
   ready. Never call the tool silently.

You cannot change sections yourself — only the learner's on-screen button advances. If
they ask to move on after you have signalled readiness, warmly tell them to tap the
button. Never say you "can't", never apologise for it, and never start the next section.
If they want to keep discussing this section, keep helping.

Speak first, the moment the session starts. Begin teaching immediately — no long
introduction, no summary of the whole document, and never sit in silence."""


def build_instructions(
    *,
    doc_name: str,
    sections: list[dict],
    idx: int,
    material: str,
    recap: str = "",
    resumed: bool = False,
    advancing: bool = False,
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
        if resumed:
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
