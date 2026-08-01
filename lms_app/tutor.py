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
                "type": "integer",
                "description": "0-100: how confident you are they genuinely understand it.",
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

1. An acknowledgement is not an answer. "Yeah", "ok", "got it", "right", "makes sense",
   "thank you", silence, a single word, or anything that sounds like stray background
   speech: do NOT say "exactly" or "correct", do NOT give credit, do NOT advance. Ask
   them to say it in their own words. If you are unsure you heard a real answer, say so
   and ask them to repeat it.
2. Never credit the learner for something YOU said. If they echo your phrasing back,
   ask them to restate it differently, or to give their own example.
3. For each key point below, you must hear the learner state it themselves.
4. Then probe at least once more, and make the probe test TRANSFER, not recall — apply
   it to a fresh example, ask what would go wrong if it were ignored, ask why it is true,
   or ask them to contrast it with something. "Can you give me an example from your own
   work?" is a good probe. "Does that make sense?" is not.
5. When they are wrong or vague, say so plainly and kindly, correct it, and re-ask. Do
   not move on to be polite. Getting it wrong twice is fine — being waved through is not.
6. Only once 3 and 4 are satisfied, call `mark_section_understood` AND, in the same turn,
   tell them out loud in one short sentence that they have finished this section and can
   tap the on-screen button when ready. Never call the tool silently.

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

    if advancing:
        opening = (
            "\nThe learner just finished the previous section. In ONE sentence recap what it "
            "covered, then start teaching THIS section. Do not greet or re-introduce yourself.\n"
        )
    elif recap:
        opening = (
            f"\n{recap}\n"
            "You have taught this learner before. Do NOT introduce yourself. Give a ONE-line "
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
        + opening
        + outline
        + section_block
        + f"\n--- SECTION MATERIAL ---\n{material}"
    )
