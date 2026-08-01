from __future__ import annotations

"""Teaching intelligence: lesson-plan design and understanding assessment.

Model access goes through ``llm.py`` so the provider is configurable; nothing
here names a vendor. Everything degrades to None when no provider is set, and
callers fall back or surface a 503.
"""

import logging
import re
from typing import Optional

from . import llm
from .config import settings

logger = logging.getLogger("praxos.ai")


# Back-compat re-exports — indexing.py embeds through these.
embed_texts = llm.embed_texts
embed_one = llm.embed_one


# ---- Lesson-plan design ------------------------------------------------------

# How much of each chunk the planner sees, and the total budget. The planner
# MUST see every chunk or it will plan over only the prefix it was shown and
# silently leave the tail of the document untaught — the excerpt size shrinks to
# fit rather than the chunk list being truncated.
PLAN_TOTAL_BUDGET = 90_000
# Planning sends the WHOLE document and asks for structured output over it, so it
# runs far longer than a scoring call. It gets its own budget rather than
# inheriting the general request timeout.
PLAN_TIMEOUT_SECONDS = 300.0

# Grading sends the section's key points, its source text and the whole
# transcript, then reasons over them. Against the 60s default request budget it
# timed out and the learner's finished session was lost with a 503.
SCORE_TIMEOUT_SECONDS = 240.0
PLAN_MIN_EXCERPT = 220
PLAN_MAX_EXCERPT = 900


def _plan_corpus(chunks: list[str]) -> str:
    """Every chunk, numbered, with the per-chunk excerpt sized so the whole
    document fits the budget. Long documents get shorter excerpts — never fewer
    chunks."""
    n = max(1, len(chunks))
    per = max(PLAN_MIN_EXCERPT, min(PLAN_MAX_EXCERPT, PLAN_TOTAL_BUDGET // n))
    return "\n\n".join(f"[chunk {i}]\n{c[:per]}" for i, c in enumerate(chunks))


def _normalise_coverage(sections: list[dict], n_chunks: int) -> list[dict]:
    """Force the plan to cover the whole document: ordered, contiguous, no gaps,
    ending at ``n_chunks``.

    Models routinely emit inclusive end indices, drift by one at each boundary,
    or stop early — which is how a 46-chunk document ended up with a plan that
    taught 25 chunks and never mentioned the last 14. Rather than trusting the
    ranges, we keep the model's ORDERING and section count and re-derive the
    boundaries from its proposed starts.

    Full coverage is the hard requirement; non-overlap is not. When a plan has
    more sections than the document has chunks, neighbouring sections share a
    chunk instead of sections being dropped — several angles on the same passage
    is a reasonable lesson, losing a section an admin wrote is not.
    """
    if not sections:
        return []
    if n_chunks <= 0:
        for s in sections:
            s["chunk_start"], s["chunk_end"] = 0, 0
        return sections

    ordered = (
        sorted(sections, key=lambda s: _as_int(s.get("chunk_start"), 0))
        if len(sections) > 1
        else list(sections)
    )
    k = len(ordered)
    crowded = k > n_chunks  # sections must share chunks

    starts: list[int] = []
    for i, s in enumerate(ordered):
        want = _as_int(s.get("chunk_start"), i)
        if i == 0:
            starts.append(0)
            continue
        # Strictly increasing while there is room; otherwise merely non-decreasing.
        lo = min(starts[i - 1] + (0 if crowded else 1), n_chunks - 1)
        # Leave a chunk for each later section when we are not crowded.
        hi = n_chunks - 1 if crowded else n_chunks - (k - i)
        starts.append(max(lo, min(want, max(lo, hi))))

    for i, s in enumerate(ordered):
        s["chunk_start"] = starts[i]
        # End where the next section begins, so there is never a gap; at least
        # one chunk wide; the final section always runs to the end.
        s["chunk_end"] = n_chunks if i + 1 == k else max(starts[i] + 1, starts[i + 1])
    return ordered


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def generate_lesson_plan(
    doc_name: str,
    chunks: list[str],
    *,
    end_user: Optional[llm.EndUser] = None,
    session_id: Optional[str] = None,
) -> Optional[list[dict]]:
    """Design a section-by-section teaching plan from a document's text.

    Returns an ordered list of sections — each a coherent unit a voice tutor
    teaches in one sitting:
      {title, description, topics: [str], key_points: [str], check_questions: [str],
       minutes: int, chunk_start: int, chunk_end: int}

    ``chunk_start``/``chunk_end`` index into ``chunks`` (inclusive/exclusive) and
    are guaranteed to tile [0, len(chunks)) exactly once. None when no model is
    configured or nothing usable came back.
    """
    if not chunks:
        return None
    n = len(chunks)
    system = (
        "You are an expert curriculum designer for a voice-based tutor that teaches one "
        f"section per sitting and must VERIFY understanding before moving on. The document "
        f"'{doc_name}' is split into {n} numbered chunks (0..{n - 1}).\n"
        "Design a teaching plan of 3-8 ordered SECTIONS. Requirements:\n"
        "  • The sections must cover the WHOLE document in order — every chunk from 0 to "
        f"{n - 1} belongs to exactly one section. No gaps, no overlaps, no stopping early.\n"
        "  • chunk_start is inclusive, chunk_end is EXCLUSIVE. Section 1 starts at 0; the "
        f"last section ends at {n}. Each section's chunk_start equals the previous "
        "section's chunk_end.\n"
        "  • description: 1-2 sentences on what the learner must come away UNDERSTANDING, "
        "and how to teach it.\n"
        "  • key_points: 2-5 specific claims from the material the learner must be able to "
        "state. These are what the grader checks against, so make them concrete and "
        "self-contained — not 'the main idea' but the actual idea.\n"
        "  • check_questions: 2-4 questions that force the learner to EXPLAIN or APPLY the "
        "idea, not recall a word. At least one must ask them to apply it to a new example.\n"
        "  • topics: 2-4 short topic labels. minutes: 3-8.\n"
        'Respond ONLY as JSON: {"sections": [{"title": "...", "description": "...", '
        '"topics": ["..."], "key_points": ["..."], "check_questions": ["..."], '
        '"minutes": <int>, "chunk_start": <int>, "chunk_end": <int>}]}'
    )
    data = llm.chat_json(
        system,
        _plan_corpus(chunks),
        temperature=0.2,
        end_user=end_user,
        session_id=session_id,
        timeout=PLAN_TIMEOUT_SECONDS,
    )
    if not data:
        return None

    raw = data.get("sections") or []
    cleaned: list[dict] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            continue
        cleaned.append(
            {
                "title": str(s.get("title") or f"Section {i + 1}")[:160],
                "description": str(s.get("description") or "")[:2000],
                "topics": [str(t)[:80] for t in (s.get("topics") or [])][:6],
                "key_points": [str(t)[:400] for t in (s.get("key_points") or [])][:6],
                "check_questions": [str(t)[:300] for t in (s.get("check_questions") or [])][:5],
                "minutes": max(2, min(20, _as_int(s.get("minutes"), 5))),
                "chunk_start": _as_int(s.get("chunk_start"), i),
                "chunk_end": _as_int(s.get("chunk_end"), i + 1),
            }
        )
    if not cleaned:
        return None
    return _normalise_coverage(cleaned, n)


# ---- Understanding assessment ------------------------------------------------

# A sitting with no real answer is UNSCOREABLE, not a zero. Recording it as 10
# is what dragged learners' numbers down every time they opened and closed the
# app, and it is why "the latest score" was meaningless.
_FILLER = {
    "yeah", "yes", "no", "ok", "okay", "um", "uh", "hmm", "mhm", "mm", "right",
    "sure", "nope", "yep", "what", "huh", "idk", "dunno", "the", "a", "i", "thank",
    "thanks", "you", "thankyou", "please", "hi", "hello", "hey", "bye", "got", "it",
    "cool", "nice", "fine", "good", "great", "sorry", "repeat", "again", "and", "so",
}

MIN_SUBSTANTIVE_WORDS = 6


def has_scoreable_answer(transcript: list[dict]) -> bool:
    """True when the learner said enough for a grade to mean anything."""
    learner_text = " ".join(t.get("text", "") for t in transcript if t.get("role") == "learner")
    words = [w for w in re.findall(r"[a-z0-9']+", learner_text.lower()) if w not in _FILLER]
    return len(words) >= MIN_SUBSTANTIVE_WORDS


def _section_brief(section: Optional[dict]) -> str:
    """What 'correct' means for this section — the grader is useless without it."""
    if not section:
        return (
            "No section plan was available. Judge whether the learner explained the ideas "
            "the tutor actually taught in this transcript, in their own words."
        )
    parts = [f"SECTION: {section.get('title', '')}", f"AIM: {section.get('description', '')}"]
    if section.get("topics"):
        parts.append("TOPICS: " + ", ".join(str(t) for t in section["topics"]))
    if section.get("key_points"):
        parts.append(
            "THE LEARNER MUST BE ABLE TO STATE:\n"
            + "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(section["key_points"]))
        )
    if section.get("material"):
        parts.append("SOURCE MATERIAL (ground truth):\n" + str(section["material"])[:5000])
    return "\n".join(parts)


def score_understanding(
    doc_name: str,
    transcript: list[dict],
    *,
    section: Optional[dict] = None,
    prior_facts: Optional[list[str]] = None,
    end_user: Optional[llm.EndUser] = None,
    session_id: Optional[str] = None,
) -> Optional[dict]:
    """Grade one sitting into an understanding score (0-100) with evidence.

    ``section`` carries the plan for what was being taught — title, aim, topics,
    key_points and the source ``material``. Without it the grader is guessing
    whether an answer is correct, which is what made scores erratic.

    Returns None when no model is configured. Returns ``{"scoreable": False}``
    when the learner said too little to judge — the caller must NOT record that
    as a low score.
    """
    if not transcript:
        return None
    if not has_scoreable_answer(transcript):
        return {
            "scoreable": False,
            "score": None,
            "summary": "The learner did not say enough in this sitting to assess.",
            "topics": [],
            "strengths": [],
            "gaps": [],
        }

    convo = "\n".join(
        f"{'LEARNER' if t.get('role') == 'learner' else 'TUTOR'}: {t.get('text', '')}"
        for t in transcript
    )
    prior_block = ""
    if prior_facts:
        prior_block = (
            "\n\nWHAT THIS LEARNER HAS PREVIOUSLY DEMONSTRATED (do not re-credit a phrase they "
            "are simply repeating; DO credit a genuinely deeper or newly-applied explanation):\n"
            + "\n".join(f"  • {f}" for f in prior_facts[:6])
        )

    system = (
        "You assess demonstrated understanding for a corporate learning platform. A learner "
        f"was taught a section of '{doc_name}' by a voice tutor and answered questions.\n\n"
        "GRADE ONLY THE LEARNER'S TURNS. The tutor's explanations are not evidence — a learner "
        "who only agrees with a correct explanation has demonstrated nothing. Ignore filler, "
        "mishearings and transcription noise; judge substance, not fluency or grammar. This is "
        "speech, so fragmentary phrasing is fine if the idea is right. An answer in the "
        "learner's own words or their own example is STRONGER evidence than textbook wording.\n\n"
        "Work through the key points below one at a time. For each, decide from the learner's "
        "words alone: did they state it, state it partially, get it wrong, or never address it? "
        "Quote the learner's exact words as evidence.\n\n"
        "SCORE (0-100) = how much of this section the learner has actually demonstrated:\n"
        "  90-100  every key point explained correctly, in their own words, and applied or "
        "extended to something new\n"
        "  75-89   every key point substantially correct; minor imprecision only\n"
        "  60-74   most key points correct; one gap or one real misconception\n"
        "  40-59   the general idea is there but the specifics are vague or half-wrong\n"
        "  20-39   mostly wrong, or only echoing the tutor's words back\n"
        "  0-19    no relevant content\n"
        "Use the FULL range and the exact number the evidence supports — do not round to 10, "
        "41, 65 or 70. Do not penalise a learner for points the tutor never got to: if the "
        "sitting only covered some key points, grade what was covered and say so in `covered`. "
        "Do not withhold a high score out of caution — if they explained it, credit it.\n\n"
        'Respond ONLY as JSON: {"score": <int 0-100>, '
        '"covered": <int 0-100, share of the section actually taught in this sitting>, '
        '"summary": "<one sentence, addressed to the admin>", '
        '"topics": [{"name": "<key point>", "score": <int 0-100>, "evidence": "<learner quote or '
        '\'not addressed\'>"}], "strengths": ["..."], "gaps": ["..."]}'
    )
    user = f"{_section_brief(section)}{prior_block}\n\n--- TRANSCRIPT ---\n{convo}"

    data = llm.chat_json(
        system,
        user,
        temperature=0,
        end_user=end_user,
        session_id=session_id,
        timeout=SCORE_TIMEOUT_SECONDS,
    )
    if not data:
        return None
    score = data.get("score")
    if score is None:
        return None
    data["scoreable"] = True
    data["score"] = max(0, min(100, _as_int(score, 0)))
    data["covered"] = max(0, min(100, _as_int(data.get("covered"), 100)))
    return data


# ---- Document-level rollup ---------------------------------------------------


def document_score(section_scores: dict[int, int], plan_weights: dict[int, int]) -> Optional[int]:
    """Roll per-section scores up into one number for a document.

    ``section_scores`` maps module_idx -> the learner's BEST score on that
    section; ``plan_weights`` maps module_idx -> that section's minutes (its
    share of the document). Sections never attempted count as 0, so a document
    only reaches a high score once the whole thing has been demonstrated —
    while a single bad sitting can no longer erase sections already mastered.

    None when the learner has attempted nothing.
    """
    if not section_scores or not plan_weights:
        return None
    total_weight = sum(max(1, w) for w in plan_weights.values())
    if total_weight <= 0:
        return None
    earned = sum(
        max(1, plan_weights.get(idx, 1)) * max(0, min(100, score))
        for idx, score in section_scores.items()
        if idx in plan_weights
    )
    return round(earned / total_weight)
