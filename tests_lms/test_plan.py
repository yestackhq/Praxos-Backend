from __future__ import annotations

"""Lesson-plan coverage.

The production plans did not tile their documents. A 46-chunk document was
planned as 7 sections covering 25 chunks: every boundary was off by one (the
model emitted inclusive ends) and the last 14 chunks were never taught to
anyone, because the planner prompt had been truncated before it saw them.

``_normalise_coverage`` makes that structurally impossible, and
``_plan_corpus`` makes sure the planner sees the whole document.
"""

from lms_app.ai import PLAN_MIN_EXCERPT, _normalise_coverage, _plan_corpus


def _covers(sections: list[dict], n: int) -> bool:
    """Every chunk in [0, n) belongs to at least one section, in order, with no
    gap. Overlap is permitted (see _normalise_coverage) — a skipped chunk is not."""
    seen: set[int] = set()
    prev_start = -1
    for s in sections:
        if s["chunk_end"] <= s["chunk_start"] or s["chunk_start"] < prev_start:
            return False
        prev_start = s["chunk_start"]
        seen.update(range(s["chunk_start"], s["chunk_end"]))
    return seen == set(range(n))


def test_inclusive_end_indices_are_repaired():
    """What the model actually produced for a 19-chunk document: every section
    ended one short, so 6 chunks were silently skipped."""
    raw = [
        {"chunk_start": 0, "chunk_end": 2},
        {"chunk_start": 2, "chunk_end": 5},
        {"chunk_start": 5, "chunk_end": 8},
        {"chunk_start": 8, "chunk_end": 13},
        {"chunk_start": 13, "chunk_end": 16},
        {"chunk_start": 16, "chunk_end": 18},  # leaves chunk 18 untaught
    ]
    out = _normalise_coverage([dict(s) for s in raw], 19)
    assert _covers(out, 19)
    assert out[-1]["chunk_end"] == 19


def test_a_plan_that_stops_early_is_extended_to_the_end():
    """The 46-chunk case: the plan stopped at 31 and 14 chunks were never taught."""
    raw = [
        {"chunk_start": 0, "chunk_end": 2},
        {"chunk_start": 3, "chunk_end": 8},
        {"chunk_start": 9, "chunk_end": 15},
        {"chunk_start": 16, "chunk_end": 19},
        {"chunk_start": 20, "chunk_end": 23},
        {"chunk_start": 24, "chunk_end": 29},
        {"chunk_start": 30, "chunk_end": 31},
    ]
    out = _normalise_coverage([dict(s) for s in raw], 46)
    assert _covers(out, 46)
    assert out[-1]["chunk_end"] == 46


def test_more_sections_than_chunks_share_rather_than_being_dropped():
    """MOP_Chapter_4 has 6 planned sections over 4 chunks. Every section an admin
    wrote must survive; neighbouring ones share a chunk."""
    raw = [{"chunk_start": i, "chunk_end": i + 2} for i in range(6)]
    out = _normalise_coverage([dict(s) for s in raw], 4)
    assert _covers(out, 4)
    assert len(out) == 6, "a section must never be silently dropped"


def test_garbage_indices_still_yield_a_valid_plan():
    raw = [
        {"chunk_start": "x", "chunk_end": None},
        {"chunk_start": 99, "chunk_end": -4},
        {"chunk_start": 1, "chunk_end": 1},
    ]
    out = _normalise_coverage([dict(s) for s in raw], 6)
    assert _covers(out, 6)


def test_planner_sees_every_chunk_of_a_long_document():
    """The old prompt truncated at 24k characters, so a long document's tail was
    never shown to the planner — which is why plans stopped early."""
    chunks = [f"chunk-{i} " + ("lorem ipsum " * 120) for i in range(200)]
    corpus = _plan_corpus(chunks)
    for i in (0, 99, 199):
        assert f"[chunk {i}]" in corpus
        assert f"chunk-{i} " in corpus
    # Excerpts shrink to fit rather than chunks being dropped.
    assert len(corpus) < 200 * len(chunks[0])
    assert f"[chunk 199]\n{chunks[199][:PLAN_MIN_EXCERPT]}"[:60] in corpus
