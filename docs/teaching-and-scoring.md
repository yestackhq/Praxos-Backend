# How documents are indexed, taught and scored

Written against the live `praxos_lms` schema (13 documents, 63 plan sections,
128 sittings, 17 people) as it stood before the `e3a91c7b40d5` migration.

---

## 1. Indexing: healthy

`indexing.index_document` extracts PDF text with pypdf, packs it into ~1200-char
chunks on paragraph boundaries, embeds each chunk, and stores it with a pgvector
column. In production every document was fully chunked and **every chunk had an
embedding** — 104/104. There was nothing wrong here.

One naming trap, now fixed: the chunk count was stored in `documents.sections`,
so a 46-chunk book displayed as a "46-section course" while its actual teaching
plan had 7 sections. The column is now `documents.chunk_count`, and `sections`
in the API means the number of teaching sections.

## 2. Lesson plans: silently incomplete

`ai.generate_lesson_plan` asked the model to split a document into 3-8 sections
and return a `chunk_start`/`chunk_end` range per section. Two defects compounded:

**The planner never saw the whole document.** The prompt was built as
`"\n\n".join(f"[chunk {i}]\n{c[:900]}" ...)[:24000]` — chunks truncated to 900
characters, then the *whole corpus* cut at 24 000. At ~900 chars/chunk that is
about 26 chunks. Anything past that was invisible to the planner.

**Nothing verified that the plan covered the document.** Each section's range
was clamped individually; no check that the sections tiled `[0, n)`.

The result, measured on the live data:

| Document | Chunks | Plan covers | Never taught |
|---|---|---|---|
| Business_model_generation… | 46 | chunks 0–30, with gaps | **21 chunks (46%)** |
| Engineering vs Product | 19 | 0–17 | 1 chunk |
| MOP_Chapter_9 | 3 | 0–1 | 1 chunk |
| MOP_Chapter_2 | 4 | overlapping ranges | — (taught twice) |

The 46-chunk document's plan reads `[0,2) [3,8) [9,15) [16,19) [20,23) [24,29)
[30,31)`: the model emitted **inclusive** end indices, so one chunk was dropped
at every boundary, and it stopped at 31 because it had never been shown chunks
32–45. Roughly a third of that document was never taught to anyone.

**Fixed by** `_plan_corpus` (shrinks the per-chunk excerpt to fit the budget so
every chunk is always shown) and `_normalise_coverage` (keeps the model's
ordering and re-derives boundaries so the plan tiles the document exactly once).
`GET /api/documents/{id}/coverage` now reports coverage, and `tests_lms/test_plan.py`
pins all four failure shapes above.

## 3. Teaching: the tutor had no bar to clear

The old prompt asked for rigour in prose ("warm but RIGOROUS", "keep probing")
and gave the tutor a **zero-argument** `ready_for_next_section` tool. A model
can call a no-argument tool after any turn, and it did — which is the behaviour
in the report that "if I say something, it just says the user has understood".

Two structural changes:

- The plan now carries `key_points` (what the learner must be able to state) and
  `check_questions` (at least one must test *transfer*, not recall). The tutor
  is told it may not advance until it has heard each key point from the learner.
- The tool is now `mark_section_understood(learner_explanation, key_points_covered,
  confidence)`. A model that cannot quote the learner's own explanation cannot
  fill in the argument; the worker additionally rejects calls with fewer than six
  words of explanation or confidence below 60, and tells the tutor to keep probing.
  The arguments are the audit trail for why a section was passed.

## 4. Scoring: four independent defects

### 4a. The grader was never shown the material

`score_understanding(doc_name, transcript)` passed the **document's filename**
and the transcript. Nothing else. The model was asked to judge whether spoken
answers were correct without being told what correct was.

This is the single biggest cause of the score distribution. Of 128 sittings:

```
score  41 → 31 sittings      score 70 → 21 sittings
score  10 → 24 sittings      score  0 → 22 sittings
```

Half the scores are two magic numbers. With no ground truth the model anchored
on the rubric's band edges. The grader now receives the section's title, aim,
key points, and its **source text**, and must return per-key-point evidence
quoted from the learner.

### 4b. "Said nothing" was recorded as 10/100

`score_session` had a filler gate: if the learner's words were all in a
stop-list, it skipped the LLM and wrote `score = 10`. In production **23 rows**
match that fallback exactly (`score=10, duration='1 exchanges', topics='0 topics'`).

Opening a document and closing it wrote a 10. That is not a measurement, and it
counted like one. A sitting with nothing to assess is now `score = NULL`, kept
for audit, and excluded from every average.

### 4c. The admin saw the last sitting, not the learner

```python
user.understanding = score   # every single scoring call
```

`users.understanding` was overwritten on every sitting. Since sittings are
per-section, the number an admin saw was whatever the learner's most recent
section scored — often the noise-gate 10. That is the "latest score is what the
admin sees" complaint, exactly.

There was an EMA layer (`workspace._doc_ema`) added later for the People table,
but it averaged sitting scores **across sections of the same document**, which
answers no useful question: a learner's section-1 and section-4 scores are not
repeated attempts at the same thing.

Understanding is now derived, never stored:

```
sitting   one section, one attempt   → sessions.score (nullable)
section   best score ever achieved   → section_progress.best_score
document  minutes-weighted mean of section bests, over ALL planned sections
learner   mean of document scores, over documents started
cohort    mean of member scores, scoped to the cohort's documents
```

A section keeps its **best** result, so a cut-short sitting can no longer erase
understanding already demonstrated. A document counts its untaught sections as
zero, so the document score answers "how much of this do they know" rather than
"how did the last five minutes go".

### 4d. Sittings were graded and thrown away

Transcripts went to the external memory service and nowhere else. Nothing in the
database recorded *why* a learner received a score, so a score could not be
audited or re-graded. `sessions.transcript` now stores them, and
`GET /api/people/{uid}/sessions/{sid}` returns the transcript with the
per-key-point evidence behind the number.

## 5. What was done to the live data

Applied to `praxos_lms` (migration `e3a91c7b40d5`, full `pg_dump` taken first).

**Historical scores are retired, not re-graded.** Re-grading needs transcripts,
and none of the 128 sittings stored one — the transcript went to the external
memory service and nowhere else. Every historical score was also produced by a
grader shown only a filename, which is why half of them are exactly 41 or 70 and
23 more are the hard-coded 10. Rather than carry numbers nobody can reproduce,
each sitting is kept as a record — with its section attributed from the order of
that learner's sittings, the only surviving per-section signal — and `score`
set to NULL. Nothing historical feeds an average; every learner reads as
"Not started" until they sit a session under the new pipeline.

Path items previously marked `mastered` were re-opened (`up_next`): leaving them
would assert a completion that nothing now evidences.

**Plan coverage was repaired in place** (`scripts/repair_plan_coverage.py`).
Five of the thirteen live documents had chunks in no section at all, including
the 46-chunk one missing 21. The script re-derives section boundaries so the plan
covers the whole document, keeping every section's title and order — no model
required. All 13 documents now report full coverage. The splits it produces are
mechanical (one section absorbed chunks 30–45), so regenerating those plans with
`POST /api/documents/{id}/plan/generate` once an LLM key is configured will give
a better division — and will also populate `key_points`/`check_questions`, which
old plans do not have.

**Not done, and why:** the existing plans have empty `key_points`, so until they
are regenerated the grader falls back to judging against the section's source
text and the tutor cannot enforce per-key-point checks. Regeneration needs a
configured provider (`LLM_API_KEY`), which this environment did not have.

## 6. Where re-grading becomes free

Sittings now store `sessions.transcript` alongside the per-key-point evidence.
The next time the rubric changes, a re-grade is a replay over stored transcripts
rather than a manual exercise — which is the thing that was impossible here.
