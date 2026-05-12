# GizzBrain Devlog

---

## 2026-05-12 — Applause trimming in chunk indexer

**File:** `gizzbrain/encoder.py`

**Change:** Added `trim_seconds=30.0` parameter to `create_chunk_index()`. Chunks whose start offset falls within the first 30 seconds, or whose end offset falls within the last 30 seconds of a file, are now skipped.

**Why:** Live recordings open and close with crowd applause. Those chunks are acoustically identical across all songs — pure noise that actively degrades model accuracy by teaching the CNN that broadband crowd roar belongs to a specific title. Trimming them tightens the training signal. The same trim applies at inference time (via the same function) so training and prediction see consistent input.

**Trade-off:** Songs with distinctive intros or outros lose that material. 30 seconds is conservative enough that most songs retain plenty of body content. The value is a named parameter so it can be tuned without a code change.
