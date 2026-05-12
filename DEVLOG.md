# GizzBrain Devlog

---

## 2026-05-12 — `gizzbrain evaluate` command

**Files:** `gizzbrain/cli.py`, `gizzbrain/model.py`

**Changes:**
- Added `run_inference()` to `model.py`. Runs a loaded model over a `GizzDataset` in eval mode (same per-batch normalization as training) and returns the chunk DataFrame with `predicted_label` and `confidence` columns appended.
- Extracted the vanguard top-10 + file-level 80/20 split logic from the `train` command into a `_build_splits()` helper in `cli.py`. Both `train` and `evaluate` now call the same function, guaranteeing identical splits.
- `train` command now saves `label_map.json` alongside `gizzbrain_weights.pt`. This records which class index maps to which song title so evaluate doesn't have to reconstruct it.
- Added `gizzbrain evaluate` subcommand with `--data`, `--weights`, `--labels`, `--split` (train/val/all), and `--batch-size` flags.
- Added `_print_eval_report()` to `cli.py` which prints: chunk-level accuracy, file-level accuracy (majority vote across chunks), a per-song breakdown table (files, correct, accuracy, avg confidence), and a top-confusions list for any mispredicted files.

**Why:** No way to measure model quality after training — epoch loss is a training signal, not a clean accuracy number. The evaluate command gives a reproducible snapshot per trained model, making it possible to compare the impact of architecture or data changes.

---

## 2026-05-12 — Applause trimming in chunk indexer

**File:** `gizzbrain/encoder.py`

**Change:** Added `trim_seconds=30.0` parameter to `create_chunk_index()`. Chunks whose start offset falls within the first 30 seconds, or whose end offset falls within the last 30 seconds of a file, are now skipped.

**Why:** Live recordings open and close with crowd applause. Those chunks are acoustically identical across all songs — pure noise that actively degrades model accuracy by teaching the CNN that broadband crowd roar belongs to a specific title. Trimming them tightens the training signal. The same trim applies at inference time (via the same function) so training and prediction see consistent input.

**Trade-off:** Songs with distinctive intros or outros lose that material. 30 seconds is conservative enough that most songs retain plenty of body content. The value is a named parameter so it can be tuned without a code change.
