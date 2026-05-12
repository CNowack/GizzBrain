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

## 2026-05-12 — Pre-computed spectrograms + parallel data loading

**Files:** `gizzbrain/encoder.py`, `gizzbrain/model.py`, `gizzbrain/cli.py`

**Problem:** GPU utilization was stuck around 10% during training. The GPU was starving — it can process a batch in milliseconds, but was sitting idle waiting for the CPU to decode audio and compute Mel Spectrograms from scratch before every single batch, on every epoch.

**Changes:**

- Added `precompute_chunks()` to `encoder.py`. Iterates over the full chunk index once, calls `audio_to_tensor()` on each chunk, and saves the resulting tensor as a numbered `.pt` file (e.g. `spectrograms/0000001.pt`). Returns the chunk DataFrame with a `tensor_path` column pointing to each saved file.

- Modified `GizzDataset.__getitem__()` in `encoder.py`. If the chunk DataFrame has a `tensor_path` column, it loads the pre-saved tensor from disk with `torch.load()`. Otherwise, it falls back to the original on-the-fly computation. No changes required at the call site — the dataset detects which mode to use automatically.

- Added `num_workers` parameter to `train_model()` and `run_inference()` in `model.py`, wired through to both `DataLoader` calls. `persistent_workers=True` is set automatically when `num_workers > 0` to avoid respawning worker processes between epochs.

- Added `gizzbrain preprocess` command to `cli.py`. Filters to the vanguard top-10 set, builds the chunk index, runs `precompute_chunks()`, and saves the result as `chunks.parquet`. Prints the next command to run when finished.

- Added `--chunks` flag to `gizzbrain train` and `gizzbrain evaluate`. When provided, skips on-the-fly chunking and loads pre-computed tensors from the chunk index instead.

- Added `--workers` flag to `gizzbrain train` and `gizzbrain evaluate`. Defaults to 0. Set to 4 for parallel loading when using pre-computed chunks.

**New workflow:**
```
gizzbrain preprocess --data library.parquet          # run once
gizzbrain train --data library.parquet --chunks chunks.parquet --workers 4
gizzbrain evaluate --data library.parquet --chunks chunks.parquet --workers 4
```

---

## 2026-05-12 — Applause trimming in chunk indexer

**File:** `gizzbrain/encoder.py`

**Change:** Added `trim_seconds=30.0` parameter to `create_chunk_index()`. Chunks whose start offset falls within the first 30 seconds, or whose end offset falls within the last 30 seconds of a file, are now skipped.

**Why:** Live recordings open and close with crowd applause. Those chunks are acoustically identical across all songs — pure noise that actively degrades model accuracy by teaching the CNN that broadband crowd roar belongs to a specific title. Trimming them tightens the training signal. The same trim applies at inference time (via the same function) so training and prediction see consistent input.

**Trade-off:** Songs with distinctive intros or outros lose that material. 30 seconds is conservative enough that most songs retain plenty of body content. The value is a named parameter so it can be tuned without a code change.
