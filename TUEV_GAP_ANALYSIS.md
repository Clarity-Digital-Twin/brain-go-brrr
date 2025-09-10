# TUEV Gap Analysis: Our Implementation vs Reference

**Created**: September 10, 2025  
**Purpose**: Identify concrete differences between our implementation and the EEGPT reference  
**Impact**: Our BAC=0.19–0.24 vs Reference BAC≈0.62

## Current Run Snapshot (post‑fix)

- Epoch ~20 metrics (eval):
  - Balanced accuracy ≈ 0.242
  - Weighted F1 ≈ 0.545; Kappa ≈ 0.30
  - Pattern: Strong bckg recall (~0.95), partial gped (~0.51), near‑zero for spsw/pled/eyem/artf.
- Interpretation: Model still under‑learning rare classes despite correct splits/sampling/scale.

Hypothesis set (ordered by likelihood):
- H1: Input scale mismatch subtlety — raw μV without z‑score may still be off relative to the checkpoint’s pretraining statistics; early layers may need more adaptation.
- H2: Window alignment difference — our event centering (midpoint) vs. reference’s start/end‑bounded slicing (plus per‑recording offset) might dilute event salience for brief classes (spsw/pled).
- H3: Patch granularity — using patch_size=64 at 200 Hz (≈0.32 s) may differ from the reference’s effective temporal granularity; a 50‑sample patch (≈0.25 s) could better match pretraining semantics.
- H4: Optimization nuance — effective batch not exactly 400 and layer‑wise LR decay might slow adaptation of early layers needed for minority classes.

Immediate validations (low cost):
- V1: Confirm logs contain: natural sampling message; normalization disabled; DropPath enabled message; effective batch print; parity mode on.
- V2: Log input stats right before the mapper on a batch: min/median/max of `x*1e6`; expect ~[−100, +100] μV typical ranges.

Fast follow‑ups (1–2 short runs):
- F1: Exact 400 effective batch (e.g., `--batch_size 40`, accum=10) and reduce LR decay to 0.75→0.65 baseline check. Monitor epochs 1–5.
- F2: Diagnostic scale A/B (debug‑only): keep μV scaling, toggle wrapper normalization on vs. off for 5 epochs; choose setting that lifts minority recalls early. Final setting should still match reference if possible.
- F3: Patch granularity ablation: set `patch_size=50, patch_stride=50` (with `time_steps=1000`) to restore ≈0.25 s receptive field at 200 Hz. Validate checkpoint compatibility (weights load for conv kernel 1×64 vs 1×50 may require re‑init of `patch_embed` which deviates from reference weights; consider as diagnostic, not final).

If no improvement by epoch 10 in any fast follow‑up:
- Revisit event extraction alignment: implement the reference’s start/end‑bounded slicing with enforced 1000 samples and any per‑recording offset logic; compare a 200‑sample visual overlay for a handful of spsw/pled cases.

## Pre‑Flight Critical Issues (Fix FIRST)

### 0. Data Split Mismatch ✅ FIXED
| Aspect        | Reference                      | Ours (BEFORE)                          | Ours (FIXED)                   |
|---------------|--------------------------------|----------------------------------------|--------------------------------|
| Split source  | Uses TUEV's pre-split dirs     | Used wrong cache with seed=42         | Now uses official dirs         |
| Split method  | train/ and eval/ directories   | 80/20 random split (wrong!)           | train/ and eval/ dirs          |
| Train files   | 359 files                      | Unknown (wrong split)                  | 359 files (REBUILDING)         |
| Eval files    | 159 files                      | Unknown (only 4 subjects!)            | 159 files (REBUILDING)         |
| Cache status  | N/A                            | WRONG - had path bug                  | Fixed & rebuilding now         |

**BUGS FIXED**: 
1. Cache had wrong file paths (only basename, not full path)
2. cache_dir undefined in build loop - FIXED
3. Now rebuilding with correct splits & paths

Action:
- Verify our splits are subject‑based and reproducible. No subject must appear in both splits.
- Align fallback split seed to 4523 if using programmatic splits.

Quick check:
```bash
uv run python - << 'PY'
import json
from pathlib import Path

def subjects_from(index_path: Path):
    data = json.loads(Path(index_path).read_text())
    # Our index stores 'subject' explicitly; fallback: derive from file stem
    subs = [s.get('subject') or s['file'].split('_')[0] for s in data['segments']]
    return set(subs)

train_idx = Path('data/datasets/tuev/cache/tuev_event_segments/train/index.json')
eval_idx  = Path('data/datasets/tuev/cache/tuev_event_segments/eval/index.json')
train_subs = subjects_from(train_idx)
eval_subs = subjects_from(eval_idx)
overlap = train_subs & eval_subs
print(f"Train subjects: {len(train_subs)}  Eval subjects: {len(eval_subs)}  Overlap: {len(overlap)}")
if overlap:
    print("OVERLAP SUBJECTS:", sorted(list(overlap))[:10])
PY
```

## Critical Divergences (Likely Driving the Gap)

### 1) Weight Normalization in Classifier Head 🔴 CRITICAL MISSING PIECE
| Aspect        | Reference                         | Ours                    | Impact                               |
|---------------|-----------------------------------|-------------------------|--------------------------------------|
| Head layer    | LinearWithConstraint(30720, 6)   | nn.Linear(30720, 6)     | **TRAINING COLLAPSE**                |
| Weight norm   | max_norm=1 every forward pass     | None                    | Weights explode with 30k features    |
| Implementation| torch.renorm(weights, p=2, dim=0) | Standard Linear         | Gradients unstable, minority classes die |
| Result        | Stable training, all classes learn| Only 2/6 classes work   | **THIS IS THE SMOKING GUN**          |

**Why this is critical**: With 30,720 input features and Dropout(0.8), weight magnitudes can explode without constraints. The renormalization keeps each output neuron's incoming weights bounded, preventing the collapse to majority classes we're seeing.

### 2) Class Balancing ✅ FIXED
| Aspect        | Reference      | Ours (FIXED)            | Impact                               |
|---------------|----------------|-------------------------|--------------------------------------|
| Sampling      | No balancing   | No balancing            | Natural distribution                 |
| Class weights | None           | None                    | Matches reference                    |
| Result        | ~62% BAC       | 24% BAC (other issues)  | Not the cause anymore                |

### 3) Data Scale / Normalization ✅ FIXED
| Aspect      | Reference           | Ours (FIXED)               | Impact                         |
|-------------|---------------------|----------------------------|--------------------------------|
| Units       | Microvolts (μV)     | Microvolts (μV)            | Matches reference              |
| Range       | ~[−100, +100] μV    | ~[−100, +100] μV           | Same scale                     |
| Normalizing | Raw μV values       | Normalization disabled     | Matches reference              |

### 3) Batch Size & Accumulation
| Aspect        | Reference        | Ours                    | Impact                                 |
|---------------|------------------|-------------------------|----------------------------------------|
| Total batch   | 400 (DDP, 2 GPUs)| 32×12 steps ≈ 384       | Slightly different gradient statistics |
| Update cadence| Every step (DDP) | After N micro‑batches   | Different update cadence (not stale)   |

## Moderate Divergences (5–10% each)

### 4) Mean Pooling Strategy
| Aspect            | Reference                 | Ours                        | Impact                     |
|-------------------|---------------------------|-----------------------------|----------------------------|
| Feature reduction | `use_mean_pooling` option (often enabled) | Flatten 4 tokens (→ 2048)   | Different head behavior    |
| Classifier input  | 512 (if pooled)           | 2048                        | Head capacity difference   |

### 5) Training Infrastructure
| Aspect | Reference                | Ours                        | Impact                         |
|--------|--------------------------|-----------------------------|--------------------------------|
| GPUs   | 2 GPUs (DDP/Deepspeed)   | 1 GPU (accumulation)        | Different gradient statistics  |

## Minor Divergences

| Aspect            | Reference                 | Ours                        | Status/Impact              |
|-------------------|---------------------------|-----------------------------|----------------------------|
| Window extraction | Fixed 1000 samples        | Fixed 1000 samples          | ✅ Same                    |
| Window alignment  | Slice via start/end ±2 s (plus per‑recording offset); enforces 1000 samples | Centered window (tmin=−2, tmax=+3), enforces 1000 samples | ⚠️ Minor alignment difference |
| Channel mapping   | 23→20 Conv2d              | 23→20 mapper (equivalent)   | ✅ Equivalent              |
| Label mapping     | 1–6 → 0–5 (−1)            | Explicit dict {spsw:0…}     | ✅ Same semantics          |
| Label smoothing   | 0.1                       | 0.1                         | ✅ Same                    |
| Learning rate     | 5e‑4                      | 5e‑4                        | ✅ Same                    |
| Layer decay       | 0.65                      | 0.65                        | ✅ Same                    |
| Warmup            | 5 epochs                  | 5 epochs                    | ✅ Same                    |
| DropPath          | 0.2                       | Implemented (0.2)           | ✅ Same                    |

## Revised Action Plan (in order)

### 0) 🔴 Pre‑Flight: Verify Splits (no leakage)
- Run the overlap check above. If any overlap, rebuild cache with subject‑level 80/20 and seed=4523.

### 1) 🔴 Remove Class Balancing
```python
# REMOVE
# train_sampler = WeightedRandomSampler(...)
# train_loader = DataLoader(dataset, sampler=train_sampler, ...)

# REPLACE
train_loader = DataLoader(dataset, shuffle=True, ...)
```
Rationale: Reference reaches ≈62% without balancing. Expect earlier and higher eval BAC progression.

### 2) 🟡 Align Normalization
Run A/B:
- A: μV inputs, no normalization (set wrapper.normalize=False after creation, or use create_normalized_eegpt(normalize=False)).
- B: Corpus stats (compute mean/std on train split; apply uniformly).
Monitor per‑class recall; choose better.

### 3) 🟡 Match Effective Batch ≈ 400
```text
batch_size × accumulation_steps ≈ 400
examples: 32×13=416  |  34×12=408
```

### 4) 🟢 DropPath (0.2) — Implemented
No action needed; wired end-to-end. Trainer passes `drop_path_rate=0.2`; model prints enablement on init.
Add drop‑path in the encoder blocks (stabilization). Expect regularization more than a direct BAC jump.

### 5) 🟢 Temporal Tokens — Implemented
Use ALL temporal summary tokens (N_temporal×4×512 → 30,720) with Dropout(0.8) → Linear(6), matching the authors’ classifier.

## Validation & Monitoring
- Confusion matrix and per‑class classification_report on every eval (watch rare class recall: spsw/gped/pled).
- Print batch label distributions (first few batches) to confirm natural prevalence after removing balancing.
- Track BAC by epoch; acceptance gates: ≥0.25 by epoch 2–3, ≥0.40 by epoch 5, final ≈0.62±0.02 by ~30.

## Expected Outcomes (UPDATED WITH CRITICAL FINDINGS)
1) **LinearWithConstraint in head** → MASSIVE IMPACT, expect BAC jump 0.24 → 0.45-0.55
2) **Conv2dWithConstraint + full mapper** → Additional +0.05-0.10 BAC from stable channel mapping
3) **Both fixes combined** → Should reach 0.55-0.62 BAC (paper target)

Previous fixes already applied:
- Natural sampling ✅
- μV scale ✅ 
- Temporal tokens ✅
- Boundary handling ✅
- Exact batch 400 ✅

## Smoking Guns 🔫
1) Split mismatch: If subjects overlap or splits differ from reference, comparisons are invalid.
2) Class balancing paradox: Reference tolerates severe imbalance; forced balancing may harm generalization.
3) Normalization mismatch: Reference μV vs our z‑scaling can shift feature magnitudes.

## Next Steps (strict order)
1) Verify subject splits (no overlap; seed=4523 if programmatic).  
2) Remove sampler; re‑run parity mode with workers=0, no pin_memory on WSL.  
3) Test normalization A/B (μV vs corpus stats).  
4) Match batch/accum; consider drop‑path and pooling toggle.  
5) Reassess BAC trajectory and per‑class recall; iterate only on the smallest change needed.

## Non‑Blocking Tech Debt: MNE pick_channels() Warnings

- Symptom: During cache rebuild, messages like `NOTE: pick_channels() is a legacy function. New code should use inst.pick(...)`.
- Cause: Some code paths still call `raw.pick_channels(...)` or `mne.pick_channels(...)`.
- Impact: Cosmetic only; no change to extracted data or results.
- Locations (key):
  - `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py`
  - `src/brain_go_brrr/infra/preprocessing/flexible_preprocessor.py`
  - `src/brain_go_brrr/domain/preprocessing/eegpt_preprocessing.py`
  - `src/brain_go_brrr/infra/preprocessing/eeg_preprocessor.py`
  - `src/brain_go_brrr/infra/preprocessing/snippets/maker.py`
- Migration plan (post‑parity):
  - Route selections through `mne_compat.pick_channels(raw, picks)` or call `raw.pick(picks=names_in_target_order)` directly.
  - Preserve channel order by passing `picks` in final desired order (no `ordered=True` needed).
  - Log missing channels; do not raise.
