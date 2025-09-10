# TUEV Gap Analysis: Our Implementation vs Reference

**Created**: September 10, 2025  
**Purpose**: Identify concrete differences between our implementation and the EEGPT reference  
**Impact**: Our BAC=0.19–0.24 vs Reference BAC≈0.62

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

### 1) Class Balancing ⚠️ MAJOR
| Aspect        | Reference      | Ours                    | Impact                               |
|---------------|----------------|-------------------------|--------------------------------------|
| Sampling      | No balancing   | WeightedRandomSampler   | Model sees artificial distribution   |
| Class weights | None           | 1 / class_count        | Alters gradient magnitudes           |
| Result        | ~62% BAC       | 19–24% BAC              | Likely overcorrected rare classes    |

Hypothesis: Oversampling rare classes harms generalization on the natural eval distribution.

### 2) Data Scale / Normalization ⚠️ MAJOR
| Aspect      | Reference           | Ours                               | Impact                         |
|-------------|---------------------|------------------------------------|--------------------------------|
| Units       | Microvolts (μV)     | Volts → normalized to N(0, 50 μV)  | Different input distributions  |
| Range       | ~[−100, +100] μV    | ~[−2, +2] after z‑scoring          | Feature magnitude mismatch     |
| Normalizing | Not emphasized      | Default mean=0, std=50 μV          | May miscalibrate features      |

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
| DropPath          | 0.2                       | Not implemented             | ⚠️ Add for stability       |

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

### 4) 🟢 Add DropPath (0.2)
Add drop‑path in the encoder blocks (stabilization). Expect regularization more than a direct BAC jump.

### 5) 🟢 Mean Pooling Toggle
Test mean pooling vs flatten(4×512) → head; choose the better head behavior.

## Validation & Monitoring
- Confusion matrix and per‑class classification_report on every eval (watch rare class recall: spsw/gped/pled).
- Print batch label distributions (first few batches) to confirm natural prevalence after removing balancing.
- Track BAC by epoch; acceptance gates: ≥0.25 by epoch 2–3, ≥0.40 by epoch 5, final ≈0.62±0.02 by ~30.

## Expected Outcomes
1) Remove sampler → BAC likely jumps from 0.24 → 0.35–0.45.
2) Align normalization → +0.10–0.15 BAC (choose μV or corpus stats based on eval).
3) Match batch/accum → +0.02–0.05 BAC.
4) DropPath/Pooling → Stabilization and incremental gains.

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
