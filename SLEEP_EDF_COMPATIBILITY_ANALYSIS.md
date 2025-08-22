# Sleep-EDF, YASA, and EEGPT Compatibility Analysis (Repo-Audited)

## Executive Summary

**YES, Sleep-EDF data CAN be processed through EEGPT (with caveats)** — sampling rate differences (100 Hz vs 256 Hz) are routine and handled via resampling. The main real constraint is channel availability: Sleep-EDF typically provides only two EEG channels (Fpz-Cz, Pz-Oz) plus EOG/EMG, while EEGPT was designed for 19–58 channels. This requires an explicit policy (gate, pad, or spatially interpolate) when extracting EEGPT features from Sleep-EDF.

## Key Findings

### 1. Sleep-EDF Dataset
- **Yes, this is the public dataset we have** in `/data/datasets/external/sleep-edf/`
- Contains Sleep Cassette and Sleep Telemetry studies
- **Native sampling rate: 100 Hz** (confirmed by checking actual EDF files)
- Gold standard public dataset for sleep staging research
- Used by YASA and many other sleep staging papers
- Typical channels: `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`, `EMG submental` (2 EEG, +EOG/EMG)

### 2. YASA Requirements
- **Designed for 100 Hz data** (see yasa.md line 314)
- Actually **downsamples higher frequency data TO 100 Hz** for efficiency
- Processes Sleep-EDF natively without any issues
- Achieves 87% accuracy on Sleep-EDF

### 3. EEGPT Requirements  
- Target sampling: **256 Hz** (patch size 64 → 250 ms windows)
- Paper resamples all datasets to 256 Hz (EEGPT literature notes)
- Channel expectation: typically 19+ standard 10–20 channels (up to 58). Sleep-EDF has far fewer; extraction must handle this explicitly.

## The Solution: Resampling Is Standard Practice (But Mind Channels)

### Evidence from Literature

1. **EEGPT Paper (EEGPT.md)**:
   - Line 189: "preprocessing steps, including... resampling (256Hz)"
   - Line 477: "first downsampled to 256 Hz"
   - Line 493: "first downsampled to 256 Hz"  
   - Line 513: "first upsampled to 256 Hz" (for data below 256 Hz)

2. **YASA Paper (yasa.md)**:
   - Line 272: MESA dataset "sampled at 256 Hz" but with "100 Hz cutoff filter"
   - Line 314: "downsampled to 100 Hz to speed up computation"

### Why Resampling Works

1. **Nyquist Theorem**: 100 Hz sampling captures frequencies up to 50 Hz
2. **Sleep EEG Range**: Most relevant sleep features are 0.5-30 Hz
3. **No information loss by upsampling**: 100→256 Hz preserves original <50 Hz content
4. **Industry standard**: Both papers show resampling is routine
5. **Channel caveat**: Sleep-EDF’s 2 EEG channels limit EEGPT’s spatial modeling; decide on strategy (see below).

## Implementation Requirements

### Current Reality in Our Repo (audited)
- Resampling utilities already exist:
  - `src/brain_go_brrr/domain/preprocessing/eegpt_preprocessing.py::preprocess_for_eegpt` resamples to 256 Hz when needed.
  - `src/brain_go_brrr/infra/ml_models/eegpt_compat.py::preprocess_for_eegpt` also resamples to 256 Hz.
  - `src/brain_go_brrr/infra/preprocessing/flexible_preprocessor.py` supports mode-specific `target_sfreq` (sleep defaults to 100 Hz; abnormality/event to 256 Hz).
- API gap: `src/brain_go_brrr/api/routers/sleep.py::analyze_sleep_stages_eegpt` currently uses the input sampling rate for 4 s windows without resampling to 256 Hz — we should resample here.
- Streaming gap: `src/brain_go_brrr/infra/data/edf_streaming.py` does not resample on load; add optional resample-on-read when targeting EEGPT.
- Tests do NOT skip "Sleep-EDF incompatible with EEGPT" for incompatibility reasons — they skip only when dataset files are missing. We should align our plan with actual tests.

### Correct Implementation (minimal)
```python
def prepare_for_eegpt(raw_eeg, target_rate=256):
    """Resample EEG data for EEGPT processing."""
    current_rate = raw_eeg.info['sfreq']
    
    if current_rate != target_rate:
        # MNE's resample preserves signal integrity
        raw_eeg.resample(target_rate)
    
    return raw_eeg
```

### Channel Policy for Sleep-EDF (critical)
- Option A (default recommended):
  - For Sleep-EDF, run YASA-only path (100 Hz). For EEGPT-derived features, either:
    - Skip by default when available EEG channels < 10–19, and return a clear message/flag; or
    - Allow opt-in with channel padding, clearly labeling reduced fidelity.
- Option B (advanced):
  - Spatially interpolate Sleep-EDF channels to a 10–20 grid and then extract EEGPT features. This requires montage positions and careful validation; not implemented yet.
- Current code state:
  - `domain/preprocessing/eegpt_preprocessing.py` can pad to `n_channels`, but there’s no robust interpolation or strict gating. We should add explicit gating and user-facing messaging.

## Parallel Processing Architecture

The three systems can work together:

```
Sleep-EDF (100 Hz) → Resample → EEGPT (256 Hz) → Features
                  ↓
                YASA (100 Hz) → Sleep Stages
                  ↓
            Combined Analysis
```

### Pipeline Design
1. **Load Sleep-EDF** at native 100 Hz
2. **Fork processing**:
   - Path A: Keep at 100 Hz → YASA sleep staging
   - Path B: Resample to 256 Hz → EEGPT feature extraction (only if channel policy allows)
3. **Merge results**: Combine YASA stages with EEGPT features

## Performance Implications

### Computational Cost
- Resampling: ~0.1-0.5 seconds per 8-hour recording
- YASA at 100 Hz: Faster processing (5 seconds per night)
- EEGPT at 256 Hz: More data points but designed for this rate

### Accuracy Impact
- **No degradation**: Both models trained with resampled data
- YASA: 87% accuracy (validated on Sleep-EDF)
- EEGPT: Designed to handle resampled inputs; Sleep-EDF’s limited channels reduce fidelity; document limitations when enabled.

## Recommendations

### Immediate Actions (repo-driven)
1. API: Resample to 256 Hz in `api/routers/sleep.py::analyze_sleep_stages_eegpt` before windowing.
2. Feature extractor: Ensure preprocessor for EEGPT path enforces `target_sfreq=256` (inject `FlexiblePreprocessorAdapter` with `target_sfreq=256`).
3. Streaming: Add optional `target_rate` to `infra/data/edf_streaming.py::load_header` and/or expose resampling utility on first access.
4. Channel gating: Implement a clear policy for Sleep-EDF when EEG channels < threshold (e.g., < 10 or < 19). Default to disabled with informative message; allow opt-in via config.
5. Documentation: Update README and API docs to reflect resampling and channel policy.

### Code Touch Points (correct paths)
- `src/brain_go_brrr/domain/preprocessing/eegpt_preprocessing.py`
  - Already resamples to 256 Hz; keep as canonical utility for model-facing preprocessing.
- `src/brain_go_brrr/infra/ml_models/eegpt_compat.py`
  - `preprocess_for_eegpt` resamples; keep consistent with domain utility.
- `src/brain_go_brrr/infra/preprocessing/flexible_preprocessor.py`
  - Sleep mode defaults to 100 Hz; abnormality/event modes default to 256 Hz. Make the EEGPT path use `target_sfreq=256` explicitly.
- `src/brain_go_brrr/api/routers/sleep.py`
  - In `/eeg/sleep/stages` (EEGPT linear probe), resample to 256 Hz and enforce channel policy.
- `src/brain_go_brrr/infra/data/edf_streaming.py`
  - Optionally add `target_rate` param to resample on `load_header()` when needed (EEGPT-only).

### Testing Strategy (aligned to current tests)
- Resampling quality:
  - Add `tests/unit/test_resampling_quality.py` with synthetic stage-transition preservation (100→256 Hz) and band-limit checks.
- API EEGPT staging route:
  - Update/add tests in `tests/api/test_sleep_analysis_api.py` or a new `tests/api/test_sleep_stages_eegpt.py` to assert resampling to 256 Hz and enforce channel gating.
- YASA compliance:
  - `tests/unit/test_yasa_compliance.py` already asserts sleep-mode preprocessing resamples to 100 Hz and avoids filtering; keep stable.
- Integration pipeline:
  - `tests/integration/test_parallel_pipeline.py` and `tests/integration/test_yasa_integration.py` cover baseline flows; extend to assert EEGPT path resamples and logs warning when channels are insufficient.

## External Validation

### Independent Sources Confirm Our Analysis
1. **PhysioNet Sleep-EDF**: "The EOG and EEG signals were each sampled at 100 Hz" - official dataset documentation
2. **YASA eLife Paper (2021)**: "signals were then downsampled to 100 Hz" - explicit design choice
3. **MNE Documentation**: `raw.resample()` with proper anti-aliasing is standard practice
4. **SciPy**: `resample_poly(up=64, down=25)` for 100→256 Hz preserves signal integrity

### Physics Confirmation
- **Upsampling 100→256 Hz does NOT create new information** - it interpolates existing <50 Hz content
- **Sleep EEG lives in 0.5-30 Hz range** - well below Nyquist for both rates
- **Zero-phase FIR filtering** prevents phase distortion during resampling

## Current Implementation Gaps (audited)

- API EEGPT staging route (`api/routers/sleep.py::analyze_sleep_stages_eegpt`) does not resample to 256 Hz — fix required.
- No explicit channel gating for EEGPT on Sleep-EDF with < 10–19 EEG channels — define and implement policy.
- Streaming utilities (`infra/data/edf_streaming.py`) lack optional resample path for EEGPT — add param.
- Documentation does not state channel limitations for EEGPT on Sleep-EDF — add notes.

## Implementation Plan

### Phase 1: Core Resampling Infrastructure

#### 1.1 Update EDF Loader (Priority: HIGH)
Implementation details:
- Prefer resampling at preprocessing/model boundaries (domain/infra), not in generic loaders, to avoid surprising behavior.
- If needed, expose `target_rate` in `infra/data/edf_streaming.py::load_header` for EEGPT streaming workflows.

#### 1.2 Enforce EEGPT Resampling at Boundaries (Priority: HIGH)
- Inject `FlexiblePreprocessorAdapter(target_sfreq=256)` into EEGPT feature extraction paths (e.g., `ParallelEEGPipeline`).
- Update `api/routers/sleep.py::analyze_sleep_stages_eegpt` to resample loaded `raw` to 256 Hz before windowing.

### Phase 2: Service Layer Updates

#### 2.1 Dual-Path EDF Streamer (Priority: MEDIUM)
Add an optional resampling hook to `infra/data/edf_streaming.py` to produce 256 Hz windows for EEGPT without loading the full file. Keep YASA path at 100 Hz.

#### 2.2 Update Pipeline Orchestrator (Priority: MEDIUM)
Our repo already has `application/pipeline/parallel.py` running EEGPT and YASA independently. Update EEGPT extractor to use a preprocessor with `target_sfreq=256` and apply channel gating.

### Phase 3: Test Updates

#### 3.1 Fix Integration Tests (Priority: HIGH)
Add targeted tests instead of removing non-existent skips:
- New: `tests/unit/test_resampling_quality.py` (sleep stage boundary preservation; frequency content)
- Update: API tests to assert EEGPT staging route resamples to 256 Hz and applies channel gating.

#### 3.2 Add Resampling Quality Tests (Priority: MEDIUM)
```python
# tests/unit/test_resampling_quality.py

def test_resampling_preserves_sleep_stages():
    """Verify resampling doesn't corrupt sleep stage boundaries."""
    # Create synthetic sleep EEG at 100 Hz
    raw_100 = create_sleep_eeg(sfreq=100)
    
    # Mark stage transitions
    transitions_100 = find_stage_transitions(raw_100)
    
    # Resample to 256 Hz
    raw_256 = raw_100.copy().resample(256)
    
    # Verify transitions preserved (within 1 sample tolerance)
    transitions_256 = find_stage_transitions(raw_256)
    for t100, t256 in zip(transitions_100, transitions_256):
        assert abs(t100 - t256) < (1.0 / 100)  # Within 10ms
```

### Phase 4: Configuration Updates

#### 4.1 Add Resampling Config (Priority: LOW)
Support via `pyproject.toml`/settings or a new `configs/processing.yaml` (optional):
- `eegpt_target_rate: 256`
- `yasa_target_rate: 100`
- `auto_resample: true`
- `channel_gating_min_eeg: 10` (or `19`)
- `allow_channel_padding: false`

#### 4.2 Update API Settings (Priority: LOW)
Expose API settings (if/where applicable) for resampling and channel policy.

## Validation Checklist

- [ ] All Sleep-EDF files load successfully
- [ ] YASA maintains 87% accuracy at 100 Hz
- [ ] API EEGPT staging route resamples to 256 Hz
- [ ] Channel gating policy implemented and documented (default: off for Sleep-EDF)
- [ ] Resampling adds <0.5s overhead per 8 h recording
- [ ] Sleep stage boundaries preserved within 10 ms after 100→256 Hz resampling
- [ ] Parallel pipeline uses 100 Hz (YASA) and 256 Hz (EEGPT) appropriately
- [ ] Documentation updated with resampling and channel notes

## Conclusion

Sampling-rate differences (100 vs 256 Hz) are routine and solved via resampling. The key constraints are:
1. Keep YASA processing at 100 Hz (its trained rate).
2. Resample to 256 Hz only for EEGPT (its required rate) and enforce this at API/model boundaries.
3. Decide and implement a clear channel policy for Sleep-EDF (default gate EEGPT features or allow opt-in with reduced fidelity).
4. Run both paths in parallel and merge results where meaningful.

With these repo-aligned fixes (API resampling; channel gating; optional streaming resample), we maintain fidelity to YASA and EEGPT practices while making Sleep-EDF a first-class, well-documented use case.
