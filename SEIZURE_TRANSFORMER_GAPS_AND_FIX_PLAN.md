# SeizureTransformer – Gaps and Fix Plan

Scope: SeizureTransformer on TUSZ (train/dev/eval). This is the single place to track gaps vs spec and the concrete steps to close them.

Canonical docs
- Spec: `IDEAL_REFERENCE_SEIZURE_TRANSFORMER_DATAFLOW.md`
- Current: `SEIZURE_TRANSFORMER_CURRENT_STATUS.md`
- This plan: `SEIZURE_TRANSFORMER_GAPS_AND_FIX_PLAN.md`

Current status (snapshot)
- Discrimination: window‑level AUROC computed (see README usage). Last observed ≈ 0.84 on eval.
- Clinical metrics: NEDC wrapper available, not yet integrated into the standard eval script; FA/24h not reported yet.
- Implementation shape: Architecture vendored; SSOT preprocessing present; canonical channels enforced; evaluation path stable.

Gaps vs spec (what to verify/tighten)
- Dataset splits: Ensure exact TUSZ v2.0.3 split parity (train/dev/eval file lists).
- SSOT preprocessing parity: z‑score → resample 256 Hz → causal 0.5–120 Hz band‑pass → notches at 1 Hz and 60 Hz (Q=30). Confirm identical ordering/coefficients in both train and eval.
- Montage/channels: enforce unipolar montage and canonical TUAB‑19 ordering everywhere (train/dev/eval).
- Labels: training uses per‑timestep masks; validation/eval use scalar window labels; event scoring uses references from TUSZ annotations only.
- Evaluation windows: 60 s windows, stride 60 s for eval AUROC; no post‑processing applied to AUROC.
- Safety/CI: safe `torch.load` usage (weights_only or explicit justification), no Lightning, no sys.path hacks.

Plan to reduce FA/24h (dev tuning only)
1) Expose operating point params (if not already):
   - Threshold: `0.3–0.95`
   - Morph kernel (samples at 256 Hz): `5–31`
   - Min event duration: `2–10 s`
   - Merge gap: `5–30 s`
   - Optional probability smoothing: MA window `0.5–2 s`
2) Grid/line search on TUSZ dev:
   - For each setting: convert predictions → run NEDC TAES → record Sensitivity and FA/24h.
   - Select the global operating point meeting target FA/24h (e.g., 5–10/day) with maximal sensitivity.
3) Freeze the chosen params and evaluate once on TUSZ eval; report TAES sensitivity vs FA/24h and the operating point used.
4) Keep AUROC pipeline unchanged (no post‑processing) and report alongside TAES.

Concrete tasks (code locations)
- Post‑processing utils: `src/brain_go_brrr/infra/eval/post_processing.py`
- Clinical scorer: `src/brain_go_brrr/infra/eval/nedc_wrapper.py` (class `NEDCClinicalEvaluator`)
- Eval runner: `scripts/evaluate_seizure_transformer.py`
- Add a dev sweep script: `scripts/tusz_sweep_dev.py` (new)
  - Inputs: paths to dev EDF root, predictions or model, and sweep ranges.
  - Outputs: CSV of (threshold, kernel, min_dur, merge_gap, smooth_win, sensitivity, FA/24h).
  - Plot: optional Sensitivity vs FA/24h curve to select operating point.

Example integration (clinical metrics in eval):
```python
from brain_go_brrr.infra.eval.nedc_wrapper import NEDCClinicalEvaluator

evaluator = NEDCClinicalEvaluator()
fa_per_24h, sensitivity = evaluator.evaluate_predictions(pred_events, ref_events)
```

Acceptance criteria
- Parity: SSOT preprocessing and channel policy identical across train/dev/eval.
- AUROC: computed window‑level without post‑processing; within expected range.
- TAES: a clearly documented operating point tuned on dev and frozen on eval.
- Repro: single command per stage (dev sweep, freeze, eval) with seeds set and paths documented.

Next actions
- Implement `scripts/tusz_sweep_dev.py` (dev sweep) and extend `scripts/evaluate_seizure_transformer.py` to emit/refine event lists for NEDC.
- Run the sweep on dev, select operating point, and re‑run eval once.
- Update `SEIZURE_TRANSFORMER_CURRENT_STATUS.md` with the finalized operating point and results.
