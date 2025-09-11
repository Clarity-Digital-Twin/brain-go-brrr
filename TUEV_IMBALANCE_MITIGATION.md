# TUEV Imbalance Mitigation Plan (Experimental Deviation)

Purpose: Define a single, senior-auditable deviation from the reference TUEV pipeline to address severe class imbalance (e.g., SPSW ~0.5%). This is exploratory and non-clinical. Parity docs remain the SSOT for the reference implementation.

Scope:
- Keep datasets emitting Volts (SI units) and μV/100 scaling before the mapper.
- Changes live behind flags in `experiments/eegpt_linear_probe/train_tuev_events.py` and helper utils in `src/` (no reimplementation in experiments/).
- Goal: improve rare-class recall and lift eval BAC toward ~0.45–0.58.

Why: Natural sampling with extreme class skew learns background/GPED only. A principled, minimal set of imbalance mitigations can provide a better stress test of the architecture and help the community benchmark realistically.

Non-Goals:
- Clinical deployment (TUEV event detection remains clinically non-viable at 58–62% BAC).
- Large architectural changes or training from scratch.

Proposed Mitigations (Toggleable)

1) Class-weighted loss (low risk)
- Option: `--class_weights counts|cb:0.9999|none`.
- counts: `w_c = 1 / n_c` normalized.
- cb(β): Class-Balanced Loss (effective number of samples), `w_c = (1-β)/(1-β^{n_c})`, β≈0.99–0.9999.
- Implementation: compute from train cache index; pass to `nn.CrossEntropyLoss(weight=...)`.

2) Weighted sampling (moderate risk)
- Option: `--sampler weighted|none`.
- Use `WeightedRandomSampler` with per-sample weight `w[y]` from above.
- Guard: per-epoch unique sampling with length = dataset size; keep shuffle semantics.
- Note: Subject-awareness is desirable but can be deferred; keep it simple initially.

3) Rare-class augmentation (moderate risk)
- Option: `--augment minority_shift_ms=200 jitter_uV=5 noise_uV=5` (applied only to minority classes).
- Time-shift window by ±minority_shift_ms within valid bounds (preserves label).
- Amplitude jitter ±jitter_uV; add Gaussian noise σ=noise_uV (in μV units, i.e., after μV/100 scaling or applied in Volts before scaling with appropriate conversion).
- Keep augment prob small (e.g., 0.3) to avoid distribution drift.

4) Freeze/unfreeze schedule (low risk)
- Option: `--freeze_eegpt_epochs N` (default 5).
- Train mapper+head first; then unfreeze EEGPT with lower lr via existing layer decay (0.65).
- Rationale: stabilize classifier on scarce signals before adapting deep features.

5) Normalization ablation (diagnostic)
- Option: `--normalize_eegpt` (default False to match reference).
- Try wrapper normalization (z-score) in a short ablation to check impact on minority recall. Keep off for main runs unless clearly beneficial.

6) Focal loss (optional)
- Option: `--focal_loss alpha=0.25 gamma=2.0`.
- Down-weights easy (background) examples; focuses gradient on hard/rare classes.
- Implementation: add a small focal loss in `src/brain_go_brrr/infra/training/losses.py` and gate via flag.
- Note: Don’t combine with heavy class weights initially; test independently to avoid overcorrection.

Training/Eval Protocol
- Keep all other settings as in parity: μV/100 scaling, 200 Hz, 5 s windows, mapper 23→20, EEGPT (20 ch, time_steps=1000, stride=64), features=30720.
- Monitor: per-class recall, macro recall, BAC, macro F1, Cohen’s κ. Save best by macro recall or BAC.
- Seeds: run 3 seeds (e.g., 0/1/2), report mean±std.

Success Criteria (for this experiment)
- Early epochs (≤10): minority recall begins >0.1 for SPSW/PLED/EYEM/ARTF.
- Final (≤30 epochs): BAC ≥ 0.45 AND minority recall ≥ 0.2 on at least two rare classes.
- If not met, document results as negative and stop investing further.

Minimal Code Changes (where)
- `src/brain_go_brrr/utils/sampling.py`: helpers to compute class weights and build `WeightedRandomSampler`.
- `experiments/eegpt_linear_probe/train_tuev_events.py`: add flags and wire:
  - `--class_weights`, `--sampler`, `--augment ...`, `--freeze_eegpt_epochs`, `--normalize_eegpt`.
  - Build loss with weights; optionally sampler; simple minority-only augmentation in collate or dataset wrapper.
- Keep experiments thin; reuse src utilities only.

Risks & Notes
- Oversampling can overfit rare classes; anneal strength over epochs if needed.
- Time-shifts must stay within annotated event context (the triple buffer provides slack, but keep shifts small).
- Report both natural and balanced metrics; avoid overstating improvements.
- Label smoothing + class weights: timm’s LabelSmoothingCE may not accept `weight`. If needed, switch to `nn.CrossEntropyLoss(weight=...)` or implement a weighted label-smoothing loss in `src`.

Run Template (illustrative)
```bash
uv run python experiments/eegpt_linear_probe/train_tuev_events.py \
  --data_dir data/datasets/tuev \
  --cache_dir data/datasets/tuev/cache \
  --eegpt_checkpoint data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt \
  --epochs 30 --batch_size 40 --num_workers 4 \
  --class_weights cb:0.9999 --sampler weighted \
  --augment minority_shift_ms=200 jitter_uV=5 noise_uV=5 \
  --freeze_eegpt_epochs 5
```

Positioning
- This plan is intentionally non-parity. It exists alongside the parity docs and is isolated via flags.
- Outcome (positive or negative) should be documented and not conflated with the reference reproduction.
