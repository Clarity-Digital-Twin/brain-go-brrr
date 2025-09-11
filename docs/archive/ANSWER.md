# TUEV Reference Q&A (EEGPT Reference Repo)

- Scope: Answers to the compiled questions about the TUEV implementation in this repo, based on code audit and the paper text under `literature/EEGPT.md`.
- Status: Verified against current repo; where the repo cannot confirm, we mark as Unknown and suggest an action.

## Questions & Answers

1) Class balancing / oversampling
- Answer: None found. Training uses the natural class distribution. No `WeightedRandomSampler`, no class weights, and no oversampling paths in `downstream_tueg/utils.py`, `engine_for_finetuning_EEGPT.py`, or the training script.

2) Version path mismatch (v2.0.0 vs v2.0.1)
- Answer: Repo shows mismatch, but treat as non-blocking. Use whatever processed directory exists locally (commonly `v2.0.1`). This alone is unlikely to explain the performance gap.

3) DropPath usage
- Answer: Not applied. Even though the CLI passes `--drop_path 0.2`, the custom model hard-codes `drop_path_rate=0.0` for both encoder and reconstructor in `EEGPT_mcae_finetune_change_tuev.py`.

4) Checkpoint initialization
- Answer: The code loads from `checkpoint['state_dict']` (file `../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`). The repo does not indicate whether this is the exact paper checkpoint or if it’s been fine-tuned on TUEV/event data. Unknown — please provide the exact file hash used in the paper.

5) Hidden data augmentation (Mixup, noise, jitter)
- Answer: None. Mixup is imported but never instantiated; Random Erase args exist but are unused; no other augmentations are applied in the training/evaluation loops.

6) Split seeds and protocol
- Answer: Preprocessing script creates train/val/test using subject-based split with `seed=4523` (20% of training subjects to val), and uses the dataset’s `eval` split as test. The training script then reads the resulting `processed_{train,eval,test}` folders. This matches the code here; we cannot assert it is identical to BIOT beyond the stated approach.

7) 62.32% BAC — averaging details
- Answer: The paper states “we repeated each experiment three times and calculated the standard deviation.” The 62.32% ± 1.14% appears to be the mean ± std across repeated runs (not folds), per `literature/EEGPT.md`.

8) Gradient clipping
- Answer: Disabled by default. `--clip_grad` exists but is not set in the launch script; code respects `None` (no clipping) in both DS and non-DS flows.

9) Channel mapper initialization (23→20 Conv2d)
- Answer: Weights use truncated normal initialization with `init_std=0.02` (see model `_init_weights` handling `nn.Conv2d`). No identity/orthogonal special init.

10) Early stopping / checkpoint selection
- Answer: No early stopping logic. The code DOES save best validation accuracy checkpoint when `max_accuracy < val_stats["accuracy"]` (saves as epoch="best"), but ONLY if `--save_ckpt` flag is provided (the launch script sets `--save_ckpt_freq` but not `--save_ckpt` by default). See lines 527-532 in `run_class_finetuning_EEGPT_change_tuev.py`.

11) Evaluation frequency during training
- Answer: Evaluates every epoch by default. The flag `--disable_eval_during_finetuning` is not used in the provided script, so validation/test run each epoch.

12) Reshape to (B,23,5,200) then flatten back
- Answer: This reshape is a no-op for the transformer path. The engine reshapes for compatibility, and the model immediately flattens back to `(B,23,1000)` before patching.

## Additional Clarifications

- Mixed precision:
  - DeepSpeed enabled: FP16 is handled by DS; inputs cast to half; no torch.amp context in train.
  - No DeepSpeed: train uses `torch.cuda.amp.autocast()` + NativeScaler; eval uses `autocast()`.
- Dropout placement: Transformer blocks use `drop_rate=0.0` and `attn_drop_rate=0.0`; heavy dropout (0.8) occurs in the channel mapper and classifier head only.
- CLI flags ignored in custom model path: `--abs_pos_emb`, `--disable_rel_pos_bias`, and `--drop_path` are parsed but not consumed by the finetune model constructor.
- Metrics: Computed via PyHealth’s `multiclass_metrics_fn`; balanced accuracy aligns with macro recall per standard definitions.
- No-weight-decay params: Encoder `{chan_embed, summary_token}`; reconstructor `{pos_embed, cls_token, time_embed, chan_embed}`.
- cuDNN determinism: `cudnn.benchmark=True` (faster, not fully deterministic); seeds reduce variance but do not guarantee bitwise reproducibility.

## Unknowns (need upstream confirmation)

- Exact checkpoint file used in the paper (name/hash) and whether it differs from `eegpt_mcae_58chs_4s_large4E.ckpt`.
- Which dataset path/version was used during the paper runs (v2.0.0 vs v2.0.1) and whether any external symlink/copy was applied.
- Whether the reported 62.32% used the best validation checkpoint (epoch="best") or the final epoch checkpoint.

## Actionable Next Steps

- Align preprocessing/training data paths (v2.0.0 vs v2.0.1) before parity runs.
- Enable checkpoint saving with `--save_ckpt` flag to persist both periodic and best validation checkpoints (paired with `--save_ckpt_freq`).
- To diagnose imbalance effects, try a temporary weighted sampler or minority oversampling and observe BAC/recall per class.
- Note: The best validation checkpoint (epoch="best") is automatically saved when validation accuracy improves, if `--save_ckpt` is enabled.

— End —
