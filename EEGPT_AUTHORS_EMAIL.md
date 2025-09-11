# Email to EEGPT Authors — TUEV Reproduction Clarification

**To:** wangguangyu@stu.hit.edu.cn, 19S003002@stu.hit.edu.cn, malin_li@hit.edu.cn, 23b903096@stu.hit.edu.cn, congxu@hit.edu.cn, lihaifeng@hit.edu.cn

**Subject:** Unable to Reproduce TUEV ~62% BAC — Clarification on Reference Implementation Details

Dear Dr. Li, Guangyu Wang, and the EEGPT Team,

We are attempting to reproduce the TUEV event classification result (~62.32% balanced accuracy) reported in your EEGPT paper. Despite aligning our implementation with your reference code (data, model, and training), our runs consistently plateau around ~0.22–0.27 BAC, with near‑zero recall on minority classes (spsw/eyem/artf). We would appreciate your guidance on several points to resolve the gap.

## Setup Summary (Parity with the Reference)

### Data and Splits
- TUEV event segments using official train/eval directories; train ≈ 4213 segments, eval ≈ 1471 segments.
- Extraction: 5‑second windows at 200 Hz; triple‑signal buffer; slice `[start−2 s : end+2 s]` around the annotated event; data stored in Volts.
- Labels: strings→0–5 mapping (`{'spsw':0,'gped':1,'pled':2,'eyem':3,'artf':4,'bckg':5}`), equivalent to the reference’s 1–6→0–5.
- Filtering: 0.1–75 Hz bandpass + 50 Hz notch; resample to 200 Hz; referential channels only (no bipolar montage).

### Model
- 23→20 mapper: Conv2dWithConstraint(23→20) → BatchNorm → GELU → depthwise Conv2d(1×55, groups=20) → BatchNorm → Dropout(0.8).
- EEGPT configured for 20 channels, `time_steps=1000`, `patch_stride=64`; flattens temporal summary tokens (15×4×512 = 30,720).
- Classifier head: Dropout(0.8) → LinearWithConstraint(30720→6, max_norm=1).
- DropPath not applied (custom finetune model hard‑codes `drop_path_rate=0.0`); we match this behavior.
- Checkpoint: `eegpt_mcae_58chs_4s_large4E.ckpt`, loaded from `checkpoint['state_dict']` with encoder/target_encoder key stripping.

### Training
- AdamW lr=5e‑4 (eps=1e‑8, betas=0.9/0.999); weight decay=0.05 with no‑weight‑decay for embeddings/tokens (e.g., `summary_token`, `chan_embed`).
- Per‑iteration cosine LR (warmup=5, min LR=1e‑6); layer‑wise LR decay=0.65.
- Loss: LabelSmoothingCrossEntropy(0.1) for training; CrossEntropyLoss for evaluation.
- Mixed precision: AMP + GradScaler; seed=0; `cudnn.benchmark=True`.
- Natural sampling (no class weights or oversampling); effective batch ≈ 400 via accumulation.

## Observed Results
- Best BAC = 0.2247 (22.47%); minority recalls ~0; gped/bckg dominate. We can share best/final confusion matrices and full logs on request.

## Questions
1. Was any class balancing used (weighted loss, oversampling, or balanced batches)?
2. Is `eegpt_mcae_58chs_4s_large4E.ckpt` the exact checkpoint used for TUEV (could you share a hash)? Any TUEV‑specific pretraining?
3. Were any augmentations applied (e.g., mixup, noise, time‑shifts), especially for the rare classes?
4. Which processed dataset path/version was used for TUEV (e.g., `v2.0.0` vs `v2.0.1`), and were there symlinks/copies between them?
5. Did you evaluate with CrossEntropyLoss (no smoothing) while training with LabelSmoothing (0.1)?
6. Can you confirm DropPath was effectively 0.0 in the finetune path?
7. The paper states mean ± std over repeated runs — were TUEV results averaged over three runs?

We appreciate your time and would be happy to provide logs and code pointers to accelerate debugging. Thank you for sharing your work.

Best regards,
[Your Name]
[Your Institution]
[Contact info]

---

## Attachments Available Upon Request
- Full training logs (e.g., `experiments/eegpt_linear_probe/logs/tuev_parity_20250911_112431.log`)
- Implementation and hyperparameter details
- Class distributions and confusion matrices over epochs
- Reference vs. reproduction comparison notes
