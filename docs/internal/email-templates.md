# Email Templates for Expert Follow-up

## TUAB Evaluation Follow-up

```
Subject: Follow-up: Local TUAB evaluation with clinical metrics

Hi [Expert Reviewer],

Following your guidance on application-specific metrics, I've implemented
a local evaluation pipeline for TUAB abnormal detection.

Results on TUAB canonical split:
- 86.X% AUROC (approaching EEGPT paper's 86.9%)
- 7X% balanced accuracy
- At 95% sensitivity: XX% specificity
- Docker container ready: docker pull ghcr.io/...

The pipeline runs locally where the data sits - no uploads needed.
Reproducible eval bundle attached.

Would love your thoughts on whether this meets clinical utility thresholds.

Best,
[Your name]
```

## TUSZ Seizure Detection Follow-up (Future)

```
Subject: Follow-up: TUSZ evaluation with time-aligned scoring

Following your guidance on seizure detection metrics, I've implemented
TAES/ATWV scoring with false alarms per 24 hours as the primary constraint.

Results on TUSZ canonical split:
- At 95% sensitivity: X.X FA/24h
- TAES score: 0.XX
- DET curve attached showing operating points

Container runs locally: docker pull ghcr.io/...

Would love your thoughts on clinical acceptance.

Best,
[Your name]
```

## Notes
- Replace [Expert Reviewer] with actual name
- Fill in actual metrics before sending
- Attach provenance.json and results bundle
- Keep email concise and metric-focused
