# 🎯 ROADMAP: From AUROC to Clinical Reality

## 📧 Canonical Expert Feedback (September 2, 2025)

**From Dr. Joseph Picone (TUH/NEDC):**
> "I think you need input from clinicians. The evaluation metric depends on the application. 
> For years, we have distributed our own scoring software that we believe addresses a problem 
> like seizure detection where segmentation and false positive rates are very important:
> https://isip.piconepress.com/publications/book_sections/2021/springer/metrics/
> 
> In terms of pipelines, I think what we lack is adequate annotated data... There is obviously 
> use for a portal that can analyze data without a need to train models, but moving these big 
> EEG files to/from such a portal is a problem."

**Mission**: Ship a clinically-useful EEG pipeline addressing these concerns:
1. ✅ Pick specific clinical application (TUAB abnormal detection first, TUSZ seizures next)
2. ⚡ Use the RIGHT metrics for EACH task (AUROC for TUAB, FA/24h for seizures)
3. 📦 Bring compute to data (local container, no cloud BS)

**Target**: Email expert reviewer in 60 days with working container + clinical metrics

---

## Phase 1: Foundation (Week 1-2) ✅ DONE
- [x] EEGPT integrated and working
- [x] TUAB dataset loading
- [x] Basic AUROC: 86.9% (EEGPT paper baseline)
- [x] Sleep staging with YASA: 87% accuracy
- [x] 899+ tests passing

## Phase 2: Clinical Metrics (Week 3-4) 🚧 CURRENT

### For TUAB (Abnormal Detection - Classification)
- [ ] Implement sensitivity/specificity curves
- [ ] Calculate balanced accuracy
- [ ] Find specificity at fixed sensitivity (90%, 95%)
- [ ] Generate ROC curves

### For TUSZ (Seizure Detection - Temporal Events) 
- [ ] Implement FA/24h calculation from predictions
- [ ] Add TAES/ATWV for time-aligned scoring
- [ ] Generate DET curves for operating point selection
- [ ] Calculate sensitivity at fixed FA/24h thresholds

### Key Code to Add:
```python
# brain_go_brrr/domain/metrics/classification.py (TUAB)
def calculate_specificity_at_sensitivity(predictions, labels, target_sensitivity=0.95):
    """For abnormal/normal classification"""
    threshold = find_threshold_for_sensitivity(predictions, labels, target_sensitivity)
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity, threshold

# brain_go_brrr/domain/metrics/temporal.py (TUSZ)
def calculate_fa_per_24h(predictions, labels, threshold, total_hours):
    """For seizure detection - the ONE metric clinicians care about"""
    false_positives = count_temporal_false_alarms(predictions, labels, threshold)
    return (false_positives / total_hours) * 24
```

## Phase 3: Local Deployment (Week 5-6)
- [ ] Create CLI command: `bgb eval tuab --metrics clinical`
- [ ] Build Docker container with all dependencies
- [ ] Create offline wheelhouse bundle
- [ ] Write DEPLOY_LOCAL.md guide
- [ ] Test on fresh machine with no internet

### Deployment Targets:
```bash
# For TUAB (abnormal detection)
docker pull ghcr.io/clarity-digital-twin/brain-go-brrr:latest
docker run -v /data/TUAB:/data:ro brain-go-brrr eval tuab \
  --data-root /data --out results/ --metrics auroc,bac,specificity_at_sens=0.95

# For TUSZ (seizure detection) - future
docker run -v /data/TUSZ:/data:ro brain-go-brrr eval tusz \
  --data-root /data --out results/ --metrics taes,atwv,fa_per_24h
```

## Phase 4: Clinical Validation (Week 7-8)
- [ ] Test on full TUAB canonical split
- [ ] Document specificity at multiple sensitivity levels
- [ ] Create comparison table vs classical methods
- [ ] Generate reproducible results bundle
- [ ] Package as one-line install script

### Success Metrics Tables:

#### TUAB (Abnormal Detection - Classification)
| Method | AUROC | Balanced Acc | Spec@95% Sens | Status |
|--------|-------|--------------|---------------|--------|
| Classical | ~75% | ~70% | ~60% | Baseline |
| EEGPT (paper) | 86.9% | 76.9% | ??? | Literature |
| **EEGPT (ours)** | **Target: 86%+** | **Target: 75%+** | **Target: 70%+** | **TODO** |

#### TUSZ (Seizure Detection - Temporal) [Future Work]
| Method | Sensitivity | FA/24h | TAES | Status |
|--------|-------------|--------|------|--------|
| Classical | 80% | 10-15 | ~0.6 | Baseline |
| **EEGPT (tuned)** | **95%** | **<10** | **>0.7** | **TARGET** |

## Phase 5: Expert Follow-up Email (Day 60)

### Email Template (TUAB Focus):
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

### Alternative Email Template (TUSZ Seizure Focus - Future):
```
Subject: Follow-up: TUSZ evaluation with time-aligned scoring

Following your guidance on seizure detection metrics, I've implemented 
TAES/ATWV scoring with false alarms per 24 hours as the primary constraint.

Results on TUSZ canonical split:
- At 95% sensitivity: X.X FA/24h
- TAES score: 0.XX
- DET curve attached showing operating points

Container runs locally: docker pull ghcr.io/...
```

---

## Stretch Goals (If Time Permits)
- [ ] Add TUSZ seizure detection with TAES/ATWV
- [ ] Implement TUEV event classification
- [ ] Create watch-folder daemon mode
- [ ] Build Apptainer/Singularity image for HPC

## Anti-Goals (What NOT to Do)
- ❌ NO cloud/SaaS features yet
- ❌ NO fancy UI/frontend
- ❌ NO authentication/user management
- ❌ NO trying to solve data annotation problem
- ❌ NO scope creep beyond clinical metrics

---

## Daily Check-in Questions
1. Does this help clinicians reduce false alarms?
2. Can this run on a hospital workstation?
3. Would Picone be impressed by the rigor?

## Resources
- Picone's metrics paper: `/literature/markdown/evaluation-metrics/picone-2021-objective-evaluation-metrics.md`
- EEGPT paper baseline: 86.9% AUROC, 76.9% BAC on TUAB
- Key metric distinctions:
  - **TUAB (abnormal)**: AUROC, BAC, Specificity@Sensitivity
  - **TUSZ (seizures)**: FA/24h, TAES, ATWV, time-aligned scoring
- Clinical acceptance for seizures: <10 FA/24h at >95% sensitivity

---

**Remember**: We're not building "AI for EEG" - we're building "FA/24h reducer for neurologists"

*Last Updated: September 2, 2025*
*Target Completion: November 1, 2025*