#!/usr/bin/env python
"""Complete behavior test for all major components."""

from pathlib import Path

import mne
import numpy as np

print("=== COMPREHENSIVE APPLICATION BEHAVIOR TEST ===\n")
print("Testing: YASA Sleep, Quality Control, Abnormality Detection")
print("=" * 60 + "\n")

# ============================================================
# 1. YASA SLEEP STAGING
# ============================================================
print("📊 TESTING YASA SLEEP STAGING")
print("-" * 40)

# Create test data
sfreq = 256
duration = 300  # 5 minutes
n_channels = 19
ch_names = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]

np.random.seed(42)
data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)

try:
    from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager

    stager = YASASleepStager()
    print("✅ YASA adapter loaded")

    # Test with numpy array (correct signature)
    eeg_array = raw.get_data()
    stages, confidences, metrics = stager.stage_sleep(
        eeg_data=eeg_array, sfreq=raw.info["sfreq"], ch_names=raw.ch_names
    )

    print("✅ Sleep staging completed!")
    print(f"   - Epochs: {len(stages)}")
    print(f"   - Stages: {set(stages)}")
    print(f"   - Avg confidence: {np.mean(confidences):.2f}")
    yasa_works = True

except Exception as e:
    print(f"❌ YASA failed: {e}")
    yasa_works = False

# ============================================================
# 2. QUALITY CONTROL (Autoreject)
# ============================================================
print("\n📊 TESTING QUALITY CONTROL")
print("-" * 40)

try:
    from brain_go_brrr.domain.quality.controller import QualityController

    qc = QualityController()
    print("✅ Quality Controller loaded")

    # Add some artifacts to the data
    noisy_data = data.copy()
    noisy_data[0, 1000:2000] = 500e-6  # Add artifact

    noisy_raw = mne.io.RawArray(noisy_data, info)

    # Run QC
    results = qc.run_full_qc_pipeline(noisy_raw)

    if results:
        print("✅ Quality control completed!")
        print(f"   - Bad channels: {results.get('quality_metrics', {}).get('bad_channels', [])}")
        print(
            f"   - Quality grade: {results.get('quality_metrics', {}).get('quality_grade', 'N/A')}"
        )
        qc_works = True
    else:
        print("❌ QC returned None")
        qc_works = False

except Exception as e:
    print(f"❌ Quality control failed: {e}")
    qc_works = False

# ============================================================
# 3. ABNORMALITY DETECTION
# ============================================================
print("\n📊 TESTING ABNORMALITY DETECTION")
print("-" * 40)

# Check if trained model exists
model_path = Path(
    "experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt"
)

if model_path.exists():
    print(f"✅ Trained model found: {model_path.name}")

    try:
        from brain_go_brrr.domain.abnormal.detector import AbnormalityDetector

        detector = AbnormalityDetector()
        print("✅ Abnormality detector loaded")

        # Test detection
        # Note: This might fail if model isn't integrated
        result = detector.detect_abnormality(raw)

        if result:
            print("✅ Abnormality detection completed!")
            print(f"   - Classification: {result.get('classification', 'N/A')}")
            print(f"   - Confidence: {result.get('confidence', 0):.2f}")
            abnormal_works = True
        else:
            print("❌ Detection returned None")
            abnormal_works = False

    except Exception as e:
        print(f"❌ Abnormality detection failed: {e}")
        print("   (Model may not be integrated into detector)")
        abnormal_works = False
else:
    print(f"❌ Trained model not found at: {model_path}")
    abnormal_works = False

# ============================================================
# 4. API ENDPOINTS
# ============================================================
print("\n📊 TESTING API ENDPOINTS")
print("-" * 40)

try:
    from fastapi.testclient import TestClient

    from brain_go_brrr.api.app import app

    client = TestClient(app)
    print("✅ API client created")

    # Test health endpoint
    response = client.get("/api/v1/health")
    if response.status_code == 200:
        print("✅ Health endpoint works")
        api_works = True
    else:
        print(f"❌ Health endpoint failed: {response.status_code}")
        api_works = False

    # Test info endpoint
    response = client.get("/")
    if response.status_code == 200:
        print("✅ Info endpoint works")
    else:
        print(f"❌ Info endpoint failed: {response.status_code}")

except Exception as e:
    print(f"❌ API test failed: {e}")
    api_works = False

# ============================================================
# 5. PDF GENERATION (if other tests pass)
# ============================================================
print("\n📊 TESTING PDF GENERATION")
print("-" * 40)

if qc_works:
    try:
        from brain_go_brrr.presentation.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        print("✅ PDF generator loaded")

        # Try to generate a report
        pdf_bytes = generator.generate_report(
            {
                "patient_id": "TEST001",
                "quality_metrics": {
                    "bad_channels": ["T3"],
                    "quality_grade": "GOOD",
                    "abnormality_score": 0.3,
                },
                "processing_time": 1.5,
            }
        )

        if pdf_bytes:
            print(f"✅ PDF generated! Size: {len(pdf_bytes)} bytes")
            pdf_works = True
        else:
            print("❌ PDF generation returned None")
            pdf_works = False

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        pdf_works = False
else:
    print("⚠️ Skipping PDF test (QC not working)")
    pdf_works = False

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("📋 BEHAVIOR TEST SUMMARY")
print("=" * 60)

results = {
    "YASA Sleep Staging": "✅ WORKING" if yasa_works else "❌ BROKEN",
    "Quality Control": "✅ WORKING" if qc_works else "❌ BROKEN",
    "Abnormality Detection": "✅ WORKING" if abnormal_works else "❌ BROKEN",
    "API Endpoints": "✅ WORKING" if api_works else "❌ BROKEN",
    "PDF Generation": "✅ WORKING" if pdf_works else "❌ BROKEN",
}

for component, status in results.items():
    print(f"{component:.<30} {status}")

working_count = sum(1 for v in results.values() if "WORKING" in v)
total_count = len(results)

print(f"\nOVERALL: {working_count}/{total_count} components working")

if working_count == total_count:
    print("\n🎉 ALL SYSTEMS OPERATIONAL! Application is FULLY FUNCTIONAL!")
else:
    print(f"\n⚠️ {total_count - working_count} components need fixing")

print("\n=== BEHAVIOR TEST COMPLETE ===")
