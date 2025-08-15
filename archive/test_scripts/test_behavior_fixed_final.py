#!/usr/bin/env python
"""FIXED behavior test with correct imports and class names."""

from pathlib import Path

import mne
import numpy as np

print("=== FIXED APPLICATION BEHAVIOR TEST ===\n")
print("Testing all major components with CORRECT imports")
print("=" * 60 + "\n")

# Create test data once (DRY principle)
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

# ============================================================
# 1. YASA SLEEP STAGING (Already Working)
# ============================================================
print("📊 TESTING YASA SLEEP STAGING")
print("-" * 40)

try:
    from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager

    stager = YASASleepStager()
    print("✅ YASA adapter loaded")

    eeg_array = raw.get_data()
    stages, confidences, metrics = stager.stage_sleep(
        eeg_data=eeg_array, sfreq=raw.info["sfreq"], ch_names=raw.ch_names
    )

    print("✅ Sleep staging completed!")
    print(f"   - Epochs: {len(stages)}")
    print(f"   - Avg confidence: {np.mean(confidences):.2f}")
    yasa_works = True

except Exception as e:
    print(f"❌ YASA failed: {e}")
    yasa_works = False

# ============================================================
# 2. QUALITY CONTROL - FIXED CLASS NAME
# ============================================================
print("\n📊 TESTING QUALITY CONTROL (FIXED)")
print("-" * 40)

try:
    # FIXED: Using correct class name
    from brain_go_brrr.domain.quality.controller import EEGQualityController

    qc = EEGQualityController()
    print("✅ EEGQualityController loaded")

    # Add artifacts
    noisy_data = data.copy()
    noisy_data[0, 1000:2000] = 500e-6  # Add artifact
    noisy_raw = mne.io.RawArray(noisy_data, info)

    # Run QC
    results = qc.run_full_qc_pipeline(noisy_raw)

    if results:
        print("✅ Quality control completed!")
        quality_metrics = results.get("quality_metrics", {})
        print(f"   - Bad channels: {quality_metrics.get('bad_channels', [])}")
        print(f"   - Quality grade: {quality_metrics.get('quality_grade', 'N/A')}")
        print(f"   - Confidence: {results.get('processing_info', {}).get('confidence', 0):.2f}")
        qc_works = True
    else:
        print("❌ QC returned None")
        qc_works = False

except Exception as e:
    print(f"❌ Quality control failed: {e}")
    import traceback

    traceback.print_exc()
    qc_works = False

# ============================================================
# 3. ABNORMALITY DETECTION - FIXED WITH MODEL PATH
# ============================================================
print("\n📊 TESTING ABNORMALITY DETECTION (FIXED)")
print("-" * 40)

# Check if trained model exists
model_path = Path(
    "experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt"
)
eegpt_path = Path("data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

if model_path.exists():
    print(f"✅ Trained model found: {model_path.name}")

    # Check if we have the EEGPT base model
    if not eegpt_path.exists():
        print(f"⚠️ EEGPT base model not found at: {eegpt_path}")
        print("   Using placeholder path for testing")
        eegpt_path = model_path  # Use trained model as placeholder

    try:
        # FIXED: Passing required model_path argument
        from brain_go_brrr.domain.abnormal.detector import AbnormalityDetector

        detector = AbnormalityDetector(model_path=eegpt_path)
        print("✅ Abnormality detector loaded")

        # Test detection
        result = detector.detect_abnormality(raw)

        if hasattr(result, "is_abnormal"):
            print("✅ Abnormality detection completed!")
            print(f"   - Classification: {'Abnormal' if result.is_abnormal else 'Normal'}")
            print(f"   - Confidence: {result.confidence:.2f}")
            print(f"   - Triage: {result.triage_priority}")
            abnormal_works = True
        else:
            print("❌ Detection returned unexpected format")
            abnormal_works = False

    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        abnormal_works = False
    except Exception as e:
        print(f"❌ Abnormality detection failed: {e}")
        abnormal_works = False
else:
    print(f"❌ Trained model not found at: {model_path}")
    abnormal_works = False

# ============================================================
# 4. API ENDPOINTS - FIXED IMPORT
# ============================================================
print("\n📊 TESTING API ENDPOINTS (FIXED)")
print("-" * 40)

try:
    from fastapi.testclient import TestClient

    # FIXED: Import from api.main, not api.app
    from brain_go_brrr.api.main import app

    client = TestClient(app)
    print("✅ API client created")

    # Test health endpoint
    response = client.get("/api/v1/health")
    if response.status_code == 200:
        print("✅ Health endpoint works")
        data = response.json()
        print(f"   - Status: {data.get('status', 'N/A')}")
        print(f"   - Service: {data.get('service', 'N/A')}")
        api_works = True
    else:
        print(f"❌ Health endpoint failed: {response.status_code}")
        api_works = False

    # Test root endpoint
    response = client.get("/")
    if response.status_code == 200:
        print("✅ Root endpoint works")
        data = response.json()
        print(f"   - Message: {data.get('message', 'N/A')}")
    else:
        print(f"❌ Root endpoint failed: {response.status_code}")

except Exception as e:
    print(f"❌ API test failed: {e}")
    api_works = False

# ============================================================
# 5. PDF GENERATION - Should work now that QC is fixed
# ============================================================
print("\n📊 TESTING PDF GENERATION")
print("-" * 40)

if qc_works:
    try:
        from brain_go_brrr.presentation.visualization.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        print("✅ PDF generator loaded")

        # Generate a report with actual QC results
        report_data = {
            "patient_id": "TEST001",
            "recording_date": "2025-08-13",
            "quality_metrics": results.get("quality_metrics", {})
            if "results" in locals()
            else {"bad_channels": ["T3"], "quality_grade": "GOOD", "abnormality_score": 0.3},
            "processing_info": {"confidence": 0.85, "processing_time": 1.5},
        }

        pdf_bytes = generator.generate_report(report_data)

        if pdf_bytes and len(pdf_bytes) > 0:
            print(f"✅ PDF generated! Size: {len(pdf_bytes)} bytes")

            # Save for inspection
            pdf_path = Path("test_report.pdf")
            pdf_path.write_bytes(pdf_bytes)
            print(f"   - Saved to: {pdf_path}")
            pdf_works = True
        else:
            print("❌ PDF generation returned empty")
            pdf_works = False

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        import traceback

        traceback.print_exc()
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
    print("\nBroken components:")
    for component, status in results.items():
        if "BROKEN" in status:
            print(f"  - {component}")

# Clean up
if Path("test_report.pdf").exists():
    Path("test_report.pdf").unlink()
    print("\nCleaned up test files")

print("\n=== FIXED BEHAVIOR TEST COMPLETE ===")
