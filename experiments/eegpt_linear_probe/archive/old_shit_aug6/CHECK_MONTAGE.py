import mne
from pathlib import Path
import random

# Find some EDF files
edf_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf")
edf_files = list(edf_root.rglob("*.edf"))

# Sample a few files
for i, edf_file in enumerate(random.sample(edf_files, min(5, len(edf_files)))):
    print(f"\n{i+1}. {edf_file.name}")
    try:
        raw = mne.io.read_raw_edf(str(edf_file), verbose=False)
        print(f"   Channels ({len(raw.ch_names)}): {raw.ch_names[:10]}...")
        if 'T3' in raw.ch_names:
            print("   ⚠️  Uses OLD naming (T3/T4/T5/T6)")
        if 'T7' in raw.ch_names:
            print("   ✅ Uses MODERN naming (T7/T8/P7/P8)")
    except Exception as e:
        print(f"   Error: {e}")