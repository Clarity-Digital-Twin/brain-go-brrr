#!/usr/bin/env bash
set -euo pipefail

cat <<'EOS'
TUSZ download helper
--------------------

The Temple University Seizure Corpus (TUSZ) requires a data use agreement.
Please follow the official instructions to obtain access:

  https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

Once you have the dataset, place it under:

  data/datasets/tusz/v2.0.1/

with train/dev/test patient splits intact. This project assumes a single
sampling rate (default 256 Hz) will be used downstream for temporal detection.

Note: This script is a placeholder to document the expected path layout.
It does not perform any downloads.
EOS

exit 0

