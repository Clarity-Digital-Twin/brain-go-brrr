#!/usr/bin/env bash
set -euo pipefail

# Guard against accidental reintroduction of 'Oz' in TUEV paths.
# Allows historical docs and changelogs.

if rg -n "\bOz\b" src tests experiments -S --no-heading | grep -v -E "historical|README|CHANGELOG" >/dev/null; then
  echo "Found 'Oz' references. Please remove Oz from TUEV channel specs." >&2
  rg -n "\bOz\b" src tests experiments -S --no-heading | grep -v -E "historical|README|CHANGELOG" || true
  exit 1
fi

echo "No forbidden 'Oz' references detected."
