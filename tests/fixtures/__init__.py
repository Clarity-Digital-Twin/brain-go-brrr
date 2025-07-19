"""Test fixtures for brain-go-brrr.

CLEANED UP: Removed over-engineered mock EEGPT system following ML testing best practices.

Available fixtures:
- Real EEG data: tests/fixtures/eeg/*.fif (9 files with known labels)
- Simple mocks: For I/O testing only, not ML model behavior

See docs/TESTING_BEST_PRACTICES.md for guidance.
"""

# Only import what's actually needed and working
# Complex mock system has been skipped/removed

__all__ = [
    # Real EEG fixtures are in tests/fixtures/eeg/ directory
    # Simple mocks can be imported directly when needed
]
