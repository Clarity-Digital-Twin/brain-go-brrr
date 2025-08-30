"""Compatibility shim for accuracy metrics helpers.

This file used to define a custom metrics recorder with side effects on import.
To avoid side effects and duplicate implementations, shared helpers now live in
`tests/_test_utils.py`.
"""

from tests._test_utils import record_accuracy_metric  # re-export for compatibility
