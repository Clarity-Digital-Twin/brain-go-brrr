"""Compatibility shim for accuracy metrics helpers.

This file used to define a custom metrics recorder with side effects on import.
To avoid side effects and duplicate implementations, shared helpers now live in
`tests/_test_utils.py`.
"""
