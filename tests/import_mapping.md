# Test Import Guide

## From unit tests:
```python
from ..conftest import fixture_name
from ..fixtures import mock_data
from .._mocks import MockClass
```

## From integration tests:
```python
from ..conftest import fixture_name
from ..fixtures.mock_eeg_generator import generate_mock_eeg
```

## Common imports:
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
from brain_go_brrr.module_name import ClassName
```
