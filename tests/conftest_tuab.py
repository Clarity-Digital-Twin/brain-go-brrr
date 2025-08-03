"""Fixtures for fast TUAB testing."""
import pytest
from pathlib import Path
from brain_go_brrr.data.tuab_dataset import TUABDataset


@pytest.fixture(scope="session")
def tiny_tuab_root():
    """Path to tiny TUAB test dataset."""
    return Path(__file__).parent / "fixtures" / "tiny_tuab"


@pytest.fixture(scope="session")
def fast_tuab_dataset(tiny_tuab_root):
    """Create a fast TUAB dataset with tiny fixtures."""
    # This would load instantly since it only has 4 files
    return TUABDataset(
        root_dir=tiny_tuab_root,
        split="train",
        max_files=4,  # Extra safety
        cache_dir=None  # No caching needed for tiny dataset
    )