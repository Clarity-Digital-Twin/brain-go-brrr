"""Edge case tests for EDF streaming to improve coverage."""

import numpy as np
import pytest

from brain_go_brrr.infra.data import edf_streaming as s


class _FakeReader:
    """Minimal fake reader for testing streaming without real EDF."""
    
    sfreq = 256
    n_channels = 20
    
    def __init__(self, n_samples=1024):
        self._n = n_samples
    
    def get_data(self, start=0, stop=None):
        stop = self._n if stop is None else stop
        if start < 0 or stop > self._n or start >= stop:
            raise ValueError("out of range")
        return np.zeros((self.n_channels, stop - start), dtype=np.float32)


def test_stream_chunks_beyond_eof_errors():
    """Test that requesting chunks beyond EOF raises ValueError."""
    rdr = _FakeReader(n_samples=1024)
    with pytest.raises(ValueError):
        # Request chunk size larger than available data
        list(s.stream_chunks(reader=rdr, chunk_size=2048, step=2048))


def test_stream_chunks_handles_final_partial_chunk():
    """Test graceful handling of final partial chunk at EOF."""
    rdr = _FakeReader(n_samples=1100)  # 1024 + partial 76
    chunks = list(s.stream_chunks(reader=rdr, chunk_size=512, step=512))
    
    # Extract chunk lengths
    lens = [c.shape[1] for c in chunks]
    
    # Expect 2 full chunks (512 each) + 1 partial (76)
    assert lens[:2] == [512, 512]
    assert lens[-1] == 76
    
    # No overrun, no extra chunk
    assert sum(lens) == 1100


def test_stream_chunks_exact_boundary():
    """Test when data ends exactly on chunk boundary."""
    rdr = _FakeReader(n_samples=1024)
    chunks = list(s.stream_chunks(reader=rdr, chunk_size=512, step=512))
    
    # Should have exactly 2 chunks of 512 each
    assert len(chunks) == 2
    assert all(c.shape[1] == 512 for c in chunks)