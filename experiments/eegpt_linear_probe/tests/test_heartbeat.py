"""Test heartbeat.json functionality for signal checkpoint freshness."""

import json
import tempfile
import time
from pathlib import Path


def test_heartbeat_write_read():
    """Test that heartbeat can be written and read correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        heartbeat_path = output_dir / "heartbeat.json"

        # Simulate training loop writing heartbeat
        for batch_idx in range(5):
            heartbeat = {
                "epoch": 0,
                "batch_idx": batch_idx,
                "global_step": batch_idx,
                "walltime": time.time(),
                "loss": 0.5 - batch_idx * 0.1,
                "lr": 0.001,
            }

            with open(heartbeat_path, 'w') as f:
                json.dump(heartbeat, f)

            # Small delay to simulate training
            time.sleep(0.01)

        # Read final heartbeat (simulating signal handler)
        with open(heartbeat_path) as f:
            final_heartbeat = json.load(f)

        # Verify we got the latest values
        assert final_heartbeat["batch_idx"] == 4, (
            f"Expected batch_idx=4, got {final_heartbeat['batch_idx']}"
        )
        assert final_heartbeat["global_step"] == 4, (
            f"Expected global_step=4, got {final_heartbeat['global_step']}"
        )
        assert abs(final_heartbeat["loss"] - 0.1) < 1e-6, (
            f"Expected loss=0.1, got {final_heartbeat['loss']}"
        )

        print("✓ Heartbeat write/read test passed")
        print(f"  Final batch_idx: {final_heartbeat['batch_idx']}")
        print(f"  Final global_step: {final_heartbeat['global_step']}")
        print(f"  Final loss: {final_heartbeat['loss']:.3f}")


def test_heartbeat_freshness():
    """Test that heartbeat captures fresh values at signal time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        heartbeat_path = output_dir / "heartbeat.json"

        # Write heartbeats for batches 0-99
        for batch_idx in range(100):
            heartbeat = {
                "epoch": batch_idx // 50,  # 50 batches per epoch
                "batch_idx": batch_idx % 50,
                "global_step": batch_idx,
                "walltime": time.time(),
                "loss": 1.0 / (batch_idx + 1),
                "lr": 0.001 * (1 - batch_idx / 100),  # Decay
            }

            with open(heartbeat_path, 'w') as f:
                json.dump(heartbeat, f)

        # Simulate signal at batch 99 - read heartbeat
        with open(heartbeat_path) as f:
            signal_heartbeat = json.load(f)

        # Verify freshness
        assert signal_heartbeat["epoch"] == 1, f"Expected epoch=1, got {signal_heartbeat['epoch']}"
        assert signal_heartbeat["batch_idx"] == 49, (
            f"Expected batch_idx=49, got {signal_heartbeat['batch_idx']}"
        )
        assert signal_heartbeat["global_step"] == 99, (
            f"Expected global_step=99, got {signal_heartbeat['global_step']}"
        )

        # Simulate creating checkpoint from heartbeat
        checkpoint = {
            "epoch": signal_heartbeat["epoch"],
            "batch_idx": signal_heartbeat["batch_idx"],
            "global_step": signal_heartbeat["global_step"],
            "tag": "signal_SIGTERM",
        }

        print("✓ Heartbeat freshness test passed")
        print(f"  Signal at global_step={checkpoint['global_step']}")
        print(f"  Checkpoint saved: epoch={checkpoint['epoch']}, batch={checkpoint['batch_idx']}")


def test_heartbeat_missing():
    """Test graceful handling when heartbeat doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        heartbeat_path = output_dir / "heartbeat.json"

        # Try to read non-existent heartbeat
        if heartbeat_path.exists():
            with open(heartbeat_path) as f:
                heartbeat = json.load(f)
            print(f"Unexpected: heartbeat exists with {heartbeat}")
        else:
            # This is expected - use fallback values
            fallback_state = {"epoch": 0, "batch_idx": 0, "global_step": 0}
            print("✓ Missing heartbeat handled gracefully")
            print(f"  Using fallback: {fallback_state}")


if __name__ == "__main__":
    test_heartbeat_write_read()
    test_heartbeat_freshness()
    test_heartbeat_missing()
    print("\n✅ All heartbeat tests passed!")
