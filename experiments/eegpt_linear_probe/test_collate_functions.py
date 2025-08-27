#!/usr/bin/env python3
"""Test collate functions for TUAB and TUEV datasets."""

import torch


def test_collate_functions():
    """Test that collate functions enforce correct channel counts."""
    
    print("Testing Collate Functions...")
    print("=" * 60)
    
    # Test TUAB collate
    from utils.collate_tuab import collate_tuab_batch
    
    # Test case 1: Normal TUAB batch (19 channels)
    batch_19 = [
        (torch.randn(19, 1024), 0.0),
        (torch.randn(19, 1024), 1.0),
    ]
    data, labels = collate_tuab_batch(batch_19)
    assert data.shape == (2, 19, 1024), f"TUAB: Expected (2, 19, 1024), got {data.shape}"
    assert labels.dtype == torch.float32, f"TUAB: Expected float32 labels, got {labels.dtype}"
    print("✅ TUAB collate: Handles 19-channel data correctly")
    
    # Test case 2: TUAB with 20-channel contamination (should truncate)
    batch_20_workaround = [
        (torch.randn(20, 1024), 0.0),
        (torch.randn(19, 1024), 1.0),
    ]
    data, labels = collate_tuab_batch(batch_20_workaround)
    assert data.shape == (2, 19, 1024), f"TUAB workaround: Expected (2, 19, 1024), got {data.shape}"
    print("✅ TUAB collate: Workaround handles 20→19 channel truncation")
    
    # Test case 3: TUAB with wrong channel count (should raise)
    batch_wrong = [(torch.randn(18, 1024), 0.0)]
    try:
        collate_tuab_batch(batch_wrong)
        assert False, "TUAB should have raised on 18 channels"
    except RuntimeError as e:
        assert "18" in str(e) and "19" in str(e)
        print("✅ TUAB collate: Correctly raises on wrong channel count (18)")
    
    # Test TUEV collate
    from utils.collate_tuev import collate_tuev_batch
    
    # Test case 4: Normal TUEV batch (20 channels)
    batch_tuev = [
        (torch.randn(20, 1024), 0),  # Class 0: SPSW
        (torch.randn(20, 1024), 5),  # Class 5: BCKG
    ]
    data, labels = collate_tuev_batch(batch_tuev)
    assert data.shape == (2, 20, 1024), f"TUEV: Expected (2, 20, 1024), got {data.shape}"
    assert labels.dtype == torch.long, f"TUEV: Expected long labels, got {labels.dtype}"
    print("✅ TUEV collate: Handles 20-channel data correctly")
    
    # Test case 5: TUEV with 19 channels (should raise - no workaround!)
    batch_tuev_wrong = [(torch.randn(19, 1024), 0)]
    try:
        collate_tuev_batch(batch_tuev_wrong)
        assert False, "TUEV should have raised on 19 channels"
    except RuntimeError as e:
        assert "19" in str(e) and "20" in str(e) and "EXACTLY" in str(e)
        print("✅ TUEV collate: STRICTLY enforces 20 channels (no workaround)")
    
    # Test case 6: TUEV label validation
    batch_tuev_bad_label = [
        (torch.randn(20, 1024), 6),  # Invalid class (> 5)
    ]
    try:
        collate_tuev_batch(batch_tuev_bad_label)
        assert False, "TUEV should have raised on invalid label"
    except ValueError as e:
        assert "0-5" in str(e)
        print("✅ TUEV collate: Validates label range (0-5)")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("  TUAB collate: 19ch enforcement + 20→19 workaround ✅")
    print("  TUEV collate: STRICT 20ch enforcement, no workarounds ✅")
    print("\nAll collate functions are CORRECT!")
    print("=" * 60)
    

if __name__ == "__main__":
    test_collate_functions()