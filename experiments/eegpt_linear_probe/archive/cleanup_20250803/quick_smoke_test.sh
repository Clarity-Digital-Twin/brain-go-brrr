#!/bin/bash
# Quick smoke test to verify we're using the right config

echo "🧪 EEGPT Quick Smoke Test"
echo "========================="

# Check configs
echo -e "\n📄 Checking configs..."
echo "Good config (tuab_enhanced_config.yaml):"
grep -E "(use_cache|cached_dataset|window_duration|max_epochs)" experiments/eegpt_linear_probe/configs/tuab_enhanced_config.yaml || echo "Config not found!"

echo -e "\n❌ Bad config (tuab_cached_fast.yaml) - DO NOT USE:"
grep -E "(use_cache|cached_dataset|CachedDataset)" experiments/eegpt_linear_probe/configs/tuab_cached_fast.yaml 2>/dev/null || echo "Not found (good!)"

# Check for the right dataset class
echo -e "\n🔍 Checking train_enhanced.py uses correct dataset..."
grep -n "TUABEnhancedDataset" experiments/eegpt_linear_probe/train_enhanced.py | head -3

# Verify AutoReject config exists
echo -e "\n🤖 AutoReject config:"
ls -la experiments/eegpt_linear_probe/configs/tuab_enhanced_autoreject.yaml 2>/dev/null || echo "Not found!"

echo -e "\n✅ CORRECT COMMANDS:"
echo "# Baseline (no AutoReject):"
echo "EEGPT_CONFIG=configs/tuab_enhanced_config.yaml uv run python experiments/eegpt_linear_probe/train_enhanced.py"
echo ""
echo "# With AutoReject:"
echo "EEGPT_CONFIG=configs/tuab_enhanced_autoreject.yaml uv run python experiments/eegpt_linear_probe/train_enhanced.py"