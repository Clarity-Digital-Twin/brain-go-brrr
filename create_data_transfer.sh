#\!/bin/bash
# Create transfer package with ALL important gitignored data

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="brain-go-brrr-DATA-ONLY-${TIMESTAMP}.tar.gz"

echo "Creating DATA TRANSFER package: ${PACKAGE_NAME}"
echo "This includes ALL important gitignored content:"
echo ""
echo "✓ EEGPT pretrained weights (973MB)"
echo "✓ TUAB preprocessed cache (1.1GB)"
echo "✓ Full TUAB dataset (79GB)"
echo "✓ Reference repos list (for cloning)"
echo "✓ Training logs/checkpoints"
echo "✓ .env file"
echo ""
echo "Total size: ~81GB (will compress to ~60-70GB)"
echo ""
echo "Starting compression (this will take a while)..."

# Create the comprehensive data package
tar -czf "${PACKAGE_NAME}" \
    --verbose \
    data/models/ \
    data/cache/ \
    data/datasets/ \
    experiments/eegpt_linear_probe/logs/ \
    .env \
    reference_repos_list.txt

echo ""
echo "Package created: ${PACKAGE_NAME}"
echo "Size: $(du -sh ${PACKAGE_NAME} | cut -f1)"
echo ""
echo "ON THE PC:"
echo "1. Clone the repo: git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git"
echo "2. Extract this data: tar -xzf ${PACKAGE_NAME}"
echo "3. Clone reference repos:"
echo "   cd reference_repos"
echo "   git clone https://github.com/ncclab/EEGPT.git"
echo "   git clone https://github.com/mne-tools/mne-python.git"
echo "   git clone https://github.com/raphaelvallat/yasa.git"
echo "   git clone https://github.com/autoreject/autoreject.git"
echo "4. Set up environment and run training"
