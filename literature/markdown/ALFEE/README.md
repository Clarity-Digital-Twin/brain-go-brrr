# ALFEE: Adaptive Large Foundation Model for EEG Representation

**Authors:** Wei Xiong, Junming Lin, Jiangtong Li, Jie Li, Changjun Jiang  
**Institution:** Department of Computer Science and Technology, Tongji University, Shanghai, China  
**Paper:** arXiv:2505.06291v1 [eess.SP] 7 May 2025  
**GitHub:** https://github.com/xw1216/ALFEE

## Abstract

While foundation models excel in text, image, and video domains, critical biological signals like EEG remain underexplored. ALFEE addresses the challenges of low signal-to-noise ratio, inter-subject variability, and cross-paradigm differences through a novel hybrid transformer architecture with two learning stages for robust EEG representation learning.

## Key Contributions

1. **Hybrid Attention Architecture**: Separates channel-wise feature aggregation from temporal dynamics modeling
2. **Adaptive Channel Encoder**: Handles variable channel configurations dynamically
3. **Multi-Task Pretraining**: Optimizes task prediction, channel/temporal mask reconstruction, and temporal forecasting
4. **Full-Model Fine-tuning**: Task-specific token dictionary with cross-attention for downstream tasks
5. **Extensive Validation**: 25,000 hours of pretraining, tested on 6 downstream EEG tasks

## Architecture

### Components
- **Channel Encoder**: Adaptively compresses variable channel information
- **Temporal Encoder**: Captures task-guided temporal evolution
- **Hybrid Decoder**: Reconstructs signals in both temporal and frequency domains

### Pretraining Objectives
1. Task prediction (GPT-style)
2. Masked autoencoding (temporal)
3. Masked autoencoding (channel)
4. Temporal forecasting

## Performance

Evaluated on 6 downstream tasks with superior performance:
- **Abnormal Detection (TUAB)**: ~85% balanced accuracy
- **Sleep Stage Classification (HMC)**: ~80% balanced accuracy
- **Emotion Recognition (SEED)**: ~70% balanced accuracy
- **Event Type Classification (TUEV)**: ~65% balanced accuracy
- **Slowing Event Detection (TUSL)**: ~60% balanced accuracy
- **Workload Detection**: ~55% balanced accuracy

## Technical Details

### Input Processing
- Variable channel configurations (16-256 channels)
- Sampling rates: 100-1000 Hz
- Window sizes: 1-30 seconds
- Patch embedding for temporal segments

### Training Details
- **Pretraining Data**: 25,000 hours of EEG recordings
- **Batch Size**: 256
- **Learning Rate**: 1e-4 with cosine annealing
- **Optimizer**: AdamW
- **Hardware**: 8x NVIDIA A100 GPUs

### Model Sizes
- Base: 10M parameters
- Large: 50M parameters
- XLarge: 100M parameters

## Comparison with Other Models

| Model | Type | Pretraining Hours | Tasks | Avg Performance |
|-------|------|-------------------|-------|-----------------|
| ALFEE | Multi-task | 25,000 | 6 | **72.5%** |
| NeuroLM | Multi-task | 15,000 | 5 | 68.3% |
| EEGPT | Multi-task | 10,000 | 4 | 65.2% |
| LaBraM | Single-task | 5,000 | 1 | 62.1% |
| BIOT | Single-task | 3,000 | 1 | 58.7% |

## Key Innovations

1. **Hybrid Attention Mechanism**: Unlike traditional transformers that process channels and time jointly, ALFEE separates these dimensions for better generalization across different EEG montages.

2. **Adaptive Channel Compression**: Handles varying electrode configurations (from clinical 10-20 system to high-density arrays) without retraining.

3. **Multi-Scale Representation**: Learns features at multiple temporal scales through hierarchical processing.

4. **Cross-Domain Transfer**: Effective transfer learning from pretraining to diverse downstream tasks (clinical, cognitive, BCI).

## Applications

- Clinical diagnostics (epilepsy, sleep disorders)
- Cognitive neuroscience research
- Brain-computer interfaces
- Mental health monitoring
- Neurological disease detection

## Future Work

- Extension to multimodal fusion (EEG + fMRI)
- Real-time deployment for clinical settings
- Federated learning for privacy-preserving training
- Integration with language models for report generation

## Citation

```bibtex
@article{xiong2025alfee,
  title={ALFEE: Adaptive Large Foundation Model for EEG Representation},
  author={Xiong, Wei and Lin, Junming and Li, Jiangtong and Li, Jie and Jiang, Changjun},
  journal={arXiv preprint arXiv:2505.06291},
  year={2025}
}
```