# EEG Foundation Models Literature Collection

This directory contains markdown summaries of key research papers on EEG foundation models and related work.

## Papers

### 1. [ALFEE: Adaptive Large Foundation Model for EEG Representation](ALFEE/README.md)
- **Year**: 2025
- **Authors**: Xiong et al. (Tongji University)
- **Key Innovation**: Hybrid attention architecture separating channel and temporal processing
- **Scale**: 25,000 hours pretraining, up to 100M parameters
- **Performance**: State-of-the-art on 6 downstream tasks

### 2. [NeuroLM: Universal Multi-Task Foundation Model](NeuroLM/README.md)
- **Year**: 2025  
- **Authors**: Jiang et al. (SJTU & Microsoft Research)
- **Key Innovation**: Treats EEG as language for LLM processing
- **Scale**: 1.7B parameters (largest EEG model)
- **Performance**: 78% average across multiple tasks with single model

### 3. [EEGPT: EEG Conformer](../EEGPT-summary.md)
- **Year**: 2023
- **Authors**: Song et al.
- **Key Innovation**: Vision transformer adapted for EEG
- **Scale**: 10M parameters
- **Performance**: Baseline for many subsequent models

## Comparison Table

| Model | Parameters | Pretraining Hours | Multi-Task | Architecture | Avg Performance |
|-------|------------|-------------------|------------|--------------|-----------------|
| EEGPT | 10M | 10,000 | Limited | ViT | 65% |
| ALFEE | 100M | 25,000 | Yes | Hybrid Transformer | 72.5% |
| NeuroLM | 1.7B | 25,000 | Yes | LLM + VQ-VAE | 78% |

## Key Trends

1. **Scale**: Models growing from 10M → 1.7B parameters
2. **Data**: Pretraining datasets expanding to 25,000+ hours
3. **Architecture**: Evolution from CNN → Transformer → LLM integration
4. **Tasks**: Shift from single-task to multi-task foundation models
5. **Modality**: Movement toward multimodal (EEG + language) understanding

## Applications

- **Clinical**: Seizure detection, sleep staging, abnormality detection
- **Cognitive**: Emotion recognition, workload assessment, attention monitoring
- **BCI**: Motor imagery, P300 spellers, SSVEP interfaces
- **Research**: Cross-subject generalization, transfer learning

## Future Directions

1. **Multimodal Fusion**: Combining EEG with fMRI, MEG, EOG
2. **Real-time Processing**: Edge deployment for clinical monitoring
3. **Personalization**: Subject-specific fine-tuning
4. **Interpretability**: Understanding what models learn from EEG
5. **Privacy**: Federated learning for sensitive medical data

## Resources

- [EEGPT GitHub](https://github.com/chengstark/eegpt) - Original implementation
- [ALFEE GitHub](https://github.com/xw1216/ALFEE) - Hybrid attention model
- [NeuroLM GitHub](https://github.com/935963004/NeuroLM) - LLM-based approach

## How to Use This Collection

Each paper summary includes:
- Abstract and key contributions
- Technical architecture details
- Performance benchmarks
- Implementation notes
- Citation information

These summaries are designed to help researchers and engineers quickly understand the state-of-the-art in EEG foundation models and make informed decisions about which approaches to adopt for their specific use cases.