# NeuroLM: A Universal Multi-Task Foundation Model for Bridging the Gap Between Language and EEG Signals

**Authors:** Wei-Bang Jiang¹, Yansen Wang², Bao-Liang Lu¹, Dongsheng Li²  
**Institutions:** ¹Shanghai Jiao Tong University, ²Microsoft Research Asia  
**Conference:** ICLR 2025  
**GitHub:** https://github.com/935963004/NeuroLM

## Abstract

NeuroLM is the first multi-task foundation model that leverages Large Language Models (LLMs) by treating EEG signals as a foreign language. This approach enables multi-task learning and inference capabilities without requiring full fine-tuning for each downstream task, addressing resource limitations and improving versatility.

## Key Innovations

1. **EEG as Language**: Treats EEG signals as a foreign language that LLMs can learn to understand
2. **Neural Tokenizer**: Text-aligned vector-quantized temporal-frequency prediction for discrete neural tokens
3. **Multi-Channel Autoregression**: LLM learns causal EEG information across channels
4. **Instruction Tuning**: Adapts to various downstream tasks through natural language instructions
5. **Record-Breaking Scale**: 1.7B parameters (largest EEG model to date)

## Architecture

### Three-Stage Training Pipeline

#### Stage 1: Neural Tokenizer Learning
- Vector-quantized (VQ) encoder
- Temporal-frequency prediction
- Discrete neural token generation
- Text alignment for cross-modal understanding

#### Stage 2: Causal EEG Modeling
- Frozen VQ encoder generates tokens
- LLM processes tokens via multi-channel autoregression
- Learns causal relationships in EEG signals
- Enables understanding of both EEG and language modalities

#### Stage 3: Multi-Task Instruction Tuning
- Natural language instructions for task specification
- Unified model for diverse EEG tasks
- Zero-shot and few-shot capabilities

## Model Variants

| Model | Parameters | Pretraining Data | Base LLM |
|-------|------------|------------------|----------|
| NeuroLM-Base | 125M | 10,000 hours | GPT-2 |
| NeuroLM-Large | 355M | 15,000 hours | GPT-2 Large |
| NeuroLM-XL | **1.7B** | **25,000 hours** | GPT-3 variant |

## Performance Benchmarks

Evaluated on 6 diverse downstream tasks:

1. **Abnormal Detection (LaBraM)**: Single-task baseline
2. **Emotion Recognition (CNN-Transformer)**: Single-task baseline
3. **Sleep Stage Classification (SPARCNET)**: Single-task baseline
4. **Event Type Classification (ST-Transformer)**: Single-task baseline
5. **Slowing Event Detection (FFCL)**: Single-task baseline
6. **Workload Detection**: Single-task baseline

NeuroLM shows consistent improvements across all tasks, with particularly strong performance in:
- Multi-task scenarios (single model for all tasks)
- Few-shot learning (limited training data)
- Zero-shot transfer (no task-specific training)

## Technical Details

### Input Processing
- **Channels**: Variable (16-256 channels)
- **Sampling Rate**: 100-1000 Hz
- **Window Size**: 1-30 seconds
- **Tokenization**: 512 discrete tokens per second

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 2e-4 with cosine annealing
- **Batch Size**: 512 (distributed across GPUs)
- **Hardware**: 32x NVIDIA A100 80GB GPUs
- **Training Time**: ~2 weeks for XL model

### Instruction Format
```
Task: [task_name]
Input: [EEG_tokens]
Instruction: [natural_language_instruction]
Output: [prediction]
```

## Comparison with Existing Methods

| Approach | Type | Multi-Task | Parameters | Avg Performance |
|----------|------|------------|------------|-----------------|
| Task-Specific CNNs | Single | ❌ | ~10M each | 65% |
| BIOT | Foundation | ❌ | 100M | 68% |
| LaBraM | Foundation | ❌ | 200M | 70% |
| EEGPT | Foundation | Limited | 500M | 72% |
| **NeuroLM-XL** | **Foundation** | **✅** | **1.7B** | **78%** |

## Key Advantages

1. **Resource Efficiency**: Single model for multiple tasks vs. separate models
2. **Instruction Following**: Natural language control of EEG analysis
3. **Cross-Modal Understanding**: Bridges EEG and language modalities
4. **Scalability**: Performance improves with model size
5. **Transfer Learning**: Strong zero-shot and few-shot capabilities

## Applications

### Clinical Applications
- Automated EEG report generation
- Multi-pathology screening
- Real-time seizure detection
- Sleep disorder diagnosis

### Research Applications
- Brain-computer interfaces
- Cognitive state monitoring
- Neurofeedback systems
- Cross-subject generalization studies

### Future Directions
- Multimodal fusion (EEG + fMRI + MEG)
- Real-time streaming analysis
- Personalized fine-tuning
- Federated learning for privacy

## Implementation Details

### Tokenizer Architecture
```python
class NeuralTokenizer:
    - VQ-VAE backbone
    - Codebook size: 8192
    - Latent dimension: 256
    - Temporal resolution: 512 tokens/second
```

### LLM Integration
```python
class NeuroLM:
    - Base: Pretrained LLM (GPT-2/3)
    - Adapter: EEG-specific layers
    - Cross-attention: Language-EEG alignment
    - Output heads: Task-specific predictors
```

## Citation

```bibtex
@inproceedings{jiang2025neurolm,
  title={NeuroLM: A Universal Multi-Task Foundation Model for Bridging the Gap Between Language and EEG Signals},
  author={Jiang, Wei-Bang and Wang, Yansen and Lu, Bao-Liang and Li, Dongsheng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Related Work

- **EEGPT**: Vision transformer for EEG (our baseline)
- **LaBraM**: Self-supervised EEG pretraining
- **BIOT**: Biological signal transformer
- **BrainBERT**: BERT-style EEG pretraining

## Acknowledgments

Work done during Wei-Bang's internship at Microsoft Research Asia. Supported by Microsoft Azure compute credits and NSFC grants.