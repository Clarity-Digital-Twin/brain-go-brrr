---
name: Memory Optimization for EEG Processing
about: Optimize memory usage with streaming and batch processing
title: 'feat: Implement memory-efficient streaming for EEG data processing'
labels: enhancement, performance
assignees: ''
---

## Problem Statement
The current EEG processing pipeline loads entire recordings into memory, causing high memory usage (>2GB for 5-minute recordings). This limits scalability for longer recordings and concurrent processing.

## Requirements
1. Implement streaming window extraction using PyTorch DataLoader
2. Process EEG data in configurable batch sizes (default: 64 windows)
3. Free tensors after CPU transfer to reduce memory footprint
4. Support half-precision (FP16) mode via environment variable
5. Maintain backward compatibility with existing API

## Technical Approach
- Refactor `EEGPTModel.process_recording()` to accept iterables
- Create custom `EEGDataset` class for window extraction
- Use `DataLoader` with `num_workers=os.cpu_count()//2`
- Implement `torch.cuda.empty_cache()` after batch processing
- Add `EEGPT_USE_FP16` environment variable support

## Acceptance Criteria
- [ ] Memory usage stays under 1GB for 5-minute recordings
- [ ] Processing time remains within 2x of current baseline
- [ ] All existing tests pass
- [ ] New streaming tests achieve >90% coverage
- [ ] Documentation updated with streaming examples

@claude please implement this memory optimization feature following our TDD approach. Start by writing comprehensive tests for the streaming functionality, then implement the DataLoader-based solution.
