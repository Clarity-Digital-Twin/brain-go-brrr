# EEGPT Architecture Fix Plan
**Date**: 2025-08-18  
**Priority**: 🔴 CRITICAL - Blocking All EEGPT Work  
**Branch**: `fix/eegpt-feature-extraction-architecture`

## Phase 1: Immediate Stabilization (Today)

### 1.1 Clean Up Experiments Folder
```bash
# Move failed TUEV attempts to archive
experiments/eegpt_linear_probe/archive/failed_tuev/
  ├── eegpt_full_features.py
  ├── train_tuev_aligned_fixed.py
  └── output/tuev_FIXED_*/

# Keep working TUAB training
experiments/eegpt_linear_probe/
  ├── train_paper_aligned.py          # KEEP - TUAB working
  ├── train_paper_aligned_BULLETPROOF.py  # KEEP - TUAB backup
  └── output/tuab_*/                  # KEEP - TUAB results
```

### 1.2 Document Current State
- [x] Create EEGPT_EMBEDDING_INVESTIGATION.md
- [ ] Verify TUAB checkpoint performance
- [ ] Document exact features used in TUAB

## Phase 2: Create Proper Feature Extraction API

### 2.1 New Directory Structure
```
src/brain_go_brrr/models/eegpt/
├── __init__.py
├── base_model.py              # Move EEGTransformer here
├── feature_extractor.py       # NEW - Central feature API
├── normalization.py           # Move from wrapper
└── strategies/
    ├── __init__.py
    ├── summary.py             # Summary token extraction
    ├── pooling.py             # Channel/temporal pooling  
    ├── attention.py           # Attention-weighted features
    └── selection.py           # Feature selection methods
```

### 2.2 Feature Extractor Implementation

```python
# src/brain_go_brrr/models/eegpt/feature_extractor.py

from typing import Dict, Literal, Optional
import torch
import torch.nn as nn
from .base_model import EEGPTModel
from .strategies import (
    SummaryTokenStrategy,
    ChannelPoolingStrategy,
    TemporalPoolingStrategy,
    AttentionWeightedStrategy
)

class EEGPTFeatureExtractor:
    """Unified EEGPT feature extraction with multiple strategies.
    
    This is the SINGLE source of truth for EEGPT feature extraction.
    ALL experiments and production code MUST use this.
    """
    
    def __init__(self, checkpoint_path: str):
        """Initialize with pretrained EEGPT model."""
        self.model = EEGPTModel(checkpoint_path)
        self.model.eval()
        
        # Initialize all strategies
        self.strategies = {
            'summary': SummaryTokenStrategy(),
            'channel_pool': ChannelPoolingStrategy(),
            'temporal_pool': TemporalPoolingStrategy(),
            'attention': AttentionWeightedStrategy(),
        }
    
    def extract_all_representations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract ALL intermediate representations in one forward pass.
        
        This is the master method - extracts everything efficiently.
        
        Args:
            x: Input EEG (batch, channels, time)
            
        Returns:
            Dictionary with all representations:
            - 'patches': Raw patch embeddings (B, N*C, D)
            - 'summary': Summary tokens (B, 4, 512)
            - 'final': Final hidden states (B, N*C+4, D)
        """
        with torch.no_grad():
            # Get all intermediate outputs
            outputs = self.model.forward_with_intermediates(x)
        
        return {
            'patches': outputs['patch_embeddings'],      # Before transformer
            'summary': outputs['summary_tokens'],        # Final 4 tokens
            'final': outputs['final_hidden_states'],     # All tokens after transformer
        }
    
    def extract(
        self,
        x: torch.Tensor,
        strategy: Literal['summary', 'channel_pool', 'temporal_pool', 'attention'] = 'summary',
        **kwargs
    ) -> torch.Tensor:
        """Extract features using specified strategy.
        
        Args:
            x: Input EEG (batch, channels, time)
            strategy: Feature extraction strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            Features tensor with shape depending on strategy
        """
        # Get all representations in one pass
        representations = self.extract_all_representations(x)
        
        # Apply selected strategy
        return self.strategies[strategy].extract(representations, **kwargs)
    
    def get_feature_dim(self, strategy: str, **kwargs) -> int:
        """Get output dimension for given strategy."""
        return self.strategies[strategy].output_dim(**kwargs)
```

### 2.3 Strategy Implementations

```python
# src/brain_go_brrr/models/eegpt/strategies/summary.py

class SummaryTokenStrategy:
    """Original EEGPT approach - use summary tokens."""
    
    def extract(self, representations: Dict, pool: bool = True) -> torch.Tensor:
        """Extract summary token features.
        
        Args:
            representations: All model representations
            pool: Whether to average pool tokens (True) or concatenate (False)
            
        Returns:
            Features: (B, 512) if pooled, (B, 2048) if concatenated
        """
        summary = representations['summary']  # (B, 4, 512)
        
        if pool:
            return summary.mean(dim=1)  # (B, 512)
        else:
            return summary.flatten(1)  # (B, 2048)
    
    def output_dim(self, pool: bool = True) -> int:
        return 512 if pool else 2048
```

```python
# src/brain_go_brrr/models/eegpt/strategies/pooling.py

class ChannelPoolingStrategy:
    """Pool patches by channel - good for channel-specific patterns."""
    
    def extract(self, representations: Dict) -> torch.Tensor:
        """Pool patches by channel.
        
        Returns:
            Features: (B, 20*512) = (B, 10240)
        """
        patches = representations['patches']  # (B, N*C, D)
        B, NC, D = patches.shape
        
        # Reshape to (B, N, C, D)
        N = 16  # 1024/64 patches
        C = NC // N
        patches_reshaped = patches.view(B, N, C, D)
        
        # Pool over time dimension
        channel_features = patches_reshaped.mean(dim=1)  # (B, C, D)
        
        return channel_features.flatten(1)  # (B, C*D)
    
    def output_dim(self) -> int:
        return 20 * 512  # 10,240
```

## Phase 3: Fix TUAB and TUEV Training

### 3.1 Verify TUAB Performance
```python
# scripts/verify_tuab_performance.py
"""Verify that TUAB training actually achieved paper performance."""

from brain_go_brrr.models.eegpt import EEGPTFeatureExtractor

# Load checkpoint
checkpoint = torch.load('experiments/eegpt_linear_probe/output/tuab_*/best_model.pt')

# Check what features were used
print(f"Model architecture: {checkpoint['config']}")
print(f"Final metrics: {checkpoint['metrics']}")
print(f"Feature dimensions: {checkpoint['feature_dim']}")

# Re-evaluate on test set
# ...
```

### 3.2 Fix TUEV Training
```python
# experiments/eegpt_linear_probe/train_tuev_fixed.py
"""TUEV training with proper feature extraction."""

from brain_go_brrr.models.eegpt import EEGPTFeatureExtractor

class TUEVModel(nn.Module):
    def __init__(self, checkpoint_path: str, feature_strategy: str = 'channel_pool'):
        super().__init__()
        
        # Use unified feature extractor
        self.feature_extractor = EEGPTFeatureExtractor(checkpoint_path)
        self.feature_strategy = feature_strategy
        
        # Get feature dimensions
        feature_dim = self.feature_extractor.get_feature_dim(feature_strategy)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )
    
    def forward(self, x):
        # Extract features using selected strategy
        features = self.feature_extractor.extract(x, self.feature_strategy)
        return self.classifier(features)
```

## Phase 4: Systematic Feature Study

### 4.1 Benchmark Different Strategies
```python
# scripts/benchmark_feature_strategies.py
"""Systematically test different feature extraction strategies."""

strategies = ['summary', 'channel_pool', 'temporal_pool', 'attention']
tasks = ['TUAB', 'TUEV', 'Sleep', 'MI']

results = {}
for task in tasks:
    for strategy in strategies:
        model = create_model(strategy)
        metrics = train_and_evaluate(model, task)
        results[f"{task}_{strategy}"] = metrics

# Save results
pd.DataFrame(results).to_csv('feature_strategy_benchmark.csv')
```

### 4.2 Create Feature Selection Guidelines
```markdown
# EEGPT Feature Selection Guide

## Quick Reference
| Task Type | Classes | Recommended Strategy | Feature Dim | Expected Performance |
|-----------|---------|---------------------|-------------|---------------------|
| TUAB | 2 | summary (pooled) | 512 | AUROC ~0.85 |
| TUEV | 6 | channel_pool | 10,240 | BAcc ~0.62 |
| Sleep | 5 | temporal_pool | 8,192 | Kappa ~0.75 |
| MI | 4 | summary (concat) | 2,048 | Acc ~0.75 |

## Decision Tree
1. Binary classification? → Try summary (pooled) first
2. Multi-class with channel patterns? → Try channel_pool
3. Long temporal sequences? → Try temporal_pool
4. Small dataset? → Use fewer features to avoid overfitting
```

## Phase 5: Clean Up Technical Debt

### 5.1 Remove Duplicate Implementations
```bash
# Files to consolidate or remove
src/brain_go_brrr/infra/ml_models/
  ├── eegpt_model.py           # REMOVE - duplicate of wrapper
  ├── eegpt_classifier.py      # CONSOLIDATE into examples
  ├── eegpt_linear_probe.py    # CONSOLIDATE into examples
  └── eegpt_two_layer_probe.py # CONSOLIDATE into examples
```

### 5.2 Create Clear Examples
```
examples/eegpt/
├── binary_classification.py   # TUAB example
├── multiclass_events.py      # TUEV example  
├── sequence_classification.py # Sleep example
└── README.md                 # Usage guide
```

## Phase 6: Documentation and Testing

### 6.1 Comprehensive Documentation
```markdown
# docs/eegpt_architecture.md
- Architecture overview
- Feature extraction strategies
- Performance benchmarks
- Troubleshooting guide
```

### 6.2 Unit Tests
```python
# tests/test_eegpt_features.py
def test_summary_extraction():
    """Test summary token extraction."""
    
def test_channel_pooling():
    """Test channel pooling strategy."""
    
def test_feature_dimensions():
    """Verify all strategies return correct dimensions."""
    
def test_backward_compatibility():
    """Ensure old checkpoints still load."""
```

## Timeline

### Week 1 (Immediate)
- [x] Day 1: Investigation and documentation
- [ ] Day 2: Clean experiments folder, verify TUAB
- [ ] Day 3: Implement unified feature extractor
- [ ] Day 4: Test with TUEV using channel pooling
- [ ] Day 5: Achieve TUEV paper performance

### Week 2 (Consolidation)
- [ ] Benchmark all strategies
- [ ] Remove duplicate code
- [ ] Create examples
- [ ] Write documentation
- [ ] Add comprehensive tests

### Week 3 (Production Ready)
- [ ] Code review
- [ ] Performance optimization
- [ ] Create deployment guide
- [ ] Knowledge transfer

## Success Criteria

1. **TUEV achieves paper performance** (BAcc ≥ 0.60)
2. **Single feature extraction API** used everywhere
3. **Clear documentation** on feature selection
4. **No duplicate implementations**
5. **All tests passing**
6. **Reproducible results**

## Risk Mitigation

### If Paper Performance Cannot Be Achieved
1. Contact paper authors for clarification
2. Try ensemble of strategies
3. Consider fine-tuning (not just linear probe)
4. Document limitations clearly

### If Breaking Changes Required
1. Version the API (v1, v2)
2. Provide migration guide
3. Keep backward compatibility layer
4. Deprecate gradually

## Conclusion

This plan will:
1. **Fix the immediate crisis** (TUEV failure)
2. **Establish proper architecture** (unified API)
3. **Enable future development** (clear patterns)
4. **Reduce technical debt** (remove duplicates)
5. **Ensure reproducibility** (documentation + tests)

**Estimated effort**: 2-3 weeks
**Impact**: Unblocks all EEGPT development
**Priority**: CRITICAL - Do this before ANY other EEGPT work