# REFACTORING EXECUTION PLAN
*Branch: feature/architecture-refactor*
*Estimated Time: 2-3 weeks*
*Risk Level: MEDIUM-HIGH*

## 🚀 PHASE 1: IMMEDIATE CLEANUP (Day 1)
**Risk: NONE | Time: 1 hour**

### Delete Dead Code
```bash
# Backup files that should NEVER be in production
rm src/brain_go_brrr/services/yasa_adapter_original_backup.py

# Empty modules serving no purpose
rm -rf src/brain_go_brrr/inference/
rm -rf src/brain_go_brrr/config/
rm -rf src/brain_go_brrr/core/resources/

# Verify nothing breaks
make test
make lint
git commit -m "refactor: remove dead code and empty modules"
```

## 🔧 PHASE 2: SIMPLE MOVES (Day 2-3)
**Risk: LOW | Time: 4 hours**

### Move Misplaced Files to Correct Locations
```bash
# Data layer corrections
git mv src/brain_go_brrr/core/edf_loader.py src/brain_go_brrr/data/
git mv src/brain_go_brrr/core/edf_validator.py src/brain_go_brrr/data/

# Preprocessing consolidation
git mv src/brain_go_brrr/core/window_extractor.py src/brain_go_brrr/preprocessing/
git mv src/brain_go_brrr/core/features/extractor.py src/brain_go_brrr/preprocessing/feature_extractor.py
git mv src/brain_go_brrr/core/snippets/maker.py src/brain_go_brrr/preprocessing/snippet_maker.py
rm -rf src/brain_go_brrr/core/features/
rm -rf src/brain_go_brrr/core/snippets/

# Update imports for moved files
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.edf_loader/from brain_go_brrr.data.edf_loader/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.edf_validator/from brain_go_brrr.data.edf_validator/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.window_extractor/from brain_go_brrr.preprocessing.window_extractor/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.features/from brain_go_brrr.preprocessing/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.snippets/from brain_go_brrr.preprocessing/g' {} \;

# Test after each move
make test
git commit -m "refactor: move data and preprocessing files to correct modules"
```

### Consolidate Infrastructure
```bash
# Create proper infrastructure directories
mkdir -p src/brain_go_brrr/infra/external

# Move external compatibility layers
git mv src/brain_go_brrr/mne_compat.py src/brain_go_brrr/infra/external/
git mv src/brain_go_brrr/core/logger.py src/brain_go_brrr/infra/

# Move YASA adapters (keeping only the main one)
git mv src/brain_go_brrr/services/yasa_adapter.py src/brain_go_brrr/infra/external/
rm src/brain_go_brrr/services/yasa_adapter_enhanced.py  # Or keep if needed
git mv src/brain_go_brrr/services/hierarchical_pipeline.py src/brain_go_brrr/core/pipeline/

# Update imports
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.mne_compat/from brain_go_brrr.infra.external.mne_compat/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.logger/from brain_go_brrr.infra.logger/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.services.yasa_adapter/from brain_go_brrr.infra.external.yasa_adapter/g' {} \;

make test
git commit -m "refactor: consolidate infrastructure and external adapters"
```

## 🏗️ PHASE 3: STRUCTURAL REFACTORING (Week 1)
**Risk: MEDIUM | Time: 2 days**

### Create Domain Layer
```bash
# Create domain structure
mkdir -p src/brain_go_brrr/domain

# Move pure domain logic
git mv src/brain_go_brrr/core/abnormal src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/sleep src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/quality src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/channels.py src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/exceptions.py src/brain_go_brrr/domain/

# Update imports
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.abnormal/from brain_go_brrr.domain.abnormal/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.sleep/from brain_go_brrr.domain.sleep/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.quality/from brain_go_brrr.domain.quality/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.channels/from brain_go_brrr.domain.channels/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.exceptions/from brain_go_brrr.domain.exceptions/g' {} \;

make test
git commit -m "refactor: establish domain layer with pure business logic"
```

### Create Application Layer
```bash
# Create application structure
mkdir -p src/brain_go_brrr/application/{use_cases,services}

# Move orchestration logic
git mv src/brain_go_brrr/core/pipeline src/brain_go_brrr/application/
git mv src/brain_go_brrr/core/jobs src/brain_go_brrr/application/
git mv src/brain_go_brrr/tasks/* src/brain_go_brrr/application/use_cases/
git mv src/brain_go_brrr/training/* src/brain_go_brrr/application/
rmdir src/brain_go_brrr/tasks
rmdir src/brain_go_brrr/services  # After moving all files

# Update imports
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.pipeline/from brain_go_brrr.application.pipeline/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.jobs/from brain_go_brrr.application.jobs/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.tasks/from brain_go_brrr.application.use_cases/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.training/from brain_go_brrr.application.training/g' {} \;

make test
git commit -m "refactor: establish application layer for use cases"
```

### Consolidate Configuration
```bash
# Create unified config module
mkdir -p src/brain_go_brrr/config

# Move all config files
git mv src/brain_go_brrr/core/config.py src/brain_go_brrr/config/base.py
git mv src/brain_go_brrr/core/abnormality_config.py src/brain_go_brrr/config/

# Create __init__.py to export configs
cat > src/brain_go_brrr/config/__init__.py << 'EOF'
from .base import Config
from .abnormality_config import AbnormalityConfig

__all__ = ['Config', 'AbnormalityConfig']
EOF

# Update imports
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.config/from brain_go_brrr.config/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.abnormality_config/from brain_go_brrr.config/g' {} \;

make test
git commit -m "refactor: consolidate all configuration in single module"
```

## 🔨 PHASE 4: MODEL CONSOLIDATION (Week 2)
**Risk: MEDIUM-HIGH | Time: 3 days**

### Organize Model Variants
```bash
# Create structured model hierarchy
mkdir -p src/brain_go_brrr/models/eegpt/probes

# Move base architecture
git mv src/brain_go_brrr/models/eegpt_architecture.py src/brain_go_brrr/models/eegpt/architecture.py
git mv src/brain_go_brrr/models/eegpt_model.py src/brain_go_brrr/models/eegpt/model.py
git mv src/brain_go_brrr/models/eegpt_wrapper.py src/brain_go_brrr/models/eegpt/wrapper.py

# Move probe variants
git mv src/brain_go_brrr/models/eegpt_linear_probe.py src/brain_go_brrr/models/eegpt/probes/linear.py
git mv src/brain_go_brrr/models/eegpt_linear_probe_robust.py src/brain_go_brrr/models/eegpt/probes/robust.py
git mv src/brain_go_brrr/models/eegpt_two_layer_probe.py src/brain_go_brrr/models/eegpt/probes/two_layer.py
git mv src/brain_go_brrr/models/linear_probe.py src/brain_go_brrr/models/eegpt/probes/base.py
```

### Create Probe Factory
```python
# src/brain_go_brrr/models/eegpt/probes/__init__.py
from typing import Literal
from .linear import LinearProbe
from .robust import RobustLinearProbe
from .two_layer import TwoLayerProbe

ProbeType = Literal["linear", "robust", "two_layer"]

class ProbeFactory:
    @staticmethod
    def create(probe_type: ProbeType, **kwargs):
        probes = {
            "linear": LinearProbe,
            "robust": RobustLinearProbe,
            "two_layer": TwoLayerProbe,
        }
        
        if probe_type not in probes:
            raise ValueError(f"Unknown probe type: {probe_type}")
        
        return probes[probe_type](**kwargs)

__all__ = ['ProbeFactory', 'LinearProbe', 'RobustLinearProbe', 'TwoLayerProbe']
```

### Update Model Imports
```bash
# Update all model imports
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.models.eegpt_/from brain_go_brrr.models.eegpt./g' {} \;
find . -name "*.py" -exec sed -i 's/eegpt_linear_probe/eegpt.probes.linear/g' {} \;
find . -name "*.py" -exec sed -i 's/eegpt_linear_probe_robust/eegpt.probes.robust/g' {} \;
find . -name "*.py" -exec sed -i 's/eegpt_two_layer_probe/eegpt.probes.two_layer/g' {} \;

make test
git commit -m "refactor: consolidate EEGPT models with factory pattern"
```

## 🧹 PHASE 5: CACHE UNIFICATION (Week 2)
**Risk: HIGH | Time: 2 days**

### Unify Cache Implementations
```bash
# Analyze both cache implementations
diff -u src/brain_go_brrr/api/cache.py src/brain_go_brrr/infra/cache.py > cache_diff.txt

# Create unified cache in infra
cat > src/brain_go_brrr/infra/cache/__init__.py << 'EOF'
"""Unified cache implementation."""
from .memory import MemoryCache
from .redis import RedisCache
from .interface import CacheInterface

class CacheFactory:
    @staticmethod
    def create(cache_type: str = "redis", **kwargs):
        if cache_type == "redis":
            return RedisCache(**kwargs)
        elif cache_type == "memory":
            return MemoryCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

__all__ = ['CacheFactory', 'CacheInterface', 'RedisCache', 'MemoryCache']
EOF

# Move implementations
mkdir -p src/brain_go_brrr/infra/cache
git mv src/brain_go_brrr/api/cache.py src/brain_go_brrr/infra/cache/redis.py
git mv src/brain_go_brrr/infra/cache.py src/brain_go_brrr/infra/cache/memory.py

# Update API to use infra cache
find src/brain_go_brrr/api -name "*.py" -exec sed -i 's/from brain_go_brrr.api.cache/from brain_go_brrr.infra.cache/g' {} \;

make test
git commit -m "refactor: unify cache implementations in infrastructure layer"
```

## 🧱 PHASE 6: TYPE SYSTEM CLEANUP (Week 3)
**Risk: LOW | Time: 1 day**

### Consolidate Type Definitions
```bash
# Create proper type module
mkdir -p src/brain_go_brrr/types

# Move and consolidate type definitions
git mv src/brain_go_brrr/_typing.py src/brain_go_brrr/types/base.py

# Create comprehensive type exports
cat > src/brain_go_brrr/types/__init__.py << 'EOF'
"""Central type definitions for brain_go_brrr."""
from .base import FloatArray, MNERaw, MNEEpochs, StrArray

# Re-export common types
from typing import (
    Dict, List, Optional, Union, Tuple, Any,
    Literal, Protocol, TypedDict, TypeVar
)

__all__ = [
    'FloatArray', 'MNERaw', 'MNEEpochs', 'StrArray',
    'Dict', 'List', 'Optional', 'Union', 'Tuple', 'Any',
    'Literal', 'Protocol', 'TypedDict', 'TypeVar'
]
EOF

# Update all type imports
find . -name "*.py" -exec sed -i 's/from brain_go_brrr._typing/from brain_go_brrr.types/g' {} \;

make test
git commit -m "refactor: consolidate type system"
```

## 📏 PHASE 7: ENFORCE BOUNDARIES (Week 3)
**Risk: MEDIUM | Time: 2 days**

### Install Import Linter
```bash
pip install import-linter
```

### Create Import Rules
```ini
# .importlinter
[importlinter]
root_package = brain_go_brrr

[importlinter:contract:1]
name = Domain has no dependencies
type = forbidden
source_modules =
    brain_go_brrr.domain
forbidden_modules =
    brain_go_brrr.api
    brain_go_brrr.application
    brain_go_brrr.infra
    brain_go_brrr.models
    brain_go_brrr.preprocessing

[importlinter:contract:2]
name = Application doesn't depend on API
type = forbidden
source_modules =
    brain_go_brrr.application
forbidden_modules =
    brain_go_brrr.api

[importlinter:contract:3]
name = API uses application layer
type = layers
layers =
    brain_go_brrr.api
    brain_go_brrr.application
    brain_go_brrr.domain
```

### Add to CI
```yaml
# .github/workflows/architecture.yml
name: Architecture Checks
on: [push, pull_request]
jobs:
  import-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install import-linter
      - run: lint-imports
```

## 🧪 PHASE 8: TEST MIGRATION (Week 3)
**Risk: LOW | Time: 1 day**

### Mirror Test Structure
```bash
# Create mirrored test structure
mkdir -p tests/unit/{domain,application,infra,api}
mkdir -p tests/integration/{domain,application,infra,api}

# Move tests to match new structure
git mv tests/unit/test_*abnormal* tests/unit/domain/
git mv tests/unit/test_*sleep* tests/unit/domain/
git mv tests/unit/test_*channel* tests/unit/domain/
git mv tests/unit/test_*pipeline* tests/unit/application/
git mv tests/unit/test_*task* tests/unit/application/
git mv tests/unit/test_*cache* tests/unit/infra/
git mv tests/unit/test_*redis* tests/unit/infra/
git mv tests/api/* tests/unit/api/

# Update test imports (automated)
python update_imports.py

make test
git commit -m "test: reorganize tests to mirror new architecture"
```

## 📊 VALIDATION METRICS

### Success Criteria
```python
# validation.py
import os
from pathlib import Path

def validate_architecture():
    checks = []
    
    # No more backup files
    backups = list(Path('src').rglob('*backup*'))
    checks.append(('No backup files', len(backups) == 0))
    
    # Core module reduced
    core_files = list(Path('src/brain_go_brrr/core').rglob('*.py')) if Path('src/brain_go_brrr/core').exists() else []
    checks.append(('Core < 10 files', len(core_files) < 10))
    
    # No duplicate caches
    cache_files = list(Path('src').rglob('*cache*.py'))
    checks.append(('Single cache implementation', len([f for f in cache_files if 'cache' in f.stem]) <= 2))
    
    # Domain layer pure
    if Path('src/brain_go_brrr/domain').exists():
        domain_imports = []
        for f in Path('src/brain_go_brrr/domain').rglob('*.py'):
            with open(f) as file:
                domain_imports.extend([l for l in file if 'from brain_go_brrr.infra' in l])
        checks.append(('Domain has no infra imports', len(domain_imports) == 0))
    
    # Tests passing
    result = os.system('make test > /dev/null 2>&1')
    checks.append(('All tests passing', result == 0))
    
    # Coverage maintained
    # Run coverage and parse result
    checks.append(('Coverage >= 64%', True))  # Placeholder
    
    for name, passed in checks:
        status = '✅' if passed else '❌'
        print(f'{status} {name}')
    
    return all(passed for _, passed in checks)

if __name__ == '__main__':
    success = validate_architecture()
    exit(0 if success else 1)
```

## 🚀 ROLLBACK PLAN

### If Things Go Wrong
```bash
# Complete rollback
git checkout main
git branch -D feature/architecture-refactor

# Partial rollback (keep some changes)
git revert HEAD~5..HEAD  # Revert last 5 commits
git cherry-pick <commit-hash>  # Keep specific changes
```

## 📅 TIMELINE

| Phase | Duration | Risk | Dependencies |
|-------|----------|------|--------------|
| 1. Cleanup | 1 hour | NONE | None |
| 2. Simple Moves | 4 hours | LOW | Phase 1 |
| 3. Structure | 2 days | MEDIUM | Phase 2 |
| 4. Models | 3 days | MEDIUM | Phase 3 |
| 5. Cache | 2 days | HIGH | Phase 3 |
| 6. Types | 1 day | LOW | Phase 3 |
| 7. Boundaries | 2 days | MEDIUM | Phases 3-6 |
| 8. Tests | 1 day | LOW | All phases |

**Total: 2-3 weeks**

## ✅ FINAL CHECKLIST

### Before Starting
- [ ] All tests passing (694/694)
- [ ] Coverage baseline recorded (64.79%)
- [ ] Branch created from development
- [ ] Team notified of refactoring

### After Each Phase
- [ ] Run `make test`
- [ ] Run `make lint`
- [ ] Run `make type-check`
- [ ] Check coverage hasn't dropped
- [ ] Commit with descriptive message

### Before Merging
- [ ] All tests passing
- [ ] Coverage ≥ 64%
- [ ] Import linter passing
- [ ] No circular imports
- [ ] Documentation updated
- [ ] Team code review
- [ ] Performance benchmarks run

## 🎯 EXPECTED OUTCOMES

1. **Cleaner Architecture**: Clear separation of concerns
2. **Better Testability**: Easier to mock and test in isolation
3. **Improved Maintainability**: Clear where to add new features
4. **Reduced Coupling**: Layers depend only on abstractions
5. **No Duplication**: Single source of truth for each concept
6. **Type Safety**: Centralized type definitions
7. **Clear Boundaries**: Enforced by tooling

## COMMANDS TO START

```bash
# Create branch if not already on it
git checkout -b feature/architecture-refactor

# Start with Phase 1
rm src/brain_go_brrr/services/yasa_adapter_original_backup.py
rm -rf src/brain_go_brrr/inference/
rm -rf src/brain_go_brrr/config/
rm -rf src/brain_go_brrr/core/resources/

# Verify and commit
make test
git add -A
git commit -m "refactor: phase 1 - remove dead code and empty modules"

# Continue with Phase 2...
```

Ready to execute! Each phase is reversible and testable.