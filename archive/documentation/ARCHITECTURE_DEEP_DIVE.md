# ARCHITECTURE DEEP DIVE - Brain-Go-Brrr
*Date: August 12, 2025*
*Branch: feature/architecture-refactor*
*Auditor Review: VALIDATED WITH CORRECTIONS*

## 🔍 COMPLETE MODULE ANALYSIS

### Module Statistics (ACTUAL COUNTS)
```
Module          Files  Lines  Subdirs  Complexity
─────────────────────────────────────────────────
api             18     2554   2        HIGH (routers + core)
core            26     5486   9        VERY HIGH (BLOATED)
data            5      1505   1        MEDIUM
models          8      2367   1        HIGH (duplicates)
preprocessing   6      1480   1        MEDIUM
services        4      1649   1        HIGH (3 YASA!)
tasks           3      483    1        LOW
training        2      469    1        LOW
infra           6      736    2        MEDIUM
utils           2      52     1        LOW
visualization   3      797    1        MEDIUM
─────────────────────────────────────────────────
TOTAL           83     17578  21
```

## 🔴 CRITICAL FINDINGS (AUDITOR VALIDATION)

### ✅ CONFIRMED ISSUES

#### 1. **YASA ADAPTER TRIPLICATION** (100% CONFIRMED)
```
services/yasa_adapter.py                 # 16514 bytes - MAIN
services/yasa_adapter_enhanced.py        # 15564 bytes - ENHANCED VERSION
services/yasa_adapter_original_backup.py # 9845 bytes  - BACKUP FILE IN PROD!
```
**IMPACT**: 41KB of duplicate code, maintenance nightmare, unclear which to use

#### 2. **CORE MODULE EXPLOSION** (26 FILES, 5486 LINES!)
The `core/` is doing EVERYTHING:
```
core/
├── abnormal/        # Domain logic (✓)
├── features/        # Feature extraction (should be in preprocessing)
├── jobs/            # Job management (should be in application)
├── pipeline/        # Orchestration (should be in application)
├── quality/         # QC logic (✓)
├── resources/       # EMPTY!
├── sleep/           # Domain logic (✓)
├── snippets/        # Feature extraction (should be in preprocessing)
├── abnormality_config.py  # Config (should be in config/)
├── channels.py            # Domain (✓)
├── config.py              # Config (duplicate with config/)
├── edf_loader.py          # Data loading (should be in data/)
├── edf_validator.py       # Data validation (should be in data/)
├── exceptions.py          # Domain (✓)
├── logger.py              # Infra (should be in infra/)
├── preprocessing.py       # DUPLICATE with preprocessing/!
└── window_extractor.py    # Feature extraction (should be in preprocessing/)
```

#### 3. **EMPTY/UNDERUTILIZED MODULES** (CONFIRMED)
```
inference/__init__.py  # 1 line - EMPTY
config/__init__.py     # 1 line - EMPTY
core/resources/        # EMPTY DIRECTORY
```

#### 4. **MISPLACED ROOT FILES** (CONFIRMED)
```
src/brain_go_brrr/
├── mne_compat.py  # 7715 bytes - Should be in infra/external/
├── _typing.py     # 5354 bytes - Should be in utils/types/
```

#### 5. **MODEL PROLIFERATION** (8 FILES, NO STRUCTURE)
```
models/
├── eegpt_architecture.py        # Base architecture
├── eegpt_model.py               # Main model
├── eegpt_wrapper.py             # Wrapper
├── eegpt_linear_probe.py        # Variant 1
├── eegpt_linear_probe_robust.py # Variant 2
├── eegpt_two_layer_probe.py     # Variant 3
├── linear_probe.py              # Generic probe
```
**ISSUE**: No clear hierarchy, duplicated probe logic

### ⚠️ AUDITOR CORRECTIONS/ADDITIONS

#### 1. **API Cache Duplication** (AUDITOR MISSED THIS!)
```
api/cache.py      # API-level cache
infra/cache.py    # Infrastructure cache
```
**ISSUE**: Two cache implementations! Should be ONE in infra with API using it

#### 2. **Preprocessing TRIPLE Location** (WORSE THAN AUDITOR SAID)
```
core/preprocessing.py         # Core preprocessing
preprocessing/*.py            # 6 files of preprocessing
core/features/extractor.py    # Also preprocessing!
```

#### 3. **Config QUADRUPLE Definition** (AUDITOR UNDERSTATED)
```
core/config.py              # Main config
core/abnormality_config.py  # Specific config
config/__init__.py          # Empty module
api/dependencies.py         # Also has config logic
```

## 📊 DEPENDENCY ANALYSIS (DEEP SCAN)

### Layer Violations Found
```python
# GOOD - No upward dependencies found:
core → api:     0 imports ✓
data → api:     0 imports ✓
models → api:   0 imports ✓

# BAD - Scattered dependencies:
api → schemas:  27 imports (tight coupling)
core → core:    43 imports (high internal coupling)
* → _typing:    12 imports (root file dependency)
```

### Import Hotspots (Most Imported)
1. `core.exceptions` - 37 imports
2. `api.schemas` - 27 imports
3. `core.config` - 19 imports
4. `_typing` - 12 imports
5. `core.channels` - 11 imports

## 🎯 VALIDATED REFACTORING TARGETS

### IMMEDIATE FIXES (NO RISK)
1. **Delete `yasa_adapter_original_backup.py`** - It's a backup file!
2. **Remove empty `inference/` module** - Unused
3. **Remove empty `config/` module** - Redundant with core/config.py
4. **Delete `core/resources/`** - Empty directory

### QUICK MOVES (LOW RISK)
```bash
# Data layer fixes
git mv src/brain_go_brrr/core/edf_loader.py src/brain_go_brrr/data/
git mv src/brain_go_brrr/core/edf_validator.py src/brain_go_brrr/data/

# Feature extraction consolidation
git mv src/brain_go_brrr/core/window_extractor.py src/brain_go_brrr/preprocessing/
git mv src/brain_go_brrr/core/features src/brain_go_brrr/preprocessing/

# Infrastructure moves
git mv src/brain_go_brrr/mne_compat.py src/brain_go_brrr/infra/external/
git mv src/brain_go_brrr/core/logger.py src/brain_go_brrr/infra/

# Type consolidation
mkdir -p src/brain_go_brrr/utils/types
git mv src/brain_go_brrr/_typing.py src/brain_go_brrr/utils/types/
```

### STRUCTURAL REFACTORING (MEDIUM RISK)

#### Target Structure (VALIDATED & REFINED)
```
src/brain_go_brrr/
├── domain/              # Pure business logic (from core/)
│   ├── abnormal/        # Abnormality detection domain
│   ├── sleep/           # Sleep analysis domain
│   ├── quality/         # Quality control domain
│   ├── channels.py      # Channel definitions
│   └── exceptions.py    # Domain exceptions
│
├── application/         # Use cases & orchestration
│   ├── pipelines/       # From core/pipeline/
│   ├── jobs/            # From core/jobs/
│   ├── training/        # From training/
│   └── tasks/           # From tasks/
│
├── presentation/        # UI layers
│   ├── api/             # Current api/
│   ├── cli/             # From cli.py
│   └── visualization/   # Current visualization/
│
├── infrastructure/      # External dependencies
│   ├── data/            # Current data/ + edf_*
│   ├── models/          # Current models/
│   ├── preprocessing/   # Current preprocessing/ + features/
│   ├── external/        # mne_compat, yasa adapters
│   ├── cache/           # Unified cache (from api/ + infra/)
│   ├── redis/           # Current infra/redis/
│   └── config/          # All config files
│
└── shared/              # Cross-cutting
    └── utils/           # Current utils/ + types/
```

## 🔧 EXACT REFACTORING COMMANDS

### Phase 1: Delete Dead Code (SAFE)
```bash
# Remove backup files and empty modules
rm src/brain_go_brrr/services/yasa_adapter_original_backup.py
rm -rf src/brain_go_brrr/inference/
rm -rf src/brain_go_brrr/config/
rm -rf src/brain_go_brrr/core/resources/

# Remove duplicate preprocessing
rm src/brain_go_brrr/core/preprocessing.py  # After verifying not used
```

### Phase 2: Mechanical Moves (LOW RISK)
```bash
# Create new structure
mkdir -p src/brain_go_brrr/{domain,application,presentation,infrastructure,shared}
mkdir -p src/brain_go_brrr/domain/{abnormal,sleep,quality}
mkdir -p src/brain_go_brrr/application/{pipelines,jobs,training,tasks}
mkdir -p src/brain_go_brrr/presentation/{api,cli,visualization}
mkdir -p src/brain_go_brrr/infrastructure/{data,models,preprocessing,external,cache,redis,config}
mkdir -p src/brain_go_brrr/shared/utils/types

# Move domain logic
git mv src/brain_go_brrr/core/abnormal src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/sleep src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/quality src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/channels.py src/brain_go_brrr/domain/
git mv src/brain_go_brrr/core/exceptions.py src/brain_go_brrr/domain/

# Move application logic
git mv src/brain_go_brrr/core/pipeline src/brain_go_brrr/application/pipelines/
git mv src/brain_go_brrr/core/jobs src/brain_go_brrr/application/
git mv src/brain_go_brrr/training/* src/brain_go_brrr/application/training/
git mv src/brain_go_brrr/tasks/* src/brain_go_brrr/application/tasks/
git mv src/brain_go_brrr/services/hierarchical_pipeline.py src/brain_go_brrr/application/pipelines/

# Move presentation
git mv src/brain_go_brrr/api src/brain_go_brrr/presentation/
git mv src/brain_go_brrr/cli.py src/brain_go_brrr/presentation/cli/
git mv src/brain_go_brrr/visualization src/brain_go_brrr/presentation/

# Move infrastructure
git mv src/brain_go_brrr/data/* src/brain_go_brrr/infrastructure/data/
git mv src/brain_go_brrr/models/* src/brain_go_brrr/infrastructure/models/
git mv src/brain_go_brrr/preprocessing/* src/brain_go_brrr/infrastructure/preprocessing/
git mv src/brain_go_brrr/core/edf_loader.py src/brain_go_brrr/infrastructure/data/
git mv src/brain_go_brrr/core/edf_validator.py src/brain_go_brrr/infrastructure/data/
git mv src/brain_go_brrr/core/window_extractor.py src/brain_go_brrr/infrastructure/preprocessing/
git mv src/brain_go_brrr/core/features src/brain_go_brrr/infrastructure/preprocessing/
git mv src/brain_go_brrr/core/snippets src/brain_go_brrr/infrastructure/preprocessing/
git mv src/brain_go_brrr/mne_compat.py src/brain_go_brrr/infrastructure/external/
git mv src/brain_go_brrr/services/yasa*.py src/brain_go_brrr/infrastructure/external/
git mv src/brain_go_brrr/infra/redis src/brain_go_brrr/infrastructure/
git mv src/brain_go_brrr/infra/cache.py src/brain_go_brrr/infrastructure/cache/
git mv src/brain_go_brrr/api/cache.py src/brain_go_brrr/infrastructure/cache/api_cache.py
git mv src/brain_go_brrr/core/logger.py src/brain_go_brrr/infrastructure/
git mv src/brain_go_brrr/core/config.py src/brain_go_brrr/infrastructure/config/
git mv src/brain_go_brrr/core/abnormality_config.py src/brain_go_brrr/infrastructure/config/

# Move shared utilities
git mv src/brain_go_brrr/utils/* src/brain_go_brrr/shared/utils/
git mv src/brain_go_brrr/_typing.py src/brain_go_brrr/shared/utils/types/
git mv src/brain_go_brrr/modules src/brain_go_brrr/shared/
git mv src/brain_go_brrr/infra/serialization.py src/brain_go_brrr/shared/utils/
git mv src/brain_go_brrr/infra/safe_load.py src/brain_go_brrr/shared/utils/

# Clean up old directories
rmdir src/brain_go_brrr/{core,services,tasks,training,data,models,preprocessing,infra,utils}
```

### Phase 3: Update Imports (AUTOMATED)
```bash
# Create import update script
cat > update_imports.py << 'EOF'
import os
import re
from pathlib import Path

REPLACEMENTS = [
    # Domain moves
    (r'from brain_go_brrr\.core\.abnormal', 'from brain_go_brrr.domain.abnormal'),
    (r'from brain_go_brrr\.core\.sleep', 'from brain_go_brrr.domain.sleep'),
    (r'from brain_go_brrr\.core\.quality', 'from brain_go_brrr.domain.quality'),
    (r'from brain_go_brrr\.core\.channels', 'from brain_go_brrr.domain.channels'),
    (r'from brain_go_brrr\.core\.exceptions', 'from brain_go_brrr.domain.exceptions'),

    # Application moves
    (r'from brain_go_brrr\.core\.pipeline', 'from brain_go_brrr.application.pipelines'),
    (r'from brain_go_brrr\.core\.jobs', 'from brain_go_brrr.application.jobs'),
    (r'from brain_go_brrr\.training', 'from brain_go_brrr.application.training'),
    (r'from brain_go_brrr\.tasks', 'from brain_go_brrr.application.tasks'),

    # Presentation moves
    (r'from brain_go_brrr\.api', 'from brain_go_brrr.presentation.api'),
    (r'from brain_go_brrr\.cli', 'from brain_go_brrr.presentation.cli'),
    (r'from brain_go_brrr\.visualization', 'from brain_go_brrr.presentation.visualization'),

    # Infrastructure moves
    (r'from brain_go_brrr\.data', 'from brain_go_brrr.infrastructure.data'),
    (r'from brain_go_brrr\.models', 'from brain_go_brrr.infrastructure.models'),
    (r'from brain_go_brrr\.preprocessing', 'from brain_go_brrr.infrastructure.preprocessing'),
    (r'from brain_go_brrr\.mne_compat', 'from brain_go_brrr.infrastructure.external.mne_compat'),
    (r'from brain_go_brrr\.infra\.redis', 'from brain_go_brrr.infrastructure.redis'),
    (r'from brain_go_brrr\.infra\.cache', 'from brain_go_brrr.infrastructure.cache'),

    # Shared moves
    (r'from brain_go_brrr\.utils', 'from brain_go_brrr.shared.utils'),
    (r'from brain_go_brrr\._typing', 'from brain_go_brrr.shared.utils.types'),
    (r'from brain_go_brrr\.modules', 'from brain_go_brrr.shared.modules'),
]

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    for old, new in REPLACEMENTS:
        content = re.sub(old, new, content)

    with open(filepath, 'w') as f:
        f.write(content)

# Update all Python files
for filepath in Path('src').rglob('*.py'):
    update_file(filepath)
for filepath in Path('tests').rglob('*.py'):
    update_file(filepath)
EOF

python update_imports.py
```

## 📋 VALIDATION CHECKLIST

### Pre-Refactor Metrics
- [ ] Tests passing: 694/694 ✓
- [ ] Coverage: 64.79% ✓
- [ ] Lint errors: 0 ✓
- [ ] Type errors: 0 ✓

### Post-Refactor Requirements
- [ ] Tests passing: 694/694
- [ ] Coverage: ≥ 64.79%
- [ ] Lint errors: 0
- [ ] Type errors: 0
- [ ] No circular imports
- [ ] Layer violations: 0

### Architecture Metrics
- [ ] Core module files: 26 → <10
- [ ] Duplicate implementations: 0
- [ ] Empty modules: 0
- [ ] Misplaced files: 0
- [ ] Clear layer boundaries: ✓

## 🚦 RISK MATRIX

| Change | Risk | Impact | Rollback |
|--------|------|--------|----------|
| Delete backup files | NONE | Clean codebase | N/A |
| Remove empty modules | LOW | Cleaner structure | Git revert |
| Move files to correct layer | MEDIUM | Clear boundaries | Git revert |
| Rename core → domain | MEDIUM | Better clarity | Find/replace |
| Consolidate caches | HIGH | Single source of truth | Keep both temporarily |
| Refactor probe models | HIGH | Cleaner API | Factory pattern |

## 📊 FINAL ASSESSMENT

### Auditor Assessment: **70% ACCURATE**

#### What They Got Right:
- ✅ YASA adapter duplication (3 files!)
- ✅ Core module bloat (26 files)
- ✅ Empty modules (inference, config)
- ✅ Misplaced files (mne_compat, _typing)
- ✅ Services/tasks/inference confusion

#### What They Missed:
- ❌ API/infra cache duplication
- ❌ Config quadruplication (4 places!)
- ❌ Preprocessing in 3 locations
- ❌ Core has 9 subdirectories (not mentioned)
- ❌ 83 total files (they didn't count)

#### What They Overstated:
- ⚠️ "Tree is in decent shape" - It's actually quite messy
- ⚠️ "Low-drama refactors" - Some are high risk
- ⚠️ Import violations aren't bad (they're actually good!)

## 🎯 RECOMMENDED ACTION PLAN

### Week 1: Quick Wins
1. Delete all backup/empty files
2. Move misplaced files
3. Consolidate duplicate preprocessing

### Week 2: Structure
1. Implement new directory structure
2. Update all imports
3. Add import-linter rules

### Week 3: Consolidation
1. Unify cache implementations
2. Create probe factory
3. Merge config files

### Week 4: Documentation
1. Update all docstrings
2. Create architecture diagrams
3. Write migration guide

## CONCLUSION

The codebase needs significant restructuring. The external auditor identified real issues but understated their severity. The refactoring is MORE complex than suggested, requiring careful planning and execution. However, the current architecture is functional and the proposed changes will significantly improve maintainability, testability, and clarity.

**Recommendation**: Proceed with refactoring in phases, validating tests after each phase.
