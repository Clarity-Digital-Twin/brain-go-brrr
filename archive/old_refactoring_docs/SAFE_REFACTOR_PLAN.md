# SAFE REFACTORING PLAN - ZERO BREAKAGE
*Branch: feature/architecture-refactor*
*Current Status: 694 TESTS PASSING ✅*

## 🟢 PHASE 1: RISK-FREE CLEANUP (Day 1)

### Delete These Files NOW (They're Dead Code):
```bash
# 1. Delete backup file (NOBODY uses this)
rm src/brain_go_brrr/services/yasa_adapter_original_backup.py
make test  # VERIFY: 694 tests pass ✅

# 2. Delete empty modules (literally empty)
rm -rf src/brain_go_brrr/inference/
rm -rf src/brain_go_brrr/config/
rm -rf src/brain_go_brrr/core/resources/
make test  # VERIFY: 694 tests pass ✅

# 3. Commit this safe change
git add -A
git commit -m "refactor: remove dead code - no functionality change"
git push
```

**IMPACT**: Zero. These files are not used anywhere.

## 🟡 PHASE 2: CONSOLIDATE YASA (Day 2)

### Current Situation:
```
yasa_adapter.py (16KB) - Used by some tests
yasa_adapter_enhanced.py (15KB) - Better implementation
yasa_adapter_original_backup.py (10KB) - Already deleted ✅
```

### Safe Consolidation:
```bash
# 1. Check which one tests actually use
grep -r "yasa_adapter" tests/ --include="*.py" | grep -v enhanced

# 2. If tests use yasa_adapter.py:
#    - Check if yasa_adapter_enhanced.py is compatible
#    - Run tests with enhanced version
#    - If pass, delete original

# 3. Or keep BOTH for now (safer)
#    - Move to infra/external/yasa/
#    - Keep both working
```

## 🟡 PHASE 3: ORGANIZE PREPROCESSING (Day 3-4)

### Current Mess:
```
core/preprocessing.py (300 lines) - Basic filters
preprocessing/
  ├── autoreject_adapter.py (229 lines)
  ├── chunked_autoreject.py (255 lines)
  ├── eeg_preprocessor.py (469 lines)
  ├── flexible_preprocessor.py (386 lines)
  └── sleep_preprocessor.py (126 lines)
```

### Safe Organization:
```bash
# 1. Move core/preprocessing.py content to preprocessing/basic.py
cp src/brain_go_brrr/core/preprocessing.py src/brain_go_brrr/preprocessing/basic.py

# 2. Update core/preprocessing.py to import from new location
cat > src/brain_go_brrr/core/preprocessing.py << 'EOF'
"""Deprecated: Use brain_go_brrr.preprocessing instead."""
import warnings
warnings.warn(
    "brain_go_brrr.core.preprocessing is deprecated. "
    "Use brain_go_brrr.preprocessing.basic instead.",
    DeprecationWarning,
    stacklevel=2
)
from brain_go_brrr.preprocessing.basic import *
EOF

# 3. Test - should still work
make test  # MUST PASS

# 4. Update imports gradually
# Find all imports: grep -r "from brain_go_brrr.core.preprocessing"
# Update them one by one, testing after each
```

## 🟡 PHASE 4: UNIFY CACHE (Day 5)

### Current Duplication:
```
api/cache.py - Redis implementation
infra/cache.py - Memory implementation
```

### Safe Unification:
```python
# 1. Create unified structure
mkdir -p src/brain_go_brrr/infra/cache

# 2. Move implementations
mv src/brain_go_brrr/api/cache.py src/brain_go_brrr/infra/cache/redis.py
mv src/brain_go_brrr/infra/cache.py src/brain_go_brrr/infra/cache/memory.py

# 3. Create factory
cat > src/brain_go_brrr/infra/cache/__init__.py << 'EOF'
from .redis import RedisCache
from .memory import MemoryCache

def get_cache(cache_type="redis", **kwargs):
    if cache_type == "redis":
        return RedisCache(**kwargs)
    return MemoryCache(**kwargs)

__all__ = ["get_cache", "RedisCache", "MemoryCache"]
EOF

# 4. Update API to use new cache
# In api files: from brain_go_brrr.infra.cache import get_cache
```

## 🟢 PHASE 5: SIMPLE MOVES (Day 6-7)

### Move Misplaced Files:
```bash
# These moves are safe - just updating locations

# 1. Data files belong in data/
git mv src/brain_go_brrr/core/edf_loader.py src/brain_go_brrr/data/
git mv src/brain_go_brrr/core/edf_validator.py src/brain_go_brrr/data/

# 2. Window extraction is preprocessing
git mv src/brain_go_brrr/core/window_extractor.py src/brain_go_brrr/preprocessing/

# 3. External compatibility
mkdir -p src/brain_go_brrr/infra/external
git mv src/brain_go_brrr/mne_compat.py src/brain_go_brrr/infra/external/

# 4. Update imports (automated)
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.edf_loader/from brain_go_brrr.data.edf_loader/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.edf_validator/from brain_go_brrr.data.edf_validator/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.core.window_extractor/from brain_go_brrr.preprocessing.window_extractor/g' {} \;
find . -name "*.py" -exec sed -i 's/from brain_go_brrr.mne_compat/from brain_go_brrr.infra.external.mne_compat/g' {} \;

# 5. Test after EACH move
make test  # MUST PASS after each
```

## 🔴 PHASE 6: CORE SPLIT (Week 2 - CAREFUL!)

### Current core/ Contents:
```
KEEP IN CORE (domain logic):
- abnormal/ - Detection logic ✓
- sleep/ - Sleep analysis ✓
- quality/ - QC logic ✓
- channels.py - Domain model ✓
- exceptions.py - Domain exceptions ✓

MOVE OUT OF CORE:
- config.py → config/
- abnormality_config.py → config/
- pipeline/ → application/
- jobs/ → application/
- features/ → preprocessing/
- snippets/ → preprocessing/
- logger.py → infra/
- preprocessing.py → preprocessing/ (already done)
- edf_*.py → data/ (already done)
- window_extractor.py → preprocessing/ (already done)
```

### Safe Split Approach:
```bash
# 1. Create new structure
mkdir -p src/brain_go_brrr/domain
mkdir -p src/brain_go_brrr/application

# 2. COPY first (don't move)
cp -r src/brain_go_brrr/core/abnormal src/brain_go_brrr/domain/
cp -r src/brain_go_brrr/core/sleep src/brain_go_brrr/domain/
cp -r src/brain_go_brrr/core/quality src/brain_go_brrr/domain/
cp src/brain_go_brrr/core/channels.py src/brain_go_brrr/domain/
cp src/brain_go_brrr/core/exceptions.py src/brain_go_brrr/domain/

# 3. Test with copies
# Update a few imports to test
# If works, then delete originals
```

## ✅ TESTING STRATEGY

### After EVERY Change:
```bash
# 1. Run unit tests
make test

# 2. Check coverage didn't drop
make test-cov

# 3. Run type checking
make type-check

# 4. Run linting
make lint

# ALL must pass before next change
```

### Rollback Plan:
```bash
# If anything breaks:
git stash  # Save current work
git checkout HEAD~1  # Go back one commit
make test  # Verify working again

# Then try smaller change
```

## 📊 VALIDATION CHECKLIST

### Before Each Phase:
- [ ] All tests passing (694/694)
- [ ] Coverage at 64.79%
- [ ] Create backup branch

### After Each Phase:
- [ ] All tests still passing
- [ ] Coverage maintained or improved
- [ ] No new type errors
- [ ] No new lint errors
- [ ] Commit with clear message

### Final Validation:
- [ ] All 694 tests passing
- [ ] Coverage ≥ 64%
- [ ] No dead code
- [ ] No duplicate implementations
- [ ] Clear module boundaries

## 🎯 PRINCIPLES TO FOLLOW

### SOLID Principles:
1. **S**ingle Responsibility - Each module does ONE thing
2. **O**pen/Closed - Extend, don't modify
3. **L**iskov Substitution - Subclasses work like base
4. **I**nterface Segregation - Small, focused interfaces
5. **D**ependency Inversion - Depend on abstractions

### Clean Code (Robert C. Martin):
1. **Names** - Reveal intent
2. **Functions** - Do one thing
3. **Comments** - Explain why, not what
4. **Formatting** - Consistent
5. **Error Handling** - Use exceptions

### Design Patterns:
1. **Factory** - For model creation
2. **Strategy** - For preprocessing options
3. **Adapter** - For external libraries
4. **Repository** - For data access
5. **Facade** - For complex subsystems

## 🚀 START NOW WITH:

```bash
# 1. Ensure on correct branch
git checkout feature/architecture-refactor

# 2. Create safety backup
git checkout -b refactor-backup-$(date +%Y%m%d)

# 3. Go back to feature branch
git checkout feature/architecture-refactor

# 4. Start with Phase 1 (risk-free)
rm src/brain_go_brrr/services/yasa_adapter_original_backup.py
make test

# If passes (it will):
git add -A
git commit -m "refactor: remove backup file"

# Continue with empty modules...
```

## ⚠️ WHAT NOT TO DO

### DON'T:
- ❌ Change logic while moving files
- ❌ Move multiple things at once
- ❌ Skip tests between changes
- ❌ Rename without updating imports
- ❌ Delete without checking usage

### DON'T RUSH:
- Take 2 weeks, not 2 days
- Test after every change
- Commit frequently
- Keep detailed notes
- Ask for review

## 💚 EXPECTED OUTCOME

### Week 1 End:
- Dead code removed ✓
- Duplicates consolidated ✓
- Files in right places ✓
- All tests still passing ✓

### Week 2 End:
- Clear module boundaries ✓
- No duplication ✓
- SOLID principles followed ✓
- Clean architecture ✓
- Professional codebase ✓

**THE CODE WORKS NOW - KEEP IT WORKING!**
