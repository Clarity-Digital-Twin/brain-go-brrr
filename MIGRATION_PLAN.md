# Migration Plan: Consolidate to src/ Layout

## Current State
- `api/` (root) - FastAPI application
- `core/` (root) - NEW domain modules (jobs/, sleep/, quality/, etc.)
- `infra/` (root) - Infrastructure code (cache, redis)
- `src/brain_go_brrr/core/` - OLD config files (config.py, logger.py, abnormality_config.py)

## Migration Steps (NO CONFLICTS!)

### Phase 1: Move without conflicts
1. `git mv api src/brain_go_brrr/api`
2. `git mv infra src/brain_go_brrr/infra`

### Phase 2: Merge core directories
Since there are NO filename conflicts:
1. Move root core modules into src: `git mv core/* src/brain_go_brrr/core/`
2. This will add jobs/, sleep/, quality/, etc. alongside existing config.py, logger.py

### Phase 3: Fix imports
1. Replace `from api.` → `from brain_go_brrr.api.`
2. Replace `from core.` → `from brain_go_brrr.core.`
3. Replace `from infra.` → `from brain_go_brrr.infra.`
4. Replace `import api` → `import brain_go_brrr.api`
5. Replace `import core` → `import brain_go_brrr.core`
6. Replace `import infra` → `import brain_go_brrr.infra`

### Phase 4: Update configs
1. Update pyproject.toml
2. Update pytest.ini
3. Update mypy.ini

### Phase 5: Fix remaining issues
1. Remove sys.path hacks
2. Fix JobData type errors
3. Run tests