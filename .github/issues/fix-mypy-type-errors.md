# Fix MyPy Type Errors and Add Type Annotations

## Current Status
We have 22 mypy errors that need to be addressed. Most are related to missing type annotations and untyped imports.

## Common Issues
1. Missing type annotations for function arguments and return types
2. Untyped imports from external libraries (mne, yasa, etc.)
3. Type-arg issues with generic types
4. Missing stubs for some dependencies

## Tasks
- [ ] Add type annotations to all public functions
- [ ] Create `py.typed` marker file
- [ ] Add type stubs or ignore directives for untyped imports
- [ ] Fix generic type arguments (Dict -> Dict[str, Any], etc.)
- [ ] Update mypy configuration to handle external libraries
- [ ] Ensure all new code has proper type hints

## Suggested Approach
1. Start with core modules (config, logger, utils)
2. Move to models and services
3. Add ignore directives for problematic external imports
4. Run `make typecheck` after each module to verify fixes

## Priority
Medium - Important for code quality but not blocking functionality

@claude Please fix the mypy type errors by adding proper type annotations throughout the codebase and configuring mypy to handle external dependencies.