# Global CLAUDE.md Template for AI-Agentic Development

*Note: This is a template for ~/.claude/CLAUDE.md - place in your home directory for global rules*

## Universal Development Principles

### 1. Research Before Coding
- Always understand the problem domain first
- Read existing code before writing new code
- Check for similar implementations in the codebase
- Use `think hard` for architectural decisions

### 2. Safety First
- Never expose credentials or API keys
- Validate all user inputs
- Handle errors gracefully
- Return safe defaults when uncertain
- Log security events appropriately

### 3. Code Quality Standards
- Type hints in Python, TypeScript everywhere
- Meaningful variable names (no single letters except loops)
- Functions do ONE thing (Single Responsibility)
- Comments explain WHY, not WHAT
- Tests for every public function

### 4. Performance Awareness
- Profile before optimizing
- Use async/await for I/O operations
- Batch operations when possible
- Cache expensive computations
- Monitor memory usage in data processing

## Language-Specific Rules

### Python
```python
# Always use
from pathlib import Path  # not os.path
import logging           # not print() for debugging
from typing import ...   # full type annotations

# Project structure
src/package_name/        # source code
tests/                   # mirrors src structure
docs/                    # documentation
scripts/                 # utility scripts
```

### JavaScript/TypeScript
```typescript
// Prefer
const/let over var
async/await over promises.then()
?.optional?.chaining
interface over type for objects
```

### Shell/Bash
```bash
#!/usr/bin/env bash
set -euo pipefail  # Always include
use shellcheck     # Validate scripts
```

## AI-Agentic Workflow

### 1. Planning Phase
```
think about the requirements and architecture
- Understand the full scope
- Identify dependencies
- Consider edge cases
- Plan test scenarios
```

### 2. Implementation Phase
```
# TDD Approach
1. Write failing test
2. Implement minimal code to pass
3. Refactor for clarity
4. Document complex logic
```

### 3. Verification Phase
```
- Run all tests
- Check type safety
- Verify error handling
- Review security implications
```

## Git Best Practices
```bash
# Commit messages
feat: add new feature
fix: resolve bug
docs: update documentation
test: add test cases
refactor: improve code structure
chore: update dependencies

# Branch naming
feature/description
bugfix/issue-number
refactor/component-name
```

## Documentation Standards
- README.md: Project overview and setup
- CONTRIBUTING.md: Development guidelines
- API.md: API documentation
- CHANGELOG.md: Version history
- CLAUDE.md: AI agent instructions

## Testing Philosophy
1. Unit tests for logic
2. Integration tests for workflows
3. E2E tests for critical paths
4. Property-based tests for edge cases
5. Benchmark tests for performance

## Debugging Approach
1. Reproduce the issue
2. Isolate the problem
3. Form hypothesis
4. Test hypothesis
5. Implement fix
6. Verify fix doesn't break other things

## Communication
- Be concise in responses
- Show, don't just tell (include code examples)
- Explain tradeoffs in decisions
- Suggest alternatives when rejecting ideas
- Ask for clarification when ambiguous

## Resource Management
- Close files and connections
- Clean up temporary files
- Release memory explicitly for large data
- Use context managers (with statements)
- Monitor resource usage

## Security Checklist
- [ ] No hardcoded secrets
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS prevention (web apps)
- [ ] Proper authentication
- [ ] Authorization checks
- [ ] Audit logging
- [ ] Error messages don't leak info

## Performance Checklist
- [ ] Algorithms are appropriate O(n)
- [ ] Database queries optimized
- [ ] Caching implemented where needed
- [ ] Pagination for large datasets
- [ ] Async for I/O operations
- [ ] Memory usage bounded

## Universal Commands
```bash
# These should work in most projects
make test          # Run tests
make lint          # Check code style
make format        # Format code
make build         # Build project
make clean         # Clean artifacts
```

## When to Ask for Human Input
- Architectural decisions with long-term impact
- Security-sensitive implementations
- Performance vs. readability tradeoffs
- Breaking changes to APIs
- Decisions affecting user data

## Remember
- Code is read more than written
- Explicit is better than implicit
- Simple is better than complex
- Errors should never pass silently
- Now is better than never
- Although never is often better than *right* now

*Adapt these rules to specific project needs in project-level CLAUDE.md files*