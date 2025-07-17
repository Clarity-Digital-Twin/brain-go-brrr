# Agentic Development Workflow for Brain-Go-Brrr

## Overview
This document outlines the optimal workflow for developing the Brain-Go-Brrr project using Claude and other AI coding assistants in 2025.

## Core Principles

### 1. Context is King
Before starting any task:
```bash
# Ensure Claude reads these files
/docs/literature-master-reference.md  # Technical specifications
/CLAUDE.md                           # Project rules
/docs/PRD-product-requirements.md    # What we're building
```

### 2. Think-First Development
Always start complex tasks with thinking:
```
"think hard about implementing sleep stage classification using EEGPT"
```

This triggers deeper analysis before coding begins.

### 3. Test-Driven Agentic Development (TDAD)

#### Step 1: Define the Test
```
"Write a test for a function that takes raw EEG data and returns sleep stages. 
The test should verify:
- Correct input validation
- Proper output format
- Edge cases (short recordings, missing channels)
- Performance (<30s for 8-hour recording)"
```

#### Step 2: Implement Against Tests
```
"Now implement the sleep staging function to pass these tests.
Use the YASA integration from /services/sleep_metrics.py as reference."
```

#### Step 3: Refactor with Confidence
```
"Refactor this implementation to use async processing.
Ensure all tests still pass."
```

## Specific Workflows

### 1. Adding a New EEG Analysis Feature

```markdown
1. "think about adding [FEATURE] to our EEG pipeline"
   
2. "Create comprehensive tests for [FEATURE] including:
    - Normal operation tests
    - Edge cases  
    - Error handling
    - Performance benchmarks"

3. "Implement [FEATURE] following our service pattern.
    Reference /services/qc_flagger.py for structure"

4. "Add documentation for [FEATURE] including:
    - Docstrings
    - Usage examples
    - Update README"

5. "Create an integration test that uses [FEATURE] 
    with actual Sleep-EDF data"
```

### 2. Debugging Complex Issues

```markdown
1. "analyze the error: [PASTE ERROR]
    Check if this relates to our EEG processing pipeline"

2. "think hard about potential root causes considering:
    - EEG data format issues
    - Model compatibility
    - Memory constraints
    - Async processing edge cases"

3. "Create a minimal reproduction test case"

4. "Fix the issue with comprehensive error handling"
```

### 3. Performance Optimization

```markdown
1. "Profile the current [FUNCTION] implementation.
    Create benchmarks for baseline performance"

2. "think harder about optimization strategies for [FUNCTION]
    considering:
    - Batch processing
    - GPU utilization
    - Memory efficiency
    - Caching opportunities"

3. "Implement optimizations while maintaining test coverage"

4. "Verify performance improvements with benchmarks"
```

## Advanced Techniques

### 1. Multi-File Refactoring
```
"Refactor the sleep analysis service to:
1. Extract interfaces to /src/brain_go_brrr/interfaces/
2. Move implementations to /src/brain_go_brrr/analyzers/
3. Update all imports
4. Ensure tests still pass"
```

### 2. Architecture Decisions
```
"ultrathink about converting our service architecture to event-driven.
Consider:
- Current service dependencies
- Async processing needs
- Scalability requirements
- Testing implications
Create an ADR (Architecture Decision Record) with recommendation"
```

### 3. Literature-Driven Implementation
```
"Read /literature/markdown/EEGPT/EEGPT.md section on preprocessing.
Implement the exact preprocessing pipeline described:
- Reference specific equations
- Match the paper's parameters
- Include citations in comments"
```

## Collaboration Patterns

### 1. Code Review Preparation
```
"Review my implementation of [FEATURE]:
- Check against SOLID principles
- Verify error handling
- Assess test coverage
- Identify potential edge cases
Provide specific improvement suggestions"
```

### 2. Documentation Generation
```
"Generate comprehensive documentation for [MODULE]:
- API documentation with examples
- Architecture diagrams using mermaid
- Performance characteristics
- Integration guide"
```

### 3. Research Integration
```
"Compare our implementation with the approach in 
/reference_repos/EEGPT/downstream/finetune_EEGPT_SleepEDF.py
Identify:
- Key differences
- Potential improvements
- Missing features
Create a plan to align with best practices"
```

## Anti-Patterns to Avoid

### 1. ❌ Jumping to Code
```
"Implement sleep staging"  # Too vague, no context
```

### 2. ❌ Ignoring Tests
```
"Add this feature quickly, we'll test later"
```

### 3. ❌ Overengineering
```
"Create a fully distributed microservices architecture"
# We're not there yet - follow ROUGH_DRAFT.md MVP approach
```

### 4. ❌ Copying Without Understanding
```
"Just copy the EEGPT implementation exactly"
# We need to adapt for our use case
```

## Productivity Commands

### Quick Iterations
```bash
# Alias for test-driven development
alias tdd="uv run pytest -xvs --tb=short"

# Watch mode for continuous testing
alias watch="uv run pytest-watch -- -xvs"
```

### Context Switching
When switching between features:
```
"Summarize the current state of [FEATURE] implementation.
What's done, what's pending, what are the blockers?"
```

### Progress Tracking
```
"Update the TodoWrite with:
- Completed: [WHAT WAS DONE]
- In Progress: [CURRENT TASK]
- Next: [UPCOMING TASKS]"
```

## Integration with Tools

### 1. With VSCode/Cursor
- Use command palette: "Claude: Ask about selected code"
- Highlight error messages before asking for help
- Use inline comments `// @claude: explain this`

### 2. With GitHub
```
"Create a PR description for [FEATURE] that includes:
- What changed and why
- Testing approach
- Performance impact
- Breaking changes
- Related issues"
```

### 3. With Documentation
```
"Update /docs/api-specification.md with the new endpoint:
- OpenAPI 3.0 specification
- Request/response examples
- Error codes
- Rate limiting details"
```

## Measuring Success

### Code Quality Metrics
- Test coverage >80%
- Type coverage 100%
- Linting errors: 0
- Performance benchmarks pass

### Development Velocity
- Features completed per sprint
- Time from test to implementation
- Bug discovery in testing vs production

### AI Assistance Effectiveness
- Fewer clarification requests needed
- Higher first-attempt success rate
- Reduced debugging time

## Continuous Improvement

After each feature:
```
"Reflect on the development process for [FEATURE]:
1. What worked well?
2. What was challenging?
3. How can we improve the workflow?
Update CLAUDE.md with learnings"
```

## Emergency Procedures

### When Claude Gets Confused
1. Clear context: Start fresh conversation
2. Provide explicit file paths
3. Break complex tasks into smaller steps
4. Use thinking triggers for clarity

### When Tests Fail Mysteriously
```
"Debug this test failure systematically:
1. Verify test isolation
2. Check for race conditions
3. Validate mock behaviors
4. Print intermediate states"
```

Remember: The goal is to augment human capabilities, not replace human judgment. Always review AI-generated code for safety, correctness, and adherence to medical software standards.