---
name: Claude Code Workflow Guide
about: Documentation for our autonomous development workflow
title: 'docs: Claude Code autonomous workflow implementation guide'
labels: documentation, automation
assignees: ''
---

## Claude Code Autonomous Workflow

This issue documents our autonomous development workflow using Claude Code GitHub Actions.

### How It Works

1. **Issue Creation**: Strategic issues are created for upcoming features
2. **Claude Trigger**: Issues tagged with `@claude` trigger autonomous PR creation
3. **Parallel Development**: While Claude works on PRs, human developers continue immediate tasks
4. **Review & Merge**: Completed PRs are reviewed and integrated

### Active Autonomous Tasks

The following issues have been created and tagged for autonomous PR generation:

1. **Memory Optimization** (#pending)
   - Implement streaming DataLoader for efficient processing
   - Reduce memory footprint to <1GB for 5-minute recordings

2. **DateTime Deprecation Fix** (#pending)
   - Update all deprecated datetime.utcnow() calls
   - Quick win for cleaner test output

3. **Bayesian Autoreject** (#pending)
   - Enhance artifact detection with adaptive thresholds
   - Improve QC accuracy by 5%+

4. **Sleep Event Detection** (#pending)
   - Add spindle and slow wave detection
   - Enhance sleep analysis metrics

5. **Report ID Refactor** (#pending)
   - Replace base64 PDFs with efficient report ID pattern
   - Improve API performance and mobile compatibility

### Workflow Benefits

- **Parallel Development**: PRs cook while we work on immediate tasks
- **Consistent Quality**: Claude follows our TDD and code standards
- **Accelerated Delivery**: Features ready for review when we return
- **Knowledge Capture**: All implementations follow our documentation

### Setup Checklist

- [x] GitHub Actions workflow created (`.github/workflows/claude.yml`)
- [x] Auto-documentation workflow added
- [x] Strategic issues created and tagged
- [ ] Repository secrets configured (requires admin):
  - `ANTHROPIC_API_KEY`
  - `APP_ID`
  - `APP_PRIVATE_KEY`
- [ ] Claude GitHub App installed on repository

### Next Steps

1. Admin needs to install Claude GitHub App and add secrets
2. Once configured, Claude will automatically start creating PRs
3. Continue with immediate development tasks
4. Return to review and merge completed PRs

### Model Configuration

We're using `claude-opus-4-20250514` (Opus 4) for maximum quality in autonomous PR creation.

---

This workflow enables efficient parallel development where strategic features are implemented autonomously while human developers focus on immediate priorities.