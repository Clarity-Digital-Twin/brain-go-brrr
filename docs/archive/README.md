# Archive Index

This directory contains historical documentation from the Brain-Go-Brrr project development. Documents are organized by category for easy navigation.

## Directory Structure

### 📅 01-checkpoints/
**Project milestones and progress reports**
- `CHECKPOINT_2025-07-29_FINAL.md` - Final July 2025 checkpoint with complete status
- `CHECKPOINT_2025-07-23.md` - Mid-July progress update
- `CHECKPOINT_2024-07-22.md` - Initial July 2024 checkpoint
- Various work summaries and progress reports
- `MVP_SUMMARY.md` - Original MVP implementation summary
- `TESTING_FIXES_SUMMARY.md` - Test suite fixes and improvements

### 🔍 02-investigations/
**Bug investigations and technical fixes**
- **NaN Issues**: Root cause analysis and fixes for NaN training crashes
- **Cache Issues**: Cache mismatch investigations and solutions
- **AutoReject**: Integration findings and critical issues
- **EEGPT**: Investigation findings and summary token fixes
- Various audit findings and fix summaries

### 📋 03-planning/
**Historical planning documents and roadmaps**
- **EEGPT Implementation**: Various roadmaps and implementation plans
- **Linear Probe**: Training plans and analysis
- **TDD Planning**: Test-driven development phases and specifications
- **Reference Repos**: Analysis and audit of reference implementations
- **Vertical Slice**: Strategy documents for MVP implementation
- Future work and optimization plans (not implemented)

### 🏗️ 04-refactoring/
**Architecture refactoring and clean architecture migration**
- **Clean Architecture**: Migration from core.* to domain/application/infra
- **Architecture Audits**: Multiple comprehensive audits and final state
- **Option A**: Investigation and execution of Option A refactoring
- **Migration Guides**: Step-by-step refactoring guides
- Release notes and PR descriptions

### 🧠 05-training/
**EEGPT model training documentation**
- `EEGPT_REALITY_CHECK.md` - Realistic assessment of EEGPT capabilities
- `EEGPT_MVP_REALISTIC.md` - Achievable MVP plan
- `EEGPT_CORRECTED_ROADMAP.md` - Corrected implementation roadmap
- `EEGPT_TDD_MVP_PLAN.md` - TDD approach for EEGPT
- `MONITOR_TRAINING.sh` - Training monitoring script
- `TRAINING_CACHE_STATUS.md` - Cache implementation status

### 🔧 06-technical-debt/
**Technical debt, cleanup, and legacy documentation**
- **Cleanup Plans**: Final boss cleanup and technical debt resolution
- **Technical Docs**: Benchmarks, literature references, development standards
- **Legacy**: Old README, directory structures, handoff prompts
- Various technical specifications and debt paydown summaries

### 📊 analysis/
**Current state analysis and reports**
- `CURRENT_STATE_ANALYSIS.md` - Latest project state assessment
- `CRITICAL_DIVE_SAFETY_REPORT.md` - Safety and compliance analysis
- `REFACTORING_INVESTIGATION.md` - Refactoring impact analysis
- `ROOT_DOCUMENTATION_GUIDE.md` - Documentation organization guide
- `tuev_verification_report.json` - TUEV dataset verification

### ⚠️ deprecated/
**Truly deprecated documents** (currently empty - items moved to appropriate categories)

## Navigation Tips

1. **Looking for latest status?** → Check `01-checkpoints/CHECKPOINT_2025-07-29_FINAL.md`
2. **Debugging an issue?** → Check `02-investigations/` for similar problems
3. **Understanding architecture?** → See `04-refactoring/ARCHITECTURE_FINAL_STATE.md`
4. **Training EEGPT?** → Start with `05-training/EEGPT_REALITY_CHECK.md`
5. **Historical context?** → Browse `03-planning/` for original plans

## Important Notes

- These documents are **historical** - always check main `/docs` for current information
- Many planned features were never implemented (marked in documents)
- FDA/clinical claims have been removed - this is research software only
- For current documentation, see parent directories:
  - `/docs/00-overview/` - Current project overview
  - `/docs/01-architecture/` - Current architecture
  - `/docs/02-implementation/` - Implementation guides

## Quick Stats
- **Total archived documents**: ~90 files
- **Date range**: July 2024 - August 2025
- **Major refactoring**: Clean Architecture migration (July 2025)
- **Key achievement**: 100% green CI/CD pipeline

---

*Last organized: August 21, 2025*