# External Reference Documentation

## ⚠️ IMPORTANT NOTE

These documents are **compiled summaries** based on:
1. Web-fetched documentation summaries from official sites
2. Local reference repositories in `/reference_repos/`
3. Code examples from the actual source code

**For authoritative, unmodified documentation, please visit:**
- MNE-Python Official Docs: https://mne.tools/stable/
- Autoreject Official Docs: https://autoreject.github.io/stable/

## Document Structure

### MNE Documentation
- `MNE_PREPROCESSING_GUIDE.md` - Comprehensive preprocessing techniques
- `MNE_AUTOREJECT_INTEGRATION.md` - How to use Autoreject with MNE

### Autoreject Documentation  
- `AUTOREJECT_COMPLETE_GUIDE.md` - Full Autoreject usage guide
- `AUTOREJECT_TUAB_SPECIFIC.md` - Settings for TUAB dataset

## Why Summaries Instead of Raw Downloads?

Due to web access limitations, we cannot download raw HTML/PDF documentation directly. Instead, we've created comprehensive summaries that:
- Cover all essential functionality
- Include accurate code examples
- Reference specific parameters and methods
- Are validated against the local source code in `/reference_repos/`

## Local Source Code References

For the actual implementation details, refer to:
```
/reference_repos/mne-python/    # Full MNE-Python source
/reference_repos/autoreject/     # Full Autoreject source (if available)
```

## Using These Documents

These summaries are designed to:
1. Provide quick reference for implementation
2. Guide integration into the Brain-Go-Brrr project
3. Highlight TUAB-specific considerations
4. Support the external auditor's review

For critical implementation decisions, always cross-reference with:
- Official documentation websites
- Source code in reference_repos
- Published papers (Gramfort et al. 2013, Jas et al. 2017)