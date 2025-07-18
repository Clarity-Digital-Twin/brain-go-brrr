---
name: Refactor Base64 PDF to Report ID Pattern
about: Replace large base64 PDFs in JSON with report ID and separate endpoint
title: 'refactor: Implement report ID pattern for PDF delivery'
labels: refactoring, api, performance
assignees: ''
---

## Problem Statement
Current API returns base64-encoded PDFs directly in JSON responses, causing:
- Large response payloads (>1MB)
- Mobile client performance issues
- Memory overhead in API responses

## Requirements
1. Generate unique report IDs for each analysis
2. Store reports temporarily with configurable TTL
3. Create GET endpoint for report retrieval
4. Support multiple report formats (PDF, Markdown, JSON)
5. Add report metadata endpoint

## API Design
```yaml
# Current (problematic)
POST /api/v1/eeg/analyze/detailed
Response: {
  "report_pdf": "base64_encoded_huge_string...",
  ...
}

# New (efficient)
POST /api/v1/eeg/analyze/detailed
Response: {
  "report_id": "rpt_1234567890abcdef",
  "report_url": "/api/v1/reports/rpt_1234567890abcdef",
  "expires_at": "2025-01-20T12:00:00Z",
  ...
}

GET /api/v1/reports/{report_id}
Response: Binary PDF data with proper headers

GET /api/v1/reports/{report_id}/metadata
Response: {
  "format": "pdf",
  "size_bytes": 245678,
  "created_at": "2025-01-17T10:00:00Z",
  "expires_at": "2025-01-20T12:00:00Z"
}
```

## Implementation Details
- Use Redis or file-based storage for temporary reports
- Implement cleanup job for expired reports
- Add Content-Disposition headers for downloads
- Support range requests for large files
- Add compression for storage efficiency

## Acceptance Criteria
- [ ] JSON responses stay under 100KB
- [ ] Reports retrievable via separate endpoint
- [ ] Automatic cleanup of expired reports
- [ ] Backward compatibility with deprecation notice
- [ ] Tests for report lifecycle management
- [ ] Documentation updated with new flow

@claude please refactor the PDF delivery system to use report IDs instead of base64 encoding. Follow TDD by first writing tests for the new endpoints and report storage system. Ensure backward compatibility during transition.
