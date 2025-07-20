# Update Dockerfile for New Directory Structure

## Problem
Dockerfile still references old directory structure before the migration to `src/brain_go_brrr/`.

## Current Issues
- Missing proper Python path setup
- May have incorrect COPY commands
- CMD path needs verification

## Solution
1. Update all COPY commands to use new paths
2. Set PYTHONPATH correctly
3. Verify uvicorn command path
4. Test container builds and runs properly

## Acceptance Criteria
- [ ] Docker build succeeds
- [ ] Container starts without import errors
- [ ] API endpoints accessible from container
- [ ] All services properly initialized

## Testing
```bash
docker build -t brain-go-brrr .
docker run -p 8000:8000 brain-go-brrr
curl http://localhost:8000/api/v1/health
```

@clod please work on this autonomously
