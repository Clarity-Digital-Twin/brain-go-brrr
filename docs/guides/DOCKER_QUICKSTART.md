# Docker Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- EEGPT model checkpoint downloaded to `data/models/pretrained/`

## Quick Start

1. **Start all services:**

   ```bash
   docker compose up -d
   ```

2. **Check service health:**

   ```bash
   docker compose ps
   curl http://localhost:8000/api/v1/health
   ```

3. **View logs:**

   ```bash
   docker compose logs -f api
   ```

4. **Stop services:**
   ```bash
   docker compose down
   ```

## Services

- **API**: FastAPI application on http://localhost:8000
- **Redis**: Cache and job queue on localhost:6379
- **Worker**: (Coming soon) Celery background processing

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.

## Development Mode

For development with hot-reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Troubleshooting

1. **Model not found**: Ensure EEGPT checkpoint is in `data/models/pretrained/`
2. **Port conflicts**: Change ports in docker-compose.yml if 8000 or 6379 are in use
3. **Memory issues**: Increase Docker memory allocation for large EEG files
