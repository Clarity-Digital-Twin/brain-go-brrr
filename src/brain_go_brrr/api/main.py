"""Main entry point for Brain-Go-Brrr API."""

import uvicorn

from brain_go_brrr.api.app import create_app

# Create the application
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104 - Binding to all interfaces required for Docker deployment
