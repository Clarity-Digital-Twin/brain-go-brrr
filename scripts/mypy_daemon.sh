#!/bin/bash
# Fast MyPy daemon for development - avoids hangs!

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if dmypy is installed
if ! command -v dmypy &> /dev/null; then
    echo -e "${YELLOW}Installing mypy with daemon support...${NC}"
    uv pip install mypy[dmypy]
fi

case "$1" in
    start)
        echo -e "${GREEN}Starting MyPy daemon...${NC}"
        uv run dmypy start -- --config-file=mypy.ini
        ;;
    stop)
        echo -e "${YELLOW}Stopping MyPy daemon...${NC}"
        uv run dmypy stop
        ;;
    restart)
        echo -e "${YELLOW}Restarting MyPy daemon...${NC}"
        uv run dmypy restart -- --config-file=mypy.ini
        ;;
    check)
        shift
        if [ $# -eq 0 ]; then
            echo -e "${GREEN}Checking all modules with daemon...${NC}"
            uv run dmypy run -- src/brain_go_brrr
        else
            echo -e "${GREEN}Checking $@ with daemon...${NC}"
            uv run dmypy run -- "$@"
        fi
        ;;
    core)
        echo -e "${GREEN}Checking core modules (strict)...${NC}"
        uv run dmypy run -- src/brain_go_brrr/core
        ;;
    models)
        echo -e "${GREEN}Checking models (strict)...${NC}"
        uv run dmypy run -- src/brain_go_brrr/models
        ;;
    api)
        echo -e "${GREEN}Checking API modules...${NC}"
        uv run dmypy run -- src/brain_go_brrr/api
        ;;
    status)
        uv run dmypy status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|check [files]|core|models|api|status}"
        echo ""
        echo "Examples:"
        echo "  $0 start              # Start the daemon"
        echo "  $0 check              # Check all modules"
        echo "  $0 check src/file.py  # Check specific file"
        echo "  $0 core               # Check core modules (strict)"
        echo "  $0 models             # Check models (strict)"
        echo "  $0 api                # Check API modules"
        exit 1
        ;;
esac