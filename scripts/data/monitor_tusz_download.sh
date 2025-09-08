#!/bin/bash
# Monitor TUSZ download progress

echo "======================================"
echo "TUSZ Download Monitor"
echo "======================================"
echo ""

# Check if download is running
if tmux has-session -t tusz_download 2>/dev/null; then
    echo "✅ Download is ACTIVE in tmux session"
    echo ""
    
    # Count files downloaded so far
    EDF_COUNT=$(find data/datasets/tusz -name "*.edf" 2>/dev/null | wc -l)
    CSV_COUNT=$(find data/datasets/tusz -name "*.csv" 2>/dev/null | wc -l)
    
    echo "Progress so far:"
    echo "----------------"
    echo "EDF files: $EDF_COUNT"
    echo "CSV annotations: $CSV_COUNT"
    
    # Check disk usage
    if [ -d "data/datasets/tusz" ]; then
        SIZE=$(du -sh data/datasets/tusz 2>/dev/null | cut -f1)
        echo "Current size: $SIZE"
    fi
    
    echo ""
    echo "Commands:"
    echo "---------"
    echo "• Watch live:    tmux attach -t tusz_download"
    echo "• Detach:        Ctrl+B then D"
    echo "• Stop download: tmux kill-session -t tusz_download"
    
else
    echo "⚠️ Download is NOT running"
    
    # Check if data exists
    if [ -d "data/datasets/tusz" ]; then
        EDF_COUNT=$(find data/datasets/tusz -name "*.edf" 2>/dev/null | wc -l)
        
        if [ $EDF_COUNT -gt 0 ]; then
            echo ""
            echo "Downloaded data found:"
            echo "----------------------"
            echo "EDF files: $EDF_COUNT"
            
            SIZE=$(du -sh data/datasets/tusz 2>/dev/null | cut -f1)
            echo "Total size: $SIZE"
            
            echo ""
            echo "To resume download:"
            echo "uv run python scripts/data/download_tusz_auto.py"
        else
            echo ""
            echo "No data downloaded yet."
            echo ""
            echo "To start download:"
            echo "uv run python scripts/data/download_tusz_auto.py"
        fi
    else
        echo ""
        echo "TUSZ directory doesn't exist."
        echo ""
        echo "To start download:"
        echo "uv run python scripts/data/download_tusz_auto.py"
    fi
fi

echo ""
echo "======================================"