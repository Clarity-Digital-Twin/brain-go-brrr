#!/bin/bash
# Monitor TUEV cache building progress

echo "=========================================="
echo "📊 TUEV CACHE BUILD MONITOR"
echo "=========================================="
echo ""

while true; do
    # Get current progress
    PROGRESS=$(tail -5 logs/cache_build.log | grep "Caching train:" | tail -1)
    
    if [ -z "$PROGRESS" ]; then
        echo "Waiting for cache build to start..."
    else
        # Extract percentage and stats
        PCT=$(echo "$PROGRESS" | grep -oP '\d+%' | head -1)
        CURRENT=$(echo "$PROGRESS" | grep -oP '\d+/\d+' | cut -d'/' -f1)
        TOTAL=$(echo "$PROGRESS" | grep -oP '\d+/\d+' | cut -d'/' -f2)
        RATE=$(echo "$PROGRESS" | grep -oP '\d+\.\d+it/s')
        
        # Count cache files
        if [ -d "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13/tuev_train_cache" ]; then
            FILE_COUNT=$(ls -1 /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13/tuev_train_cache/*.pt 2>/dev/null | wc -l)
        else
            FILE_COUNT=0
        fi
        
        clear
        echo "=========================================="
        echo "📊 TUEV CACHE BUILD MONITOR"
        echo "=========================================="
        echo ""
        echo "🔄 TRAIN SPLIT PROGRESS:"
        echo "  Progress: $PCT ($CURRENT / $TOTAL windows)"
        echo "  Speed:    $RATE"
        echo "  Files:    $FILE_COUNT cached"
        echo ""
        
        # Check for eval split
        if [ -d "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13/tuev_eval_cache" ]; then
            EVAL_COUNT=$(ls -1 /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13/tuev_eval_cache/*.pt 2>/dev/null | wc -l)
            echo "📊 EVAL SPLIT:"
            echo "  Files: $EVAL_COUNT cached"
            echo ""
        fi
        
        # Check for index files
        if [ -f "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13/tuev_train_cache/index.json" ]; then
            echo "✅ Train index created!"
        fi
        if [ -f "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13/tuev_eval_cache/index.json" ]; then
            echo "✅ Eval index created!"
        fi
        
        echo ""
        echo "Last update: $(date '+%H:%M:%S')"
        echo "Press Ctrl+C to stop monitoring"
    fi
    
    sleep 5
done