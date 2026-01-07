#!/bin/bash
# Run feedback loop comparison between original and reimplementation
#
# Usage:
#   bash scripts/compare/run_feedback_loop.sh           # Run both versions
#   bash scripts/compare/run_feedback_loop.sh original  # Run original only
#   bash scripts/compare/run_feedback_loop.sh reimpl    # Run reimplementation only
#
# Output: data/{original,reimplementation}/feedback_loop/

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

VERSION="${1:-both}"

run_original() {
    echo "=============================================="
    echo "Running ORIGINAL feedback loop..."
    echo "=============================================="
    rm -rf data/original/feedback_loop 2>/dev/null || true
    time uv run python src/scripts/train_original.py configs/original/feedback_loop.yaml
    echo ""
}

run_reimpl() {
    echo "=============================================="
    echo "Running REIMPLEMENTATION feedback loop..."
    echo "=============================================="
    rm -rf data/reimplementation/feedback_loop 2>/dev/null || true
    time uv run python src/scripts/train_reimpl.py configs/reimplementation/feedback_loop.yaml
    echo ""
}

case "$VERSION" in
    original)
        run_original
        ;;
    reimpl)
        run_reimpl
        ;;
    both|*)
        run_original
        run_reimpl
        echo "=============================================="
        echo "Comparison complete!"
        echo "Original results:        data/original/feedback_loop/"
        echo "Reimplementation results: data/reimplementation/feedback_loop/"
        echo "=============================================="
        ;;
esac
