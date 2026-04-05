#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-t5gemma2-ref"

echo "=== Creating T5Gemma2 reference virtual environment ==="
echo "  Location: ${VENV_DIR}"

if [ -d "${VENV_DIR}" ]; then
    echo "  Removing existing venv..."
    rm -rf "${VENV_DIR}"
fi

python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel -q

echo "  Installing T5Gemma2 reference dependencies..."
pip install \
    "transformers>=5.0.0" \
    "torch>=2.0" \
    "safetensors" \
    "sentencepiece" \
    "pillow" \
    "huggingface-hub" \
    "numpy" \
    2>&1 | tail -10

echo ""
echo "=== T5Gemma2 reference venv ready ==="
echo "  Activate with: source ${VENV_DIR}/bin/activate"
echo "  Key packages:"
pip list 2>/dev/null | grep -iE "transformers|torch|safetensors" || true
echo ""
echo "  Collect reference logits:"
echo "    python models/t5gemma2/parity_test.py --collect"
