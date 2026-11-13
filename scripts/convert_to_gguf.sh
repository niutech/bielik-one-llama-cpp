#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 HF_MODEL_DIR OUTPUT_BF16.gguf [LLAMA_CPP_DIR]" >&2
  exit 2
fi
HF_MODEL_DIR="$1"
OUTPUT="$2"
LLAMA_CPP_DIR="${3:-./llama.cpp}"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_MODEL_DIR" \
  --outfile "$OUTPUT" --outtype bf16

echo "Converted to $OUTPUT"
echo "IMPORTANT: run 'bielik-one-llama-cpp check $OUTPUT' before serving."
