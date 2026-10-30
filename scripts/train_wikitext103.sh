#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
DATA_DIR="${DATA_DIR:-$ROOT/data/wikitext-103}"
MODEL_DIR="${MODEL_DIR:-$ROOT/checkpoints}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"
SAMPLE_BYTES="${SAMPLE_BYTES:-2000000}"
STEPS="${STEPS:-50000}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCHES="${EVAL_BATCHES:-256}"

python3 "$ROOT/scripts/prepare_wikitext103.py" --data-dir "$DATA_DIR"

cmake_args=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  cmake_args+=(-DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH")
elif python3 -c 'import torch' >/dev/null 2>&1; then
  cmake_args+=(-DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')")
fi

cmake -S "$ROOT" -B "$BUILD_DIR" "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-2}"
mkdir -p "$MODEL_DIR"

"$BUILD_DIR/nexus" tokenizer-train \
  "$DATA_DIR/train.txt" "$MODEL_DIR/wikitext103.bpe" "$VOCAB_SIZE" "$SAMPLE_BYTES"

"$BUILD_DIR/nexus" train \
  "$DATA_DIR/train.txt" "$MODEL_DIR/nexus-wikitext103.pt" \
  "$MODEL_DIR/wikitext103.bpe" "$STEPS" "$SEQUENCE_LENGTH" "$BATCH_SIZE" \
  "$DATA_DIR/validation.txt"

"$BUILD_DIR/nexus" evaluate \
  "$MODEL_DIR/nexus-wikitext103.pt" "$MODEL_DIR/wikitext103.bpe" \
  "$DATA_DIR/test.txt" "$EVAL_BATCHES"

printf '\nTraining finished.\nCheckpoint: %s\nTokenizer: %s\n' \
  "$MODEL_DIR/nexus-wikitext103.pt" "$MODEL_DIR/wikitext103.bpe"
