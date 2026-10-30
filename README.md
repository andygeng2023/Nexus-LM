# NEXUS-LM

NEXUS-LM is a local-first C++20 language-model research framework. It trains from random initialisation and experiments with a selective recurrent state-space path, alternating causal attention, and sparse top-k expert routing.

> **Status: research prototype.** The repository contains a trainable implementation and tests, not pretrained weights or a ready-made assistant. The architecture is experimental and has not been shown to outperform a Transformer.

## Features

- Native C++20 executable and LibTorch tensor/autograd backend.
- Reversible byte-level BPE tokenizer trained on a local text corpus.
- Input-selective diagonal recurrent state-space layer with input-dependent step size and B/C selectors.
- Causal multi-head attention in alternating hybrid blocks.
- Sparse top-k SwiGLU expert routing.
- Dense causal Transformer/SwiGLU reference implementation.
- Training loop with AdamW, gradient clipping, held-out validation loss and checkpoint saving.
- Tests for tokenizer round-tripping, SSM causality, model causality, gradients and baseline output shape.
- Microbenchmark for training-step time, throughput and parameter counts.

## Requirements

- CMake 3.20+
- C++20 compiler (Clang, GCC or MSVC)
- LibTorch distribution matching your platform and compiler

Download LibTorch from the official PyTorch selector: https://pytorch.org/get-started/locally/ .

## Build

\`\`\`bash
cmake -S . -B build \\
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \\
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
\`\`\`

## Prepare a tokenizer

Train the tokenizer on the same text corpus you intend to use for language-model training:

\`\`\`bash
./build/nexus tokenizer-train data/corpus.txt data/tokenizer.bpe 2048
\`\`\`

The last argument is the target vocabulary size. The tokenizer learns merge rules only from the leading 90% of the supplied text; the final 10% is kept unseen by tokenizer fitting. The actual vocabulary may be smaller if no pair occurs at least twice. The included implementation is a simple byte-level BPE baseline; train it on a substantial representative corpus for real experiments.

## Train from random initialisation

\`\`\`bash
./build/nexus train data/corpus.txt checkpoints/nexus.pt data/tokenizer.bpe 10000 128 8
\`\`\`

Arguments after the tokenizer are steps, sequence length and batch size. The model uses width 128, four hybrid blocks, four experts and top-2 routing by default. The initialization seed defaults to 42 and is applied before model construction. The code automatically selects CUDA when the installed LibTorch build supports it; otherwise it uses CPU.

Training reserves the last 10% of tokenised text for held-out validation when there are enough tokens. Checkpoints contain model weights. Optimiser state/resume training and full checkpoint metadata are not implemented yet; use the same architecture settings and tokenizer file when loading a checkpoint.

## Generate locally

\`\`\`bash
./build/nexus generate checkpoints/nexus.pt data/tokenizer.bpe "Your prompt" 100
\`\`\`

The tokenizer file must be the exact same one used for training.

## Benchmark against a dense Transformer

\`\`\`bash
./build/nexus_bench 128 20
\`\`\`

The benchmark reports stored parameters, estimated active parameters per token, training-step time, throughput and final random-batch loss. It uses a widened dense Transformer reference to approximately match NEXUS's active parameter budget. It does not measure held-out language quality and does not imply universal speed or quality gains. Results should be recorded separately on each target hardware backend.

See [initial local benchmark results](docs/benchmark-results.md).

## Current limitations

- The selective SSM is a simple diagonal recurrent layer. The time loop is not fused or parallelised and may be slower than attention for short sequences.
- Attention is dense and quadratic in context length.
- Expert dispatch has indexing overhead and requires profiling.
- The BPE trainer rescans the corpus for every merge and has no special tokens or byte-fallback metadata beyond its fixed byte alphabet.
- Apple Metal is not implemented or verified. CUDA depends on the installed LibTorch build.
- No pretrained weights or training data are supplied. The included example corpus is for software smoke tests only.

## Research discipline

Compare variants with the same tokenizer, data split, token budget, evaluation suite, optimiser and model-quality target. Record the hardware, compiler, LibTorch version, wall-clock training time, peak memory, tokens/second, validation loss and held-out task results. Do not infer intelligence from training loss or synthetic benchmark throughput alone.

## Licence

MIT. Check the licences, provenance, privacy constraints and permitted uses of any training data before use.
