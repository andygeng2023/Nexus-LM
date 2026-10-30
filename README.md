# NEXUS-LM

NEXUS-LM is a local-first C++20 language-model research framework. It trains from random initialisation and experiments with a selective recurrent state-space path, alternating causal attention, and sparse top-k expert routing.

> **Status: research prototype.** It is not a pretrained assistant and has not been shown to outperform a Transformer. The WikiText-103 workflow below prepares a real, substantially larger language-modelling corpus; training must be run on your own machine or a suitable GPU host.

## Features

- Native C++20 executable and LibTorch tensor/autograd backend.
- Reversible byte-level BPE tokenizer with deterministic distributed corpus sampling.
- Input-selective diagonal state-space layer using an associative parallel prefix scan.
- Alternating causal multi-head attention and sparse top-k SwiGLU experts.
- Dense causal Transformer/SwiGLU reference implementation.
- AdamW training, gradient clipping, validation loss, checkpoints and held-out test perplexity.
- Tests for tokenizer round-tripping, causality, gradients and baseline output shape.

## Requirements

- CMake 3.20+
- C++20 compiler (Clang, GCC or MSVC)
- LibTorch matching your platform and compiler
- Python 3.9+ for the dataset preparation script

Download LibTorch from the official PyTorch selector: https://pytorch.org/get-started/locally/.

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```

If LibTorch was installed through Python's `torch` package, set its CMake path automatically:

```bash
export CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

## Train on the larger WikiText-103 corpus

WikiText-103 is an English Wikipedia language-modelling dataset containing over 100 million training tokens. It is much larger than the tiny test fixture, but it is encyclopaedic prose rather than an instruction-following dataset. See [data sources and licence notes](docs/data-sources.md).

On macOS/Linux, from the repository root:

```bash
python3 scripts/prepare_wikitext103.py
```

You can train locally as shown below, or use **Actions → Train NEXUS on WikiText-103 → Run workflow** to train on a GitHub-hosted CPU runner. The manual workflow uploads the native checkpoint, tokenizer, browser-exported model and training/test metrics as a downloadable artifact; it does not commit model files to Git. Start with 2,000 steps. Then run a moderate first experiment locally:

```bash
export CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
VOCAB_SIZE=1024 SAMPLE_BYTES=200000 STEPS=2000 SEQUENCE_LENGTH=256 BATCH_SIZE=4 \
  bash scripts/train_wikitext103.sh
```

The script downloads and verifies the archive, creates separate train/validation/test files, trains the tokenizer on a representative sample from the training split only, trains on the training split, monitors validation loss, and evaluates on the held-out test split. It writes the model and tokenizer under `checkpoints/`. The dataset and checkpoints are intentionally not committed to Git.

For a longer run, increase `STEPS` after confirming that the loss decreases and that your machine has enough memory. The defaults in the script are configurable; training time depends heavily on the CPU/GPU and LibTorch build. CUDA is selected automatically when available. Apple Metal/MPS is not implemented or verified, so a Mac may use CPU unless its LibTorch build exposes CUDA (normally not on Apple Silicon).

### CLI commands

```bash
./build/nexus tokenizer-train data/wikitext-103/train.txt checkpoints/tokenizer.bpe 1024 200000
./build/nexus train data/wikitext-103/train.txt checkpoints/model.pt checkpoints/tokenizer.bpe 2000 256 4 data/wikitext-103/validation.txt
./build/nexus evaluate checkpoints/model.pt checkpoints/tokenizer.bpe data/wikitext-103/test.txt 256
./build/nexus generate checkpoints/model.pt checkpoints/tokenizer.bpe "The history of language modelling" 80
```

Generation prints a continuation after the prompt. It is not expected to be coherent after a short training run. Use the exact tokenizer and matching architecture settings used during training.

## Use it locally

After training, run:

```bash
./build/nexus generate checkpoints/nexus-wikitext103.pt checkpoints/wikitext103.bpe "Once upon a time" 100
```

On Windows, use the corresponding `nexus.exe` path and PowerShell syntax.

## Offline PWA for iPad and desktop browsers

The web/ directory contains an installable, offline-first PWA. It runs inference in the browser on the device; it does not call a cloud inference API. The current browser backend is a portable JavaScript CPU implementation of the NEXUS model (SSM, causal attention and top-k experts). It does not yet use WebGPU, so larger models or long generations may be slow on an iPad. Browser memory and local-storage quotas also vary by device.

### Cloud train, download, install

1. Merge PR #1 after its CI checks pass. This puts the exporter and Pages workflow on the default branch, where GitHub Actions can run them.
2. Enable GitHub Pages for this repository with Settings → Pages → Build and deployment → GitHub Actions.
3. Run Actions → Train NEXUS on WikiText-103 → Run workflow. The workflow trains the model and uploads the native checkpoint, browser-exported files and metrics as an artifact.
4. Download and extract the workflow artifact. Keep web-model/model.json, web-model/model.bin and web-model/tokenizer.bpe together.
5. Open the published HTTPS Pages URL in Safari on iPad. Use Share → Add to Home Screen to install the PWA.
6. Open NEXUS, choose all three model files together, and import them. The app stores them in IndexedDB on that device. Once the app and model have both been cached, inference can run offline.
7. Enter a prompt and select Generate offline. The prompt and generated text remain in the browser; this PWA has no inference server.

To build the browser export locally after a native training run:

```bash
./build/nexus_export_web checkpoints/nexus-wikitext103.pt checkpoints/wikitext103.bpe web-model
```

The native LibTorch .pt checkpoint cannot be loaded directly by Safari. The exporter writes a browser manifest and contiguous float32 tensor file, plus the matching tokenizer. Keep the exported files from the same training run together.

### Browser runtime limits

- The PWA requires HTTPS (GitHub Pages provides this) or localhost for service workers and offline caching.
- The initial runtime is CPU-only JavaScript. WebGPU acceleration and model quantisation are not implemented yet.
- iPadOS Safari capabilities and available memory depend on the iPad model and OS version. This implementation does not claim WebGPU support or guarantee that every model size will fit.
- The browser stores model data in IndexedDB. Clearing website data or removing the app's website storage deletes the local model.
- The PWA is an inference client, not a training environment. Training remains in the cloud workflow.
- WikiText-103 and a short first run do not produce a capable instruction-following assistant. The project remains a research prototype, not a validated general-purpose assistant.

## Benchmarks and limitations

See [initial local benchmark results](docs/benchmark-results.md). Those measurements used tiny synthetic batches and are not a quality comparison. The original CPU microbenchmark showed this prototype slower than the dense Transformer baseline. The prefix scan and grouped expert dispatch reduce obvious overhead, but must be benchmarked again on the target hardware.

- The SSM is a research implementation, not a fused custom CUDA kernel.
- Attention remains quadratic in context length.
- Expert routing still has dispatch overhead.
- Tokenizer training is intended for a sampled corpus; very large vocabulary/sample combinations can take a long time.
- Training currently loads token IDs into host memory and checkpoints model weights only; optimiser-state resume and robust checkpoint metadata are not implemented.
- WikiText-103 alone does not teach instruction following, factual reliability or safe assistant behaviour.
- No pretrained weights are included.

## Research discipline

Compare variants using the same tokenizer, data split, token budget, optimiser, evaluation suite and hardware. Record compiler, LibTorch version, training wall time, peak memory, tokens/second, validation loss and held-out perplexity. Do not infer intelligence from training loss or synthetic benchmark throughput alone.

## Licence

The code is MIT. Check upstream dataset licences, provenance, privacy constraints and permitted uses before using or redistributing training data, checkpoints or model weights.
