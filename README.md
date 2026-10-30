# NEXUS-LM

An experimental, local-first language-model research framework written in C++20. NEXUS starts from random initialisation and explores a hybrid of causal state processing, causal attention, and sparse expert routing.

> **Status: bootstrap / research prototype.** This is not a trained assistant and does not include pretrained weights. The initial model is intentionally small so that architecture and training correctness can be tested before scaling.

## Design goals

- Native C++20 core and command-line interface.
- Random initialisation; no pretrained model weights.
- CPU fallback and optional CUDA through the installed LibTorch build.
- Local text-file training and local generation.
- Reproducible seeds, checkpoints, and tests.
- Experimental hybrid blocks with top-k expert dispatch.

## Requirements

- CMake 3.20+
- A C++20 compiler (Clang, GCC, or MSVC)
- LibTorch matching your compiler and platform

Download the LibTorch C++ distribution from the official PyTorch installation selector: https://pytorch.org/get-started/locally/ . Set `CMAKE_PREFIX_PATH` to the extracted LibTorch directory.

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
```

## Train from random initialisation

The bootstrap tokenizer maps each UTF-8 byte to one token. Provide a plain-text corpus larger than the configured sequence length.

```bash
./build/nexus train examples/tiny_corpus.txt nexus-tiny.pt 100
```

The executable uses a deliberately small default configuration in `src/main.cpp`. The generated checkpoint contains weights learned from random initialisation on the supplied text; the tiny example corpus is only sufficient for pipeline smoke testing and is not sufficient to create a useful language model.

## Generate

```bash
./build/nexus generate nexus-tiny.pt "Language models" 80
```

Generation currently uses the same compiled-in model configuration as training. Keep those settings identical when loading a checkpoint.

## Important limitations

- The byte tokenizer is a bootstrap implementation; it is inefficient for natural language compared with a trained subword tokenizer.
- The state path is a causal depthwise convolutional approximation, not a novel proven state-space architecture.
- The attention implementation is dense and quadratic in sequence length.
- The model is not instruction-tuned and should not be expected to behave like a modern assistant.
- CUDA support depends on the installed LibTorch build. CPU is the fallback. Apple Metal is not yet implemented or verified.
- This repository does not ship training data. Use datasets you have the rights to use and follow their licences and privacy requirements.

## Research protocol

When comparing architectural variants, hold the training corpus, tokenizer, parameter budget, training-token budget, optimiser, and evaluation set constant. Record throughput, peak memory, wall-clock time, validation loss, and task metrics. Keep a dense Transformer baseline so that claimed efficiency improvements can be tested.

## Roadmap

1. Establish compile/test correctness and baseline measurements.
2. Add a learned BPE tokenizer and robust corpus pipeline.
3. Add configuration-file loading and checkpoint metadata.
4. Implement and benchmark a genuine selective state-space layer.
5. Investigate sparse/block attention and adaptive computation.
6. Add a validated Metal backend and portable quantised inference.
7. Add instruction tuning and rigorous evaluation only after language pretraining is adequate.

## Licence

MIT. Dataset and model-training data licences remain the responsibility of the dataset provider and user.
