# NEXUS architecture (bootstrap)

The randomly initialised autoregressive model currently combines three paths in each block:

1. A causal depthwise convolutional state path with an input-dependent gate.
2. Causal multi-head self-attention in alternating blocks.
3. Top-k routing to feed-forward experts.

The embedding and output weights are tied.

## Research caveats

The convolutional path is a finite-window learned state approximation, not a proven selective SSM. Attention is dense and quadratic in sequence length. The byte tokenizer is reversible but inefficient. The checkpoint format is LibTorch serialization. CUDA depends on the installed LibTorch build. Apple Metal is not implemented or verified. These are explicit limitations, not claimed completed features.
