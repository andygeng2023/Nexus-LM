# NEXUS-LM architecture

## Current hybrid block

Each block applies the following residual components:

1. **Selective diagonal state-space path.** A token-dependent projection predicts a positive discretisation step and input/output selectors. Each channel has an affine transition s_t = a_t * s_(t-1) + b_t. Associative transition composition permits an inclusive parallel prefix scan with logarithmic tensor stages.
2. **Causal multi-head attention** in alternating blocks. This remains dense and quadratic in sequence length.
3. **Sparse top-k SwiGLU experts.** Tokens select a small subset of experts; selected routes are sorted and grouped so expert dispatch avoids rescanning all routes separately for every expert.

The embedding and language-model output weights are tied.

## What the optimisation does and does not do

The selective scan is vectorised as a Hillis-Steele parallel prefix scan using LibTorch tensor operators. It is not yet a hand-written fused CUDA/Metal kernel: each scan level creates backend operations and intermediate tensors. It reduces serial dependency depth from O(T) to O(log T), but can still be slower for short contexts because of launch and memory traffic.

Expert routes are top-k selected, sorted by expert ID and processed as contiguous expert ranges. There is still per-expert dispatch and an index-add operation. A fused grouped GEMM/kernel would be a later optimisation.

BPE training uses linked token positions and a lazy heap of pair counts; encoding uses a priority queue of merge ranks. Both avoid repeating full-sequence scans for every merge. It remains a simple byte-level BPE implementation, not a production tokenizer.

## Baseline and measurements

nexus_bench compares a hybrid model with a dense causal Transformer using a widened SwiGLU reference. It prints stored parameter count and an estimate of active parameter count per token. MoE's total stored capacity differs from a dense baseline with a similar active count.

Benchmark data is synthetic and measures a training step. It does not measure language-model quality. Compare with identical corpus, tokens seen, parameter budget, sequence length, hardware, optimiser, seed protocol and held-out evaluation before drawing conclusions.

## Research limitations

- The diagonal SSM is an experimental selective recurrence, not a complete Mamba implementation.
- Causal attention is still quadratic.
- MoE dispatch is grouped but not fused.
- No pretrained checkpoint is distributed.
- WikiText-103 is a language-modelling benchmark corpus rather than an instruction-following dataset.
