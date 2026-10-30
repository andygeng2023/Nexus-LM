# NEXUS hybrid architecture v0.2

## Block layout

Each NEXUS block applies residual sublayers in this order:

1. **Selective diagonal state-space layer.** A per-token projection predicts a positive step size (delta) and input/output selectors (B, C). Each channel maintains a recurrent state:

   decay_t = exp(-exp(A_log) * softplus(delta_t))

   state_t = decay_t * state_(t-1) + (1 - decay_t) * tanh(B_t) * x_t

   y_t = tanh(C_t) * state_t + D * x_t

   The recurrence is strictly causal. It has linear sequence work but is currently implemented as a sequential C++ loop over tensor operations; it is a research implementation, not a fused high-performance scan kernel.
2. **Causal multi-head attention** in alternating blocks. The current implementation is dense and has quadratic sequence complexity.
3. **Top-k routed SwiGLU experts.** The router chooses top_k experts per token and combines their outputs with a softmax over selected routes.

Token embeddings and output projection weights are tied.

## Reference baseline

TransformerBaseline is a dense causal Transformer with SwiGLU layers. nexus_bench compares training-step latency and tokens/second on the same generated batch and reports stored parameter counts. The baseline uses a widened feed-forward layer (default multiplier 9) to approximately match the hybrid model's per-token active parameter budget. This is only an approximation: NEXUS stores all experts, while only top-k experts are active for an individual token. The benchmark prints both stored and estimated active parameters to make that distinction visible.

## Byte-level BPE

The tokenizer begins with all 256 byte values, then repeatedly merges the most frequent adjacent pair over the supplied training corpus. Merge order is persisted in a compact text file. Encoding applies those merges in rank order. Decoding concatenates token byte strings, so arbitrary UTF-8 input is preserved byte-for-byte. There are no special tokens yet, and merge training is a simple corpus-wide algorithm with repeated scans; it is intended as a functional initial tokenizer rather than a production-scale tokenizer trainer.

## Validation and reproducibility

The CLI fits BPE merges on the leading 90% of the corpus, then the trainer tokenizes the full file and reserves the final 10% of token IDs for held-out loss when the validation segment is long enough. This byte-boundary/token-boundary split is approximate; for rigorous experiments, create explicit train/validation files and fit the tokenizer only on the training file. The random seed is applied before model construction.

## Known research and engineering limits

- The SSM is a selective diagonal recurrence, not a full reproduction of Mamba or a claim of a new state-of-the-art architecture.
- The SSM recurrence is not yet fused/parallelised; sequential dispatch can be slower than dense attention at short contexts.
- Attention remains quadratic and is used in every second hybrid block.
- Top-k expert dispatch currently uses tensor indexing and a per-expert loop; dispatch overhead needs profiling and optimisation.
- Benchmark results are hardware- and build-dependent. A random-batch microbenchmark does not measure language quality.
- The benchmark's widened Transformer baseline approximates active parameter count, not total stored capacity.
- No trained checkpoint is committed. Training from scratch requires a large, diverse, legally usable corpus and substantially more compute than the included smoke-test data.
