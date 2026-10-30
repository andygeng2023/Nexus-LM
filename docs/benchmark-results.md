# Local CPU microbenchmark

Environment: CPU-only PyTorch/LibTorch 2.10.0, GCC 14.2.0. Batch 2, sequence length 64, two warm-up steps and 20 measured train steps. NEXUS width 128, four hybrid blocks, four experts, top-2 routing. The Transformer reference uses a widened SwiGLU hidden multiplier of nine to approximate NEXUS's active per-token parameter budget.

| Metric | NEXUS hybrid | Dense Transformer x9 |
|---|---:|---:|
| Stored parameters | 3,697,936 | 2,177,280 |
| Estimated active parameters/token | 2,115,856 | 2,177,280 |
| Average train-step time | 73.05 ms | 32.70 ms |
| Training tokens/second | 1,752 | 3,915 |

NEXUS is still slower on this CPU microbenchmark. The parallel scan and grouped expert dispatch reduce avoidable serial/indexing work, but LibTorch's dense matrix kernels remain highly optimised. The scan is not yet a fused kernel and the hybrid architecture contains both a recurrent path and attention. Further speed work requires profiling, fused scan kernels and grouped expert GEMMs.

The two models are initialised independently; their final synthetic-batch losses are not a fair quality comparison. These results are a single local run, not a universal hardware claim. Re-run with repeated seeds, real held-out data, longer measurements and memory tracking before relying on them.
