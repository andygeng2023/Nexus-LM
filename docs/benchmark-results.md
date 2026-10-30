# Initial local smoke benchmark

These figures are an initial functional smoke test, not a reliable performance claim.

Environment: CPU-only LibTorch 2.10.0, GCC 14.2.0, CMake 3.31.6. Batch size 2, two warm-up train steps. Width 128, four blocks. NEXUS uses 4 experts with top-2 routing. The dense Transformer uses a SwiGLU hidden multiplier of 9 to approximate the NEXUS active per-token parameter budget.

| Context | Model | Stored parameters | Estimated active parameters/token | Step time | Throughput |
|---:|---|---:|---:|---:|---:|
| 16 | NEXUS hybrid | 3,697,936 | 2,115,856 | 76.37 ms | 419 tokens/s |
| 16 | Dense Transformer | 2,177,280 | 2,177,280 | 12.32 ms | 2,597 tokens/s |
| 64 | NEXUS hybrid | 3,697,936 | 2,115,856 | 158.17 ms | 809 tokens/s |
| 64 | Dense Transformer | 2,177,280 | 2,177,280 | 39.29 ms | 3,258 tokens/s |

The NEXUS reference implementation is slower in this smoke test. The likely engineering causes are the unfused sequential SSM recurrence and the current per-expert tensor-indexing dispatch. Dense matrix operations in LibTorch benefit from better optimised kernels. The SSM therefore establishes a selective, causal baseline, but **does not yet meet the efficiency objective**. The next optimisation work should fuse/parallelise the state scan and replace index-based expert routing with grouped dispatch kernels.

The benchmark uses random synthetic tokens and only a few measured iterations. It does not compare language quality, generalise to other devices or establish a stable ranking. Re-run with many iterations, fixed real corpora, peak-memory recording and held-out evaluation before making performance claims.
