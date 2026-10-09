# Teacher–student compression and deployment efficiency on CMU-MOSEI

| Model | Params ↓ | Peak GPU Mem. (GiB) ↓ | Model Latency p50 / p95 (ms) ↓ | E2E Throughput (utt/s) ↑ |
|---|---:|---:|---:|---:|
| Qwen3-Omni Teacher + Probe | 31.720B | 67.526 | 435 / 2104 | 0.735 |
| Full KD Student | 819.309M | 2.091 | **94** / 127 | **1.630** |
| **ICD Student** | 819.309M | 2.091 | 95 / **126** | **1.630** |

**Caption.** Teacher–student compression and deployment efficiency on CMU-MOSEI. All models were measured sequentially on the same physical NVIDIA RTX 6000D GPU (GPU 1) of the same machine, using batch size 1. The teacher and Probe use BF16; students use BF16 autocast inference. Statistics are computed over one timed pass through 90 distinct single-window official-validation TAV utterances (30 each from <5 s, 5–15 s, and 15–30 s), after three held-out warm-up utterances (one per bucket). CUDA was synchronized immediately before and after each timed model forward. Model latency covers preprocessed host tensors, host-to-device transfer, and model forward through the scalar regression output, excluding media I/O/decoding and preprocessing. E2E throughput additionally includes audio loading, video decoding/frame sampling, tokenization, multimodal preprocessing, transfer, and model forward; it excludes DataLoader worker queuing and multi-sample batching. Peak GPU memory is peak allocated memory measured by `torch.cuda.max_memory_allocated()` after warm-up, including model weights and batch-1 inference tensors. Full KD and ICD use seed13 checkpoints selected by validation MAE. ICD denotes Uniform Interaction Distillation.

**Appendix note.** The earlier 5.425 GiB student figure used batch size 8 in the full official-test evaluator, whereas 2.091 GiB here is batch-1 peak allocated memory. The difference is therefore expected and the two figures should not be compared directly.
