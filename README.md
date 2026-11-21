# jitNN benchmarking

Benchmarks of [jitNN](https://github.com/beestream-ai/jitNN) against PyTorch, ONNX Runtime,
TensorRT and DJL, the tools that run them, and their results. The numbers and how they were taken
are in [PERFORMANCE.md](PERFORMANCE.md).

## Latest results

NVIDIA A100 80GB PCIe, 2026-11-19: jitNN (`main`, the library code of release 0.1.0-alpha) against
PyTorch 2.14 (`torch.compile`) and ONNX Runtime 1.30 (CUDA provider). Milliseconds from host input
to host output, median of three interleaved rounds; lower is better, fastest in bold. The last
column is jitNN's time over the faster of the other two: below 1 means jitNN is faster.

**Inference**

| Model | Batch | Precision | jitNN | PyTorch | ONNX Runtime | jitNN vs best |
|---|---|---|---|---|---|---|
| ResNet-50 | 1 | FP32 | **2.75** | 4.42 | 3.34 | 0.82x |
| ResNet-50 | 32 | FP32 | 24.34 | 20.32 | **20.18** | 1.21x |
| ResNet-50 | 128 | FP32 | 84.17 | **72.24** | 78.89 | 1.17x |
| ResNet-50 | 1 | TF32 | 2.17 | 1.72 | **1.19** | 1.83x |
| ResNet-50 | 32 | TF32 | 10.89 | **8.34** | 9.72 | 1.31x |
| ResNet-50 | 128 | TF32 | 31.94 | **30.41** | 35.77 | 1.05x |
| ResNet-50 | 1 | FP16 | 1.20 | 1.23 | **0.94** | 1.27x |
| ResNet-50 | 32 | FP16 | 6.03 | **4.41** | 5.53 | 1.37x |
| ResNet-50 | 128 | FP16 | 17.44 | **14.48** | 20.62 | 1.20x |
| BERT-base | 1 x 128 | FP32 | 3.61 | **2.92** | 3.41 | 1.23x |
| BERT-base | 32 x 128 | FP32 | 49.32 | **46.28** | 50.22 | 1.07x |
| BERT-base | 8 x 512 | FP32 | 55.72 | **49.43** | 56.41 | 1.13x |
| BERT-base | 1 x 128 | TF32 | 2.19 | **1.46** | 2.45 | 1.50x |
| BERT-base | 32 x 128 | TF32 | 14.40 | **11.01** | 14.24 | 1.31x |
| BERT-base | 8 x 512 | TF32 | 20.60 | **14.79** | 16.66 | 1.39x |
| BERT-base | 1 x 128 | FP16 | **1.15** | 1.17 | 2.31 | 0.98x |
| BERT-base | 32 x 128 | FP16 | **5.82** | 5.84 | 8.65 | 1.00x |
| BERT-base | 8 x 512 | FP16 | 6.24 | **6.12** | 10.15 | 1.02x |

**Training**, milliseconds per step (forward, backward, optimiser), against PyTorch

| Workload | Precision | jitNN | PyTorch | jitNN vs PyTorch |
|---|---|---|---|---|
| ResNet-50, CIFAR-10, batch 32, SGD | FP32 | 31.69 | **31.20** | 1.02x |
| ResNet-50, CIFAR-10, batch 32, SGD | TF32 | 17.36 | **11.28** | 1.54x |
| BERT-base, batch 8 x 128, Adam | FP32 | 41.64 | **38.27** | 1.09x |
| BERT-base, batch 8 x 128, Adam | TF32 | 15.87 | **12.07** | 1.31x |
| GPT-2 (124M), batch 8 x 256, Adam | FP32 | 107.07 | **99.45** | 1.08x |
| GPT-2 (124M), batch 8 x 256, Adam | TF32 | 36.91 | **26.99** | 1.37x |

FP32 is strict (TF32 off everywhere); jitNN's FP32 rows are its faster of own kernels and
`-Djitnn.native=auto`, TF32 rows add `-Djitnn.native.tf32=true`, FP16 rows run the whole model in
half precision. ONNX Runtime runs BERT without CUDA graphs on this host (its build fails with them).
Setup, first-call times, peak memory and the raw data:
[PERFORMANCE.md](PERFORMANCE.md#nvidia-a100-80gb-pcie-2026-11-19) and
[results/a100/cuda-2026-11-19](results/a100/cuda-2026-11-19).

## Layout

| Path | Content |
|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | Inference, training, per-kernel, fusion, autotuning, multi-GPU and compile-time results on CUDA, Metal and OpenCL |
| `jitnn-bench` | Maven module: the reference workloads captured as traces (`Workloads`), every distinct operator call timed on a device (`KernelBench`), whole training and inference steps (`ModelProfile`), GPU clock monitoring |
| `scripts` | Drivers and the framework side: `inference.py` (PyTorch and ONNX Runtime inference), `gsweep.py` and `gtrain.py` (fastest PyTorch, ONNX Runtime and TensorRT variants), `gpu_final.sh` (the GPU head-to-head) and `gpu_summary.py`, `train_bench.py` (PyTorch training steps), `fp64check.py` and `jerr.py` (output error), `run_kernels.sh` and `run_models.sh` (`jitnn-bench`) with `compare.py`, `rank.py`, `sweep.py` and `nsys_kernels.py` to read their results, `kernels.py` (the same calls through PyTorch), `tune.py` (jitNN's GEMM tuning tables), `run_mac.sh` (Apple silicon), `djl`, `djl-examples` and `trace.py` (DJL), `ceilings` (device peaks) |
| `results` | Raw results per device and date |

## Building

The benchmarks build against an installed jitNN and a TornadoVM SDK:

```bash
source <TornadoVM>/setvars.sh                       # TORNADOVM_HOME
(cd <jitNN> && ./mvnw install -DskipTests)          # jitNN into the local Maven repository
./mvnw install -DskipTests                          # this repository
```

Scripts run from this directory and take the jitNN checkout from `JITNN_HOME` when they need it:
the Python side builds its PyTorch models with jitNN's `scripts/pytorch` (`inference.py`,
`train_bench.py`, and those built on `inference.py`), `gpu_final.sh` and `run_mac.sh` also run
jitNN's own inference benchmark (`InferenceBenchAccelTest`), and `tune.py` writes jitNN's tuning
tables:

```bash
JITNN_HOME=<jitNN> scripts/gpu_final.sh <out dir>
scripts/run_kernels.sh 0:0 <out dir>
```

Each script describes its arguments and environment at the top, below the license header.

## Properties

System properties of `jitnn-bench`'s tests (`KernelBenchAccelTest`, `ModelProfileAccelTest`, and the
probes), passed through `-Djitnn.test.extra=...`; `scripts/run_kernels.sh` and `run_models.sh` set them.
jitNN's own options and test properties are in its
[CONFIGURATION.md](https://github.com/beestream-ai/jitNN/blob/main/docs/CONFIGURATION.md).

| Property | Default | Meaning |
|---|---|---|
| `jitnn.kbench` | `false` | run `KernelBenchAccelTest`: every distinct operator call of the benchmark workloads, timed on the device |
| `jitnn.models` | `false` | run `ModelProfileAccelTest`: whole training and inference steps of the workloads |
| `jitnn.kbench.out` | `kernels.jsonl`, `models.jsonl` | result file, appended |
| `jitnn.kbench.label` | empty | free text recorded with every result line, for example the commits |
| `jitnn.kbench.workloads` | empty | comma-separated workload filter (for example `mlp_train_b256`); empty: all |
| `jitnn.kbench.families` | empty | comma-separated operator families (for example `gemm,convolution`); empty: all |
| `jitnn.kbench.from` | `0` | skip the first n calls, to resume a run |
| `jitnn.kbench.only` | empty | comma-separated call indices to run (1-based, in capture order) |
| `jitnn.kbench.warmup`, `jitnn.kbench.calls` | `20`, `100` | warm-up and minimum timed calls per operator call (timed for at least 0.3 s) |
| `jitnn.kbench.steps`, `jitnn.kbench.modelWarmup` | `30`, `10` | timed and warm-up steps per workload of `ModelProfileAccelTest` (with `jitnn.cpu.stats`, it also prints the CPU backend's operators by time) |
| `jitnn.kbench.gpu` | `0` | `nvidia-smi` index of the GPU whose clocks are sampled |
| `jitnn.kbench.minMhz` | `1350` | a timed window is retried when the mean SM clock is below this or the GPU was power or thermal capped |
| `jitnn.kbench.bandwidth`, `jitnn.kbench.fp32`, `jitnn.kbench.half` | `1721e9`, `19.0e12`, `245e12` | peak bytes/s and FLOP/s that the efficiency columns are relative to |
| `jitnn.lstmprobe` | `false` | run `LstmProbeAccelTest`: LSTM training-step times at the shapes of DJL's MNIST LSTM example and two wider ones |
| `jitnn.tileprobe` | `false` | run the CUDA Tile API probes `TileGemmProbeAccelTest` and `TileAttentionProbeAccelTest` |

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright (c) 2026 Beestream Ltd. Third-party material is listed in [NOTICE](NOTICE).
