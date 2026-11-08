# jitNN benchmarking

Benchmarks of [jitNN](https://github.com/beestream-ai/jitNN) against PyTorch, ONNX Runtime,
TensorRT and DJL, the tools that run them, and their results. The numbers and how they were taken
are in [PERFORMANCE.md](PERFORMANCE.md).

## Layout

| Path | Content |
|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | Inference, training, per-kernel, fusion, autotuning, multi-GPU and compile-time results on CUDA, Metal and OpenCL |
| `jitnn-bench` | Maven module: the reference workloads captured as traces (`Workloads`), every distinct operator call timed on a device (`KernelBench`), whole training and inference steps (`ModelProfile`), GPU clock monitoring |
| `scripts` | Drivers and the framework side: `t1.py` (PyTorch and ONNX Runtime inference), `gsweep.py` and `gtrain.py` (fastest PyTorch, ONNX Runtime and TensorRT variants), `gpu_final.sh` (the GPU head-to-head), `run_kernels.sh` and `run_models.sh` (`jitnn-bench`), `kernels.py` (the same calls through PyTorch), `tune.py` (jitNN's GEMM tuning tables), `run_mac.sh` (Apple silicon), `djl` and `djl-examples` (DJL), `ceilings` (device peaks) |
| `results` | Raw results per device and date |

## Building

The benchmarks build against an installed jitNN and a TornadoVM SDK:

```bash
source <TornadoVM>/setvars.sh                       # TORNADOVM_HOME
(cd <jitNN> && ./mvnw install -DskipTests)          # jitNN into the local Maven repository
./mvnw install -DskipTests                          # this repository
```

Scripts run from this directory. Those that also run jitNN's own inference benchmark
(`InferenceBenchAccelTest`: `gpu_final.sh`, `run_mac.sh`) and `tune.py`, which writes jitNN's
tuning tables, take the jitNN checkout from `JITNN_HOME`:

```bash
JITNN_HOME=<jitNN> scripts/gpu_final.sh <out dir>
scripts/run_kernels.sh 0:0 <out dir>
```

Each script describes its arguments and environment in its first lines.

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright (c) 2026 Beestream Ltd.
