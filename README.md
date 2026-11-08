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

Apache License 2.0, see [LICENSE](LICENSE). Copyright (c) 2026 Beestream Ltd.
