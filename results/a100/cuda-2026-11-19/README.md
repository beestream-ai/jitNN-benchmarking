# NVIDIA A100, CUDA, 2026-11-19

The GPU head-to-head of jitNN against PyTorch and ONNX Runtime, `scripts/gpu_final.sh` with
`FRAMEWORKS="pt ort" ROUNDS=3`, summarised by `scripts/gpu_summary.py` into `summary.txt`.

| | |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, driver 610.57, CUDA 13.4, max SM clock 1410 MHz |
| Host | AArch64 (Arm Neoverse-N1), 128 cores |
| jitNN | `main` of 2026-11-19 (`fadd058`; the same library code as release 0.1.0-alpha) on TornadoVM 8.1.0 (CUDA), JDK 25.0.2 |
| PyTorch | 2.14.1 (CUDA 13.0), `torch.compile` in the fastest mode per row |
| ONNX Runtime | 1.30.0, CUDA provider, built from source for AArch64 |
| TensorRT | not run (no build for this host) |

- `final.txt`: every timed row of the three interleaved rounds (framework order rotated per round),
  with the peak GPU memory of each run (`[mem] <run> <peak MB> <idle MB>`).
- `summary.txt`: per row and precision, the median over the rounds of each framework's host-to-host
  time, first-call time and peak memory, and jitNN's ratio to the faster of PyTorch and ONNX Runtime.
- `logs/`: the output of each run.

This ONNX Runtime build fails with CUDA error 700 (illegal memory access) in its CUDA-graph replay on
every BERT row, so BERT runs it without CUDA graphs (`opt-exh`) and ResNet-50 with them
(`exh-graph`). jitNN's test suites passed on the same machine before the run (the only failure, the
GEMM autotuning test under `jitnn.native=auto`, is a test expectation, fixed afterwards).
