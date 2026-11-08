# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""cuBLAS GEMM peaks on this GPU through PyTorch: FP32 (TF32 off), TF32, FP16 and BF16, square sizes."""
import torch, time
torch.backends.cuda.matmul.allow_tf32 = False
def peak(dtype, n, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5): a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); a @ b; e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); ms = ts[len(ts) // 2]
    return 2 * n ** 3 / ms / 1e9
for n in (4096, 8192):
    print(f"n={n}: fp32 {peak(torch.float32, n):.1f}  tf32 {peak(torch.float32, n, True):.1f}  fp16 {peak(torch.float16, n):.1f}  bf16 {peak(torch.bfloat16, n):.1f} TFLOP/s", flush=True)
a = torch.empty(2 << 28, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
ts = []
for _ in range(20):
    s.record(); b.copy_(a); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
ts.sort(); print(f"torch copy: {2 * a.numel() * 4 / ts[10] / 1e6:.1f} GB/s", flush=True)
