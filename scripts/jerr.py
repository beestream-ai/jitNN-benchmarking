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
"""Max error of jitNN's saved inference outputs (<name>.jitnn.f32) against gsweep.py's FP32 references
(<shape>.ref.npy), relative to the reference's largest magnitude: python jerr.py <work dir> <jitnn out dir>..."""
import glob
import os
import sys

import numpy as np

work = sys.argv[1]
for d in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jitnn.f32"))):
        n = os.path.basename(f)[:-len(".jitnn.f32")]
        ref = os.path.join(work, n.rsplit("_", 1)[0] + ".ref.npy")
        if not os.path.exists(ref):
            print(f"[jerr] {os.path.basename(d)} {n} no reference")
            continue
        r = np.load(ref).ravel()
        y = np.fromfile(f, dtype="<f4").astype(np.float64)
        if y.size != r.size:
            print(f"[jerr] {os.path.basename(d)} {n} size {y.size} vs {r.size}")
            continue
        print(f"[jerr] {os.path.basename(d):10s} {n:28s} err {np.max(np.abs(y - r)) / max(np.max(np.abs(r)), 1e-30):.1e}")
