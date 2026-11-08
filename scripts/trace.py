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
"""TorchScript versions of inference.py's models, for DJL's PyTorch engine: one traced file per configuration.

usage: python trace.py <work dir>
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference  # noqa: E402


def main(work):
    os.makedirs(work, exist_ok=True)
    dev = torch.device("cuda")
    configs = [("resnet50", b, None) for b in inference.RESNET_BATCHES] + [("bert", b, s) for b, s in inference.BERT_SHAPES]
    nets = {}
    for kind, b, s in configs:
        if kind not in nets:
            nets[kind] = (inference.models.resnet50(3, 1000, False, 1) if kind == "resnet50" else inference.BertEncoder()).eval()
        base = f"resnet50_b{b}" if kind == "resnet50" else f"bert_b{b}_s{s}"
        for dtype in ("fp32", "fp16"):
            net = nets[kind].to(dev) if dtype == "fp32" else copy.deepcopy(nets[kind]).half().to(dev)
            x = torch.from_numpy(inference.images(b) if kind == "resnet50" else inference.tokens(b, s)).to(dev)
            if kind == "resnet50" and dtype == "fp16":
                x = x.half()
            path = os.path.join(work, f"{base}_{dtype}.pt")
            if not os.path.exists(path):
                with torch.no_grad():
                    torch.jit.trace(net, (x,)).save(path)
            print("traced", path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
