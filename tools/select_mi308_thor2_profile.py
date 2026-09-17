# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Print public EpDispatchCombineConfig overrides for an exact measured case."""
import argparse
import json
import pathlib

MODELS = {"r1": (7168, 8, 16), "v4_flash": (4096, 6, 16), "v4_pro": (7168, 6, 24)}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--profiles",
    type=pathlib.Path,
    default=pathlib.Path(__file__).resolve().parents[1]
    / "docs/tuning/mi308x_thor2_ep16/profiles.json",
)
p.add_argument("--model", choices=sorted(MODELS), required=True)
p.add_argument("--dtype", choices=("bf16", "fp8"), required=True)
p.add_argument("--kernel", choices=("v2", "v2_ll"), required=True)
p.add_argument("--tokens", type=int, required=True)
args = p.parse_args()
hidden, topk, epr = MODELS[args.model]
wire = "bf16" if args.dtype == "bf16" else "fp8_e4m3fnuz"
payload = json.loads(args.profiles.read_text())
if payload.get("schema_version") != 1:
    p.error("Joint validation profiles have not been generated yet.")
rows = payload["profiles"]
matches = [
    r
    for r in rows
    if r["device"] == "mi308x"
    and r["world_size"] == 16
    and r["hidden_size"] == hidden
    and r["topk"] == topk
    and r["experts_per_rank"] == epr
    and r["dispatch_dtype"] == wire
    and r["combine_dtype"] == "bf16"
    and r["tokens"] == args.tokens
    and r["kernel"] == args.kernel
]
if len(matches) != 1:
    p.error("No unique exact measured profile; no interpolation is performed.")
r = matches[0]
d, c = r["dispatch"], r["combine"]
config = dict(
    kernel_backend="hip",
    internode_kernel=r["kernel"],
    num_qp_per_pe=r["num_qp"],
    dispatch_block_num=d[0],
    dispatch_rdma_block_num=d[1],
    warp_num_per_block=d[2],
    combine_block_num=c[0],
    combine_rdma_block_num=c[1],
    combine_warp_num_per_block=c[2],
)
print(
    json.dumps(
        dict(
            model=args.model,
            tokens=args.tokens,
            dispatch_dtype=wire,
            combine_dtype="bf16",
            hardware_profile=r["network"],
            measured_capacity=r["max_num_inp_token_per_rank"],
            validated_change=r["tuned"],
            config_overrides=config,
        ),
        indent=2,
    )
)
