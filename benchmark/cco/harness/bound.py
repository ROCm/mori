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
"""Turn split-path measurements into fusion bounds and a go / no-go per bucket.

Step 1 of the ``cco-fusion-loop`` skill; the model is explained in
``.claude/skills/cco-fusion-loop/reference/bounds.md``. Per bucket, the input is
the split path's phases, measured inside the pipeline (maximum over ranks), each
tagged with a role:

* ``compute``: the GEMM, and whatever a fused GEMM would take on (``a``);
* ``overlap``: transfer legs that depend only on this GEMM's output (``b``);
* ``serial``: legs that must wait for something else, such as the reduce and the
  all-gather of an all-reduce (``T_serial``).

Then

* ``T_ideal = T_serial + max(a, b)``: every overlappable leg hidden;
* ``T_real(n) = T_serial + c + m + (n - 1) * max(c, m)`` with ``c = a / n`` and
  ``m = (b + (n - 1) * t0) / n``: a two-stage pipeline of ``n`` equal chunks per
  destination where every extra put costs ``t0``. Its two limits are the
  compute-bound ``a + b/n + t0`` and the communication-bound
  ``a/n + b + (n - 1) * t0`` of the reference.

A concurrent run (the GEMM and the overlap legs on two streams, on independent
buffers) measures contention: when it takes longer than ``max(a, b)``, the larger
side is stretched to it.

The decision is pessimistic: the realistic saving uses the high per-put cost and
the best chunk count for it. Go when that saving is at least
``max(min_gain_pct, noise_band_pct)`` of ``T_split``, there is something on both
sides to overlap, and the GEMM has at least one tile per CU.

Input is JSON, or YAML when PyYAML is installed; ``bounds.example.yaml`` is the
gemm_ar worked example. Exit status: 0 on success (whatever the decisions), 2 when
the input cannot be used.

    python bound.py bounds.yaml --out-md bounds.md --out-json bounds.json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROLES = ("compute", "overlap", "serial")


class InputError(ValueError):
    """The input cannot be turned into a bound."""


@dataclass(frozen=True)
class Config:
    t0_low: float
    t0_high: float
    n_values: tuple
    noise_band_pct: float
    min_gain_pct: float
    cus: int | None

    @property
    def threshold_pct(self):
        return max(self.min_gain_pct, self.noise_band_pct)


def _number(value, what, minimum=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{what} must be a number, got {value!r}") from None
    if math.isnan(x) or x < minimum:
        raise InputError(f"{what} must be >= {minimum}, got {value!r}")
    return x


def load_input(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise InputError(
                f"{path}: reading YAML needs PyYAML; install it or pass JSON"
            ) from None
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise InputError(f"{path}: not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise InputError(f"{path}: the top level must be a mapping")
    return data


def make_config(data, min_gain_pct=None, noise_band_pct=None):
    t0 = data.get("t0_us", [4, 9])
    if isinstance(t0, (list, tuple)):
        if len(t0) != 2:
            raise InputError("t0_us must be a number or [low, high]")
        t0_low, t0_high = (_number(x, "t0_us") for x in t0)
    else:
        t0_low = t0_high = _number(t0, "t0_us")
    if t0_low > t0_high:
        raise InputError("t0_us: low is above high")

    if "n" in data:
        values = data["n"]
        if not isinstance(values, (list, tuple)) or not values:
            raise InputError("n must be a non-empty list of chunk counts")
        n_values = tuple(sorted({int(_number(v, "n", 1)) for v in values}))
    else:
        n_max = int(_number(data.get("n_max", 16), "n_max", 1))
        n_values = tuple(range(1, n_max + 1))

    cus = data.get("cus")
    return Config(
        t0_low=t0_low,
        t0_high=t0_high,
        n_values=n_values,
        noise_band_pct=_number(
            (
                data.get("noise_band_pct", 1.0)
                if noise_band_pct is None
                else noise_band_pct
            ),
            "noise_band_pct",
        ),
        min_gain_pct=_number(
            data.get("min_gain_pct", 5.0) if min_gain_pct is None else min_gain_pct,
            "min_gain_pct",
        ),
        cus=None if cus is None else int(_number(cus, "cus", 1)),
    )


def pipeline_us(a, b, n, t0):
    """Two-stage pipeline of ``n`` equal chunks: compute ``a``, then transfer ``b``.

    Every chunk beyond the first adds one put, at ``t0`` each, to the transfer.
    """
    c = a / n
    m = (b + (n - 1) * t0) / n
    return c + m + (n - 1) * max(c, m)


def _pct(saving, t_split):
    return 100.0 * saving / t_split


def evaluate_bucket(bucket, cfg):
    if not isinstance(bucket, dict):
        raise InputError("every bucket must be a mapping")
    label = bucket.get("m", "?")
    phases = bucket.get("phases")
    if not isinstance(phases, list) or not phases:
        raise InputError(f"bucket m={label}: phases must be a non-empty list")

    sums = dict.fromkeys(ROLES, 0.0)
    for phase in phases:
        role = phase.get("role") if isinstance(phase, dict) else None
        if role not in ROLES:
            raise InputError(
                f"bucket m={label}: phase {phase!r} needs a role in {ROLES}"
            )
        sums[role] += _number(phase.get("us"), f"bucket m={label}: phase us")
    a, b, serial = sums["compute"], sums["overlap"], sums["serial"]
    phase_sum = a + b + serial
    if phase_sum <= 0:
        raise InputError(f"bucket m={label}: the phases add up to zero")

    notes = []
    if bucket.get("split_us") is None:
        t_split = phase_sum
        notes.append("split_us not given: T_split is the sum of the phases")
    else:
        t_split = _number(bucket["split_us"], f"bucket m={label}: split_us")
        if t_split <= 0:
            raise InputError(f"bucket m={label}: split_us must be positive")
        if abs(t_split - phase_sum) > 0.05 * t_split:
            notes.append(
                f"the phases add up to {phase_sum:.1f} us but split_us is "
                f"{t_split:.1f} us: check the phase list"
            )

    a_eff, b_eff = a, b
    if bucket.get("concurrent_us") is not None:
        concurrent = _number(
            bucket["concurrent_us"], f"bucket m={label}: concurrent_us"
        )
        if concurrent > max(a, b):
            side = "overlap" if b >= a else "compute"
            if b >= a:
                b_eff = concurrent
            else:
                a_eff = concurrent
            notes.append(
                f"contention: the concurrent run took {concurrent:.1f} us against "
                f"max(a, b) = {max(a, b):.1f} us, so the {side} side is stretched to it"
            )
        else:
            notes.append(
                f"the concurrent run ({concurrent:.1f} us) shows no contention "
                f"beyond max(a, b) = {max(a, b):.1f} us"
            )

    t_ideal = serial + max(a_eff, b_eff)

    sweep = []
    for n in cfg.n_values:
        low = serial + pipeline_us(a_eff, b_eff, n, cfg.t0_low)
        high = serial + pipeline_us(a_eff, b_eff, n, cfg.t0_high)
        bound = (
            "compute"
            if a_eff / n >= (b_eff + (n - 1) * cfg.t0_high) / n
            else "communication"
        )
        sweep.append({"n": n, "t_real_us": [low, high], "bound": bound})
    best_low = min(sweep, key=lambda r: r["t_real_us"][0])
    best_high = min(sweep, key=lambda r: r["t_real_us"][1])
    t_real = [best_low["t_real_us"][0], best_high["t_real_us"][1]]

    tiles = bucket.get("tiles")
    tiles = (
        None if tiles is None else int(_number(tiles, f"bucket m={label}: tiles", 1))
    )
    rounds = None
    if tiles is not None and cfg.cus is not None:
        rounds = math.ceil(tiles / cfg.cus)

    saving_ideal = t_split - t_ideal
    saving_real = [t_split - t_real[1], t_split - t_real[0]]  # pessimistic first
    reasons = []
    if a <= 0 or b <= 0:
        reasons.append(
            "nothing to overlap: there is no compute phase or no overlap phase"
        )
    if tiles is not None and cfg.cus is not None and tiles < cfg.cus:
        reasons.append(
            f"{tiles} tiles for {cfg.cus} CUs: all tiles finish together, so there is "
            "nothing to pipeline (G950-ENV-02)"
        )
    pessimistic_pct = _pct(saving_real[0], t_split)
    if pessimistic_pct < cfg.threshold_pct:
        reasons.append(
            f"realistic saving {pessimistic_pct:.1f}% (high t0, best n) is below "
            f"{cfg.threshold_pct:.1f}%"
        )
    go = not reasons
    if go:
        reasons.append(
            f"realistic saving {pessimistic_pct:.1f}% (high t0, best n) reaches "
            f"{cfg.threshold_pct:.1f}%"
        )

    return {
        "m": bucket.get("m"),
        "t_split_us": t_split,
        "a_us": a,
        "b_us": b,
        "serial_us": serial,
        "a_eff_us": a_eff,
        "b_eff_us": b_eff,
        "t_ideal_us": t_ideal,
        "saving_ideal_us": saving_ideal,
        "saving_ideal_pct": _pct(saving_ideal, t_split),
        "best_n": {"t0_low": best_low["n"], "t0_high": best_high["n"]},
        "t_real_us": t_real,
        "saving_real_us": saving_real,
        "saving_real_pct": [_pct(s, t_split) for s in saving_real],
        "tiles": tiles,
        "tile_rounds_at_one_block_per_cu": rounds,
        "sweep": sweep,
        "decision": "go" if go else "no-go",
        "reasons": reasons,
        "notes": notes,
    }


def evaluate(data, cfg):
    buckets = data.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise InputError("buckets must be a non-empty list")
    return [evaluate_bucket(b, cfg) for b in buckets]


def _range(pair, digits=1):
    low, high = pair
    if abs(high - low) < 10**-digits:
        return f"{low:.{digits}f}"
    return f"{low:.{digits}f}-{high:.{digits}f}"


def _t0_text(cfg):
    if cfg.t0_low == cfg.t0_high:
        return f"{cfg.t0_low:g} us"
    return f"{cfg.t0_low:g}-{cfg.t0_high:g} us"


def render_markdown(cfg, results):
    n_text = (
        f"{cfg.n_values[0]}-{cfg.n_values[-1]}"
        if len(cfg.n_values) > 1
        else str(cfg.n_values[0])
    )
    out = [
        "# Fusion bounds",
        "",
        f"t0 {_t0_text(cfg)}; n {n_text}; noise band {cfg.noise_band_pct:g}%; "
        f"minimum gain {cfg.min_gain_pct:g}%. Times in us.",
        "",
        "| M | T_split | a | b | T_serial | T_ideal | ideal saving | best n | T_real | realistic saving | decision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        best = r["best_n"]
        best_n = (
            str(best["t0_high"])
            if best["t0_low"] == best["t0_high"]
            else f"{best['t0_high']} (high t0), {best['t0_low']} (low t0)"
        )
        out.append(
            f"| {r['m']} | {r['t_split_us']:.1f} | {r['a_eff_us']:.1f} | {r['b_eff_us']:.1f} | "
            f"{r['serial_us']:.1f} | {r['t_ideal_us']:.1f} | "
            f"{r['saving_ideal_us']:.1f} ({r['saving_ideal_pct']:.1f}%) | {best_n} | "
            f"{_range(r['t_real_us'])} | {_range(r['saving_real_us'])} "
            f"({_range(r['saving_real_pct'])}%) | {r['decision']} |"
        )
    for r in results:
        out += ["", f"## M = {r['m']}", ""]
        out.append(
            f"**Decision: {r['decision']}.** "
            + " ".join(f"{s[0].upper()}{s[1:]}." for s in r["reasons"])
        )
        if r["notes"]:
            out.append("")
            out += [f"- {note}" for note in r["notes"]]
        if r["tile_rounds_at_one_block_per_cu"] is not None:
            out.append(
                f"- {r['tiles']} tiles: {r['tile_rounds_at_one_block_per_cu']} round(s) at one "
                "block per CU; chunks smaller than a round finish unevenly, which this model ignores"
            )
        out += [
            "",
            "| n | T_real | realistic saving | bound by |",
            "|---:|---:|---:|---|",
        ]
        for row in r["sweep"]:
            low, high = row["t_real_us"]
            saving = [r["t_split_us"] - high, r["t_split_us"] - low]
            out.append(
                f"| {row['n']} | {_range([low, high])} | {_range(saving)} | {row['bound']} |"
            )
    out += [
        "",
        "Model: a two-stage pipeline of n equal chunks per destination, each extra put "
        "costing t0, with the compute and overlap times adjusted for contention when a "
        "concurrent run is given. It ignores the put-size knee (G950-SDMA-04) and uneven "
        "chunk completion. See .claude/skills/cco-fusion-loop/reference/bounds.md.",
        "",
    ]
    return "\n".join(out)


def _rounded(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fusion bounds and go / no-go per bucket from split-path phase times."
    )
    parser.add_argument("input", help="JSON or YAML file; see bounds.example.yaml")
    parser.add_argument("--out-md", help="write the report here instead of stdout")
    parser.add_argument("--out-json", help="also write the results as JSON")
    parser.add_argument("--min-gain-pct", type=float, help="override min_gain_pct")
    parser.add_argument("--noise-band-pct", type=float, help="override noise_band_pct")
    args = parser.parse_args(argv)

    try:
        data = load_input(args.input)
        cfg = make_config(data, args.min_gain_pct, args.noise_band_pct)
        results = evaluate(data, cfg)
    except (InputError, OSError) as e:
        print(f"bound.py: {e}", file=sys.stderr)
        return 2

    report = render_markdown(cfg, results)
    if args.out_md:
        Path(args.out_md).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    if args.out_json:
        payload = {
            "config": {
                "t0_us": [cfg.t0_low, cfg.t0_high],
                "n": list(cfg.n_values),
                "noise_band_pct": cfg.noise_band_pct,
                "min_gain_pct": cfg.min_gain_pct,
                "cus": cfg.cus,
            },
            "buckets": results,
        }
        Path(args.out_json).write_text(
            json.dumps(_rounded(payload), indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
