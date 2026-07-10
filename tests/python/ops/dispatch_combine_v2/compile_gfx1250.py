"""Compile-only harness for the wave32 (MI450 / gfx1250) port of the cco-LSA
dispatch/combine kernels.

Runs WITHOUT a GPU: FLYDSL_COMPILE_ONLY=1 makes @flyc.jit build + codegen the
kernel (amdclang, gfx1250) and return before launch. FLYDSL_GPU_ARCH=gfx1250
avoids rocm_agent_enumerator so no GPU is queried.

Also monkeypatches flydsl's is_rdna_arch so gfx125x (MI450) is classified as
wave32 (the shipped flydsl only special-cases gfx120x, so gfx1250 would wrongly
compile as wave64 and the toolchain rejects +wavefrontsize64 -> hard error).
"""
import os
import sys
import traceback

os.environ.setdefault("COMPILE_ONLY", "1")          # flydsl: compile, do not launch
os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")  # avoid rocm_agent_enumerator (no GPU)

# ── monkeypatch wave32 classification for gfx12x (incl. gfx1250 / MI450) ──
import flydsl.runtime.device as _dev


def _is_rdna(arch=None):
    a = (arch or _dev.get_rocm_arch() or "").lower()
    return a.startswith("gfx10") or a.startswith("gfx11") or a.startswith("gfx12")


_dev.is_rdna_arch = _is_rdna
# is_rdna_arch is imported by-value into these modules; patch each reference.
for _modname in ("flydsl.compiler.backends.rocm", "flydsl.expr.buffer_ops"):
    __import__(_modname)
    setattr(sys.modules[_modname], "is_rdna_arch", _is_rdna)

# ── path: mori python + the op dir (for flydsl_prims) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_MORI_PY = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "python"))
_V2 = os.path.join(_MORI_PY, "mori", "ops", "dispatch_combine_v2")
sys.path.insert(0, _V2)          # flydsl_prims lives here
sys.path.insert(0, _MORI_PY)

# The mori.ops / mori.cco package __init__s eagerly import compiled extensions
# (libmori_pybinds.so, the cco cython ext) that aren't built in this container.
# The device-side kernels only need the pure-python mori.cco.device.flydsl
# subpackage, so stub the parent packages (set __path__, skip their __init__)
# and load intranode_kernels.py directly by file path.
import importlib.util  # noqa: E402
import types  # noqa: E402


def _stub_pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


_stub_pkg("mori", os.path.join(_MORI_PY, "mori"))
_stub_pkg("mori.cco", os.path.join(_MORI_PY, "mori", "cco"))
_stub_pkg("mori.cco.device", os.path.join(_MORI_PY, "mori", "cco", "device"))

_spec = importlib.util.spec_from_file_location(
    "intranode_kernels_gfx1250", os.path.join(_V2, "intranode_kernels.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)      # noqa: E402

# EP8, hidden=7168, topk=8, bf16 reference config (from the op layer).
RANK, NPES = 0, 8
EPR, TOPK = 32, 8
HIDDEN = 7168
MAXTOK = 512
MAXRECV = 4096
SCALE_DIM = HIDDEN // 128       # 56 (fp8 blockwise: block_elems == 128)
# distinct dummy arena offsets (values irrelevant for codegen).
OFF = {n: (i + 1) * (1 << 20) for i, n in enumerate(
    ["tok_off", "recv_num", "tis", "out_idx", "out_wts", "out_tok", "out_scales",
     "comb_inp", "xdb", "comb_wts", "comb_scales"])}

DISP_BW = (64, 16)              # (block_num, warp_num_per_block) -> block = warp*32 = 512
COMB_BW = (80, 4)               # block = 128


def _z(n):
    return [0] * n


def _run(tag, factory, nargs):
    try:
        run = factory()
        run(*_z(nargs))          # FLYDSL_COMPILE_ONLY=1 -> compile, no launch
        print(f"[OK]   {tag}")
        return True
    except Exception as e:       # noqa: BLE001
        print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def _disp_kwargs(**extra):
    kw = dict(rank=RANK, npes=NPES, experts_per_rank=EPR, experts_per_token=TOPK,
              hidden_dim=HIDDEN, hidden_elem_size=2, max_tok_per_rank=MAXTOK, max_recv=MAXRECV,
              off_tok_off=OFF["tok_off"], off_recv_num=OFF["recv_num"], off_tis=OFF["tis"],
              off_out_idx=OFF["out_idx"], off_out_wts=OFF["out_wts"], off_out_tok=OFF["out_tok"])
    kw.update(extra)
    return kw


def main():
    db, dw = DISP_BW
    cb, cw = COMB_BW
    results = []

    # dispatch: plain, +scales, replay
    results.append(_run("dispatch bf16",
                         lambda: K.make_dispatch(block_num=db, warp_num_per_block=dw,
                                                 **_disp_kwargs()), 11))
    results.append(_run("dispatch bf16 +scales",
                         lambda: K.make_dispatch(block_num=db, warp_num_per_block=dw,
                                                 off_out_scales=OFF["out_scales"], scale_dim=SCALE_DIM,
                                                 scale_type_size=4, **_disp_kwargs()), 11))
    results.append(_run("dispatch bf16 replay",
                         lambda: K.make_dispatch(block_num=db, warp_num_per_block=dw, replay=True,
                                                 **_disp_kwargs()), 11))

    # combine gather: bf16, f32
    results.append(_run("combine gather bf16",
                         lambda: K.make_combine(rank=RANK, npes=NPES, experts_per_token=TOPK,
                                                hidden_dim=HIDDEN, hidden_elem_size=2,
                                                max_tok_per_rank=MAXTOK, max_recv=MAXRECV,
                                                block_num=cb, warp_num_per_block=cw,
                                                off_out_tok=OFF["out_tok"], off_xdb_mem=OFF["xdb"],
                                                off_out_wts=OFF["out_wts"]), 9))
    results.append(_run("combine gather f32",
                         lambda: K.make_combine(rank=RANK, npes=NPES, experts_per_token=TOPK,
                                                hidden_dim=HIDDEN, hidden_elem_size=4,
                                                max_tok_per_rank=MAXTOK, max_recv=MAXRECV,
                                                block_num=cb, warp_num_per_block=cw,
                                                off_out_tok=OFF["out_tok"], off_xdb_mem=OFF["xdb"],
                                                off_out_wts=OFF["out_wts"]), 9))

    # combine scatter: plain bf16, fp8_direct_cast, fp8_blockwise
    def _scatter(**extra):
        return K.make_combine_scatter(
            rank=RANK, npes=NPES, experts_per_token=TOPK, hidden_dim=HIDDEN, hidden_elem_size=2,
            max_tok_per_rank=MAXTOK, max_recv=MAXRECV, block_num=cb, warp_num_per_block=cw,
            off_out_tok=OFF["out_tok"], off_comb_inp=OFF["comb_inp"], off_tis=OFF["tis"],
            off_xdb_mem=OFF["xdb"], off_out_wts=OFF["out_wts"], off_comb_wts=OFF["comb_wts"],
            off_comb_scales=OFF["comb_scales"], reset_total_recv=False, **extra)

    results.append(_run("combine scatter bf16", lambda: _scatter(), 9))
    results.append(_run("combine scatter fp8_direct_cast",
                        lambda: _scatter(fp8_direct_cast=True), 9))
    results.append(_run("combine scatter fp8_blockwise",
                        lambda: _scatter(fp8_blockwise=True, scale_dim=SCALE_DIM), 9))

    # StdMoE convert (bf16) + local expert count
    MTPE = NPES * MAXTOK
    results.append(_run("convert_dispatch_output",
                        lambda: K.make_convert_dispatch_output(
                            rank=RANK, experts_per_rank=EPR, experts_per_token=TOPK,
                            hidden_dim=HIDDEN, hidden_elem_size=2, max_tok_per_expert=MTPE,
                            block_num=db, warp_num_per_block=dw), 8))
    results.append(_run("convert_combine_input",
                        lambda: K.make_convert_combine_input(
                            rank=RANK, experts_per_rank=EPR, experts_per_token=TOPK,
                            hidden_dim=HIDDEN, hidden_elem_size=2, max_tok_per_expert=MTPE,
                            block_num=cb, warp_num_per_block=cw), 5))
    results.append(_run("local_expert_count",
                        lambda: K.make_local_expert_count(
                            rank=RANK, experts_per_rank=EPR, experts_per_token=TOPK,
                            block_num=db, warp_num_per_block=dw), 3))

    # combine gather fp4 (gfx1250 pk8 fp4 block converts: v_cvt_scale_pk8_f32_fp4
    # / v_cvt_scalef32_pk8_fp4_f32 — replaces the gfx950-only pk fp4 path).
    results.append(_run("combine gather fp4 (gfx1250 pk8)",
                        lambda: K.make_combine(rank=RANK, npes=NPES, experts_per_token=TOPK,
                                               hidden_dim=HIDDEN, hidden_elem_size=2,
                                               max_tok_per_rank=MAXTOK, max_recv=MAXRECV,
                                               block_num=cb, warp_num_per_block=cw,
                                               off_out_tok=OFF["out_tok"], off_xdb_mem=OFF["xdb"],
                                               off_out_wts=OFF["out_wts"], fp4=True), 9))

    n_ok = sum(results)
    print(f"\n=== {n_ok}/{len(results)} kernels compiled for gfx1250 (wave32) ===")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
