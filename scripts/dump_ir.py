"""Write each compiler stage of a kernel into dumps/<kernel>/.

    python scripts/dump_ir.py --kernel fused_softmax
    python scripts/dump_ir.py --kernel tiled_matmul --arch 90 --set DOT_PRECISION=tf32
    python scripts/dump_ir.py --all

Needs no GPU. See docs/COMPILER_DUMPS.md for how to read the output, and for how to get a trace
of the IR after every individual compiler pass.
"""

import argparse
import json
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import Any

# Allow `python scripts/dump_ir.py` as well as `python -m scripts.dump_ir`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton

from scripts.ir_tools import compile_kernel, read_code_size
from tilekernels import matmul, softmax

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "dumps"
TEXT_STAGES = ("ttir", "ttgir", "llir", "ptx")

KERNELS: dict[str, tuple[Any, dict[str, int | str]]] = {
    "fused_softmax": (softmax._fused_softmax_kernel, {"BLOCK_SIZE": 64}),
    "online_softmax": (softmax._online_softmax_kernel, {"BLOCK_SIZE": 64}),
    # On sm_80 and newer, TF32 lowers tl.dot to tensor core MMA instructions.
    "tiled_matmul": (
        matmul._tiled_matmul_kernel,
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 1, "DOT_PRECISION": "tf32"},
    ),
    # IEEE float32 lowers tl.dot to an unrolled FMA loop instead. The tile is small because the
    # number of instructions grows with BLOCK_M * BLOCK_N * BLOCK_K.
    "tiled_matmul_fma": (
        matmul._tiled_matmul_kernel,
        {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8, "GROUP_M": 1, "DOT_PRECISION": "ieee"},
    ),
}


def strip_local_paths(text: str) -> str:
    """Replace absolute paths in line info, so that dumps do not reveal this machine's layout.

    Paths appear with single and doubled backslashes on Windows, depending on the stage.
    """
    triton_dir = Path(triton.__file__).resolve().parent
    for path, placeholder in ((triton_dir, "<triton>"), (REPO_ROOT, "<repo>")):
        for form in (str(path), str(path).replace("\\", "\\\\"), path.as_posix()):
            text = text.replace(form, placeholder)
    # What follows a placeholder is a relative path, so normalise it to forward slashes.
    return re.sub(
        r"<(?:repo|triton)>[^\"\s]*",
        lambda match: re.sub(r"\\+", "/", match.group(0)),
        text,
    )


def strip_debug_sections(ptx: str) -> str:
    """Remove the DWARF tables at the end of a PTX file.

    These tables hold the checkout path as raw byte values, which text replacement cannot catch,
    so they would make the dump depend on where the repo is. Nothing in them is meant for human
    readers, and the .loc lines that map instructions back to Python are kept.
    """
    return ptx.split(".section", 1)[0].rstrip() + "\n"


def try_disassembly(compiled: Any) -> str | None:
    """Return the SASS disassembly, or None if the CUDA tools it needs are not installed."""
    try:
        return str(compiled.asm["sass"])
    except Exception:
        # Which exception depends on the platform: a missing tool surfaced as an OSError on
        # Windows. SASS is optional, so any failure here only means that it is left out.
        return None


def dump_kernel(
    name: str,
    output_root: Path,
    *,
    overrides: dict[str, int | str],
    dtype: str,
    arch: int,
    num_warps: int,
    num_stages: int,
) -> Path:
    """Compile one kernel and write its stages and metadata. Returns the output folder."""
    kernel, defaults = KERNELS[name]
    constexprs = {**defaults, **overrides}
    result = compile_kernel(
        kernel, constexprs, dtype=dtype, arch=arch, num_warps=num_warps, num_stages=num_stages
    )
    compiled = result.compiled

    folder = output_root / name
    folder.mkdir(parents=True, exist_ok=True)
    for stage in TEXT_STAGES:
        text = compiled.asm[stage]
        text = strip_debug_sections(text) if stage == "ptx" else text
        (folder / f"{name}.{stage}").write_text(
            strip_local_paths(text), encoding="utf-8", newline="\n"
        )
    (folder / f"{name}.cubin").write_bytes(compiled.asm["cubin"])

    sass = try_disassembly(compiled)
    if sass is not None:
        (folder / f"{name}.sass").write_text(sass, encoding="utf-8", newline="\n")

    metadata = {
        "kernel": kernel.__name__,
        "constexprs": constexprs,
        "dtype": dtype,
        "target": f"sm_{arch}",
        "num_warps": num_warps,
        "num_stages": num_stages,
        "threads_per_program": num_warps * 32,
        "registers": result.stats.registers,
        "spill_store_bytes": result.stats.spill_store_bytes,
        "spill_load_bytes": result.stats.spill_load_bytes,
        "code_bytes": read_code_size(compiled.asm["cubin"], kernel.__name__),
        "shared_memory_bytes": compiled.metadata.shared,
        "enable_fp_fusion": compiled.metadata.enable_fp_fusion,
        "stages_produced": sorted(key for key in compiled.asm if key != "source"),
        "disassembly": "included" if sass is not None else "not available without CUDA tools",
        "triton_version": triton.__version__,
        "compiled_on": platform.system(),
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return folder


def parse_constexpr(text: str) -> tuple[str, int | str]:
    """Parse a NAME=VALUE argument. The value is an int if it looks like one."""
    name, separator, value = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Dump each compiler stage of a kernel.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--kernel", choices=sorted(KERNELS))
    choice.add_argument("--all", action="store_true", help="dump every kernel")
    parser.add_argument("--arch", type=int, default=86, help="CUDA architecture, 86 means sm_86")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "fp16"))
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument(
        "--set", type=parse_constexpr, action="append", default=[], metavar="NAME=VALUE",
        help="override a constexpr, for example --set BLOCK_SIZE=128",
    )  # fmt: skip
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Dump the requested kernels and print where they went."""
    args = parse_args()
    warnings.filterwarnings("ignore", message="TRITON_DEFAULT_FP_FUSION is unset")
    warnings.filterwarnings("ignore", message="Failed to find executable")

    for name in sorted(KERNELS) if args.all else [args.kernel]:
        folder = dump_kernel(
            name,
            args.output,
            overrides=dict(args.set),
            dtype=args.dtype,
            arch=args.arch,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        print(f"{name}: wrote {folder}")


if __name__ == "__main__":
    main()
