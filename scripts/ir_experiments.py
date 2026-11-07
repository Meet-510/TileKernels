"""Compile kernel variants and record how the generated code changes.

    python scripts/ir_experiments.py

Each experiment changes one thing and counts what it does to the generated code. The results
go to dumps/experiments/results.md, which docs/COMPILER_DUMPS.md quotes. Nothing here runs a
kernel or takes a timing, so nothing here needs a GPU. These counts are static: they describe
the code, and not how fast it runs.
"""

import re
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton
import triton.language as tl

from scripts.ir_tools import compile_kernel, read_code_size
from tilekernels import matmul, softmax

OUTPUT = Path(__file__).resolve().parents[1] / "dumps" / "experiments" / "results.md"

SOFTMAX_HEADERS = [
    "variant", "sizePerThread", "registers", "code bytes", "shared bytes",
    "shuffles", "barriers", "global loads", "of which predicated",
]  # fmt: skip


@triton.jit
def softmax_unit_stride(x_ptr, y_ptr, n_cols, row_stride, BLOCK_SIZE: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    """The fused softmax with a column stride known at compile time, as a contiguous row has."""
    row_offset = tl.program_id(axis=0) * row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    in_row = cols < n_cols
    x = tl.load(x_ptr + row_offset + cols, mask=in_row, other=-float("inf"))
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(y_ptr + row_offset + cols, y, mask=in_row)


@triton.jit
def softmax_unit_stride_no_mask(x_ptr, y_ptr, row_stride, BLOCK_SIZE: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    """The same kernel with no masks. It is only correct if n_cols is exactly BLOCK_SIZE."""
    row_offset = tl.program_id(axis=0) * row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row_offset + cols)
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(y_ptr + row_offset + cols, y)


@triton.jit
def matmul_unit_stride(  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_m,
    b_stride_k,
    c_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """The tiled matmul with contiguous rows, so that A, B and C have a column stride of 1."""
    program = tl.program_id(axis=0)
    n_tiles_n = tl.cdiv(N, BLOCK_N)
    rows = (program // n_tiles_n * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    cols = (program % n_tiles_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for first_k in range(0, K, BLOCK_K):
        k_range = first_k + tl.arange(0, BLOCK_K)
        a_k = k_range[None, :]
        b_k = k_range[:, None]
        a = tl.load(a_ptr + rows * a_stride_m + a_k, mask=(rows < M) & (a_k < K), other=0.0)
        b = tl.load(b_ptr + b_k * b_stride_k + cols, mask=(b_k < K) & (cols < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
    tl.store(c_ptr + rows * c_stride_m + cols, accumulator, mask=(rows < M) & (cols < N))


def count(pattern: str, text: str) -> int:
    """Count the matches of a regular expression."""
    return len(re.findall(pattern, text))


def _code_of(ptx: str) -> str:
    """Return only the kernel body of a PTX file, without the debug sections after it."""
    return ptx.split(".section", 1)[0]


def _load_summary(ptx: str) -> str:
    """List each distinct global load instruction with how many times it appears."""
    loads = re.findall(r"ld\.global(?:\.\w+)*", ptx)
    return ", ".join(f"{loads.count(op)} x `{op}`" for op in sorted(set(loads)))


def _size_per_thread(ttgir: str) -> str:
    """List the distinct sizePerThread values in the final layouts of a kernel.

    This is how many contiguous elements a thread holds, and not its total. With
    BLOCK_SIZE=1024 on 128 threads, it is 1, while a thread holds 8 elements 128 apart.
    """
    sizes = sorted(set(re.findall(r"sizePerThread = \[(\d+)\]", ttgir)), key=int)
    return ", ".join(sizes) if sizes else "-"


def softmax_row(
    kernel: Any, label: str, constexprs: dict[str, int | str], num_warps: int
) -> list[object]:
    """Summarise one softmax variant as a table row."""
    result = compile_kernel(kernel, constexprs, num_warps=num_warps)
    compiled = result.compiled
    ptx = _code_of(compiled.asm["ptx"])
    return [
        label,
        _size_per_thread(compiled.asm["ttgir"]),
        result.stats.registers,
        read_code_size(compiled.asm["cubin"], kernel.__name__),
        compiled.metadata.shared,
        count(r"shfl\.sync", ptx),
        count(r"bar\.sync", ptx),
        _load_summary(ptx),
        count(r"@%p\d+ ld\.global", ptx),
    ]


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Format a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def experiment_num_warps() -> str:
    """How does num_warps change the code for one fixed block?"""
    rows = [
        softmax_row(
            softmax._fused_softmax_kernel,
            f"BLOCK_SIZE=64, num_warps={warps}",
            {"BLOCK_SIZE": 64},
            warps,
        )
        for warps in (1, 2, 4, 8)
    ]
    return "## Experiment 1: num_warps\n\n" + markdown_table(SOFTMAX_HEADERS, rows)


def experiment_block_size() -> str:
    """How does BLOCK_SIZE change the code for a fixed number of warps?"""
    rows = [
        softmax_row(
            softmax._fused_softmax_kernel,
            f"BLOCK_SIZE={block}, num_warps=4",
            {"BLOCK_SIZE": block},
            4,
        )
        for block in (16, 64, 128, 256, 1024)
    ]
    return "## Experiment 2: BLOCK_SIZE\n\n" + markdown_table(SOFTMAX_HEADERS, rows)


def experiment_strides_and_masks() -> str:
    """What do a runtime stride and a mask each cost in the generated code?

    The block is 1024 on 128 threads, so each thread holds 8 elements. A smaller block would
    hide the effect, because a thread with one element has nothing to vectorize.
    """
    block: dict[str, int | str] = {"BLOCK_SIZE": 1024}
    rows = [
        softmax_row(softmax._fused_softmax_kernel, "runtime column stride, masked", block, 4),
        softmax_row(softmax_unit_stride, "column stride fixed at 1, masked", block, 4),
        softmax_row(softmax_unit_stride_no_mask, "column stride fixed at 1, no mask", block, 4),
    ]
    return "## Experiment 3: runtime strides and masks\n\n" + markdown_table(SOFTMAX_HEADERS, rows)


def matmul_stride_row(kernel: Any, label: str, constexprs: dict[str, int | str]) -> list[object]:
    """Summarise one matmul variant as a table row."""
    result = compile_kernel(kernel, constexprs)
    ptx = _code_of(result.compiled.asm["ptx"])
    copy_sizes = sorted(set(re.findall(r"cp\.async\.ca\.shared\.global .*?, (0x[0-9a-f]+),", ptx)))
    return [
        label,
        result.stats.registers,
        read_code_size(result.compiled.asm["cubin"], kernel.__name__),
        count(r"cp\.async\.ca\.shared\.global", ptx),
        ", ".join(copy_sizes),
    ]


def experiment_matmul_strides() -> str:
    """Do runtime column strides cost anything in the tiled matmul?"""
    blocks: dict[str, int | str] = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
    rows = [
        matmul_stride_row(
            matmul._tiled_matmul_kernel,
            "runtime strides",
            {**blocks, "GROUP_M": 1, "DOT_PRECISION": "tf32"},
        ),
        matmul_stride_row(matmul_unit_stride, "column strides fixed at 1", blocks),
    ]
    headers = ["variant", "registers", "code bytes", "async copies", "bytes per copy"]
    return "## Experiment 6: runtime strides in matmul\n\n" + markdown_table(headers, rows)


def _matmul_constexprs(precision: str) -> dict[str, int | str]:
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 1,
        "DOT_PRECISION": precision,
    }


def dot_lowering_row(arch: int, dtype: str, precision: str) -> list[object]:
    """Summarise what tl.dot became for one target, dtype and precision."""
    result = compile_kernel(
        matmul._tiled_matmul_kernel, _matmul_constexprs(precision), dtype=dtype, arch=arch
    )
    ttgir = result.compiled.asm["ttgir"]
    ptx = _code_of(result.compiled.asm["ptx"])
    dot_op = next(
        (op for op in ("ttng.tc_gen5_mma", "ttng.warp_group_dot", "tt.dot") if op in ttgir), "?"
    )
    return [
        f"sm_{arch}",
        dtype,
        precision,
        f"`{dot_op}`",
        count(r"\bfma\.", ptx),
        count(r"\bmma\.|wgmma\.|tcgen05\.mma", ptx),
        result.stats.registers,
        result.stats.spill_store_bytes,
        result.compiled.metadata.shared,
    ]


def experiment_dot_lowering() -> str:
    """What does tl.dot become on each architecture?"""
    rows = [
        dot_lowering_row(arch, dtype, precision)
        for arch in (75, 80, 86, 90, 100)
        for dtype, precision in (("fp32", "ieee"), ("fp32", "tf32"), ("fp16", "ieee"))
    ]
    headers = [
        "target", "dtype", "precision", "dot op in TTGIR", "fma in PTX", "tensor core ops in PTX",
        "registers", "spill store bytes", "shared bytes",
    ]  # fmt: skip
    return "## Experiment 4: what tl.dot becomes\n\n" + markdown_table(headers, rows)


def experiment_num_stages() -> str:
    """Does num_stages change the generated code on each architecture?"""
    rows: list[list[object]] = []
    kernel = matmul._tiled_matmul_kernel
    for arch in (75, 86, 90):
        baseline = None
        for stages in (1, 2, 3, 4):
            result = compile_kernel(
                kernel, _matmul_constexprs("tf32"), arch=arch, num_stages=stages
            )
            ptx = _code_of(result.compiled.asm["ptx"])
            baseline = ptx if baseline is None else baseline
            rows.append(
                [
                    f"sm_{arch}",
                    stages,
                    result.compiled.metadata.shared,
                    result.stats.registers,
                    read_code_size(result.compiled.asm["cubin"], kernel.__name__),
                    "yes" if ptx == baseline else "no",
                ]
            )
    headers = ["target", "num_stages", "shared bytes", "registers", "code bytes",
               "same PTX as num_stages=1"]  # fmt: skip
    return "## Experiment 5: num_stages\n\n" + markdown_table(headers, rows)


def main() -> None:
    """Run every experiment and write the results file."""
    warnings.filterwarnings("ignore", message="TRITON_DEFAULT_FP_FUSION is unset")
    warnings.filterwarnings("ignore", message="Failed to find executable")
    sections = [
        "# Compiler experiments",
        f"Generated by `scripts/ir_experiments.py` with Triton {triton.__version__}. Unless "
        "stated, the target is sm_86 and the dtype is float32. These are static counts from "
        "generated code, not timings.",
        experiment_num_warps(),
        experiment_block_size(),
        experiment_strides_and_masks(),
        experiment_dot_lowering(),
        experiment_num_stages(),
        experiment_matmul_strides(),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n\n".join(sections) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
