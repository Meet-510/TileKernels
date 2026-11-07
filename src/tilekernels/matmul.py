"""Tiled matrix multiplication, C = A @ B.

C is cut into a 2D grid of BLOCK_M x BLOCK_N tiles, and each program computes one tile. To
compute it, a program loops over K in steps of BLOCK_K and accumulates the product of a
BLOCK_M x BLOCK_K tile of A and a BLOCK_K x BLOCK_N tile of B:

    C[m_tile, n_tile] = sum over k_tile of A[m_tile, k_tile] @ B[k_tile, n_tile]
"""

from typing import Literal

import torch
import triton
import triton.language as tl

from tilekernels.utils import validate_block_size, validate_input, validate_output

DotPrecision = Literal["ieee", "tf32"]

# When lowering tl.dot for a CUDA target, Triton rejects a smaller K dimension. The interpreter
# has no such limit, so the wrapper checks this in every mode. That way a configuration cannot
# pass the interpreter tests and then fail to compile on a GPU. K itself may still be smaller,
# because the mask handles it.
_MIN_BLOCK_K = {torch.float32: 8, torch.float16: 16}

# Kernel arguments stay unannotated, except for tl.constexpr, because Triton's frontend parses
# them itself: pointers and integers get their types from the launch arguments. That is the
# reason for each targeted `type: ignore[no-untyped-def]` in this module.


@triton.jit
def _program_to_tile(program, n_tiles_m, n_tiles_n, GROUP_M: tl.constexpr):  # type: ignore[no-untyped-def]
    """Map a program id to the (row, column) of the C tile it computes.

    Programs run roughly in id order, so this mapping decides which memory is touched at
    around the same time. With GROUP_M = 1 the order is row-major: moving from one column
    to the next loads a new tile of B each time.

    With GROUP_M > 1, tiles are walked column by column inside a group of GROUP_M rows:

        GROUP_M = 1           GROUP_M = 2
        0  1  2  3            0  2  4  6
        4  5  6  7            1  3  5  7
        8  9  10 11           8  10 12 14
                              9  11 13 15

    Consecutive programs now share a column, so they read the same tiles of B, and those
    are more likely to still be in the L2 cache. The last group may hold fewer rows, which
    is why its height is computed instead of assumed to be GROUP_M.
    """
    if GROUP_M == 1:
        return program // n_tiles_n, program % n_tiles_n
    else:
        programs_per_group = GROUP_M * n_tiles_n
        first_row = (program // programs_per_group) * GROUP_M
        group_height = tl.minimum(n_tiles_m - first_row, GROUP_M)
        position = program % programs_per_group
        return first_row + position % group_height, position // group_height


@triton.jit
def _tiled_matmul_kernel(  # type: ignore[no-untyped-def]
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_m,
    a_stride_k,
    b_stride_k,
    b_stride_n,
    c_stride_m,
    c_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Compute one BLOCK_M x BLOCK_N tile of C."""
    tile_m, tile_n = _program_to_tile(
        tl.program_id(axis=0), tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), GROUP_M
    )
    # The extra axes broadcast each range against the other to address a 2D tile.
    rows = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    cols = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]

    # Accumulate in float32, whatever the input dtype, so that float16 inputs do not lose
    # precision in the sum. The accumulator is the only state carried across the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for first_k in range(0, K, BLOCK_K):
        k_range = first_k + tl.arange(0, BLOCK_K)
        a_k = k_range[None, :]
        b_k = k_range[:, None]
        # Masked elements load as 0. A zero row or column adds nothing to the product, so a
        # partial tile needs no special case in the dot itself.
        a_tile = tl.load(
            a_ptr + rows * a_stride_m + a_k * a_stride_k,
            mask=(rows < M) & (a_k < K),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptr + b_k * b_stride_k + cols * b_stride_n,
            mask=(b_k < K) & (cols < N),
            other=0.0,
        )
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision=DOT_PRECISION)

    c_ptrs = c_ptr + rows * c_stride_m + cols * c_stride_n
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=(rows < M) & (cols < N))


def _launch_tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    dot_precision: DotPrecision,
    num_warps: int,
    num_stages: int,
) -> None:
    """Launch one program per tile of c."""
    m, k = a.shape
    n = b.size(1)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _tiled_matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        DOT_PRECISION=dot_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _validate_matmul_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    """Check that a and b can be multiplied together."""
    validate_input("a", a)
    validate_input("b", b)
    if a.dtype != b.dtype:
        raise TypeError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")
    if a.device != b.device:
        raise ValueError(f"a and b must be on the same device, got {a.device} and {b.device}")
    if a.size(1) != b.size(0):
        raise ValueError(
            f"a has {a.size(1)} columns but b has {b.size(0)} rows, "
            f"so shapes {tuple(a.shape)} and {tuple(b.shape)} cannot be multiplied"
        )


def tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    group_m: int = 1,
    dot_precision: DotPrecision = "ieee",
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute C = A @ B with a tiled Triton kernel.

    Args:
        a: A float32 or float16 matrix of shape (M, K). Any strides are accepted.
        b: A matrix of shape (K, N), with the same dtype as a. Any strides are accepted.
        out: Optional output of shape (M, N) with the same dtype. It may have any strides,
            but must not overlap a or b. A new tensor is allocated if this is None.
        block_m: Tile height. A power of two, which need not divide M.
        block_n: Tile width. A power of two, which need not divide N.
        block_k: Number of K values consumed per loop iteration. A power of two, which need
            not divide K, and at least 8 for float32 or 16 for float16.
        group_m: Number of tile rows in one group of programs. A power of two. See
            _program_to_tile for how this affects L2 reuse.
        dot_precision: "ieee" gives full float32 products. "tf32" truncates float32 inputs to
            10 mantissa bits, which lets the GPU use tensor cores at the cost of accuracy.
            It has no effect in the interpreter, and none on float16 inputs.
        num_warps: Number of warps that execute each program.
        num_stages: Number of software pipeline stages for the loads in the K loop.

    Returns:
        The output tensor, of shape (M, N) with the same dtype as a.

    Raises:
        TypeError: If a dtype is not supported or does not match.
        ValueError: If a shape, device, stride, block size or option is not valid.
    """
    _validate_matmul_inputs(a, b)
    for name, value in (
        ("block_m", block_m),
        ("block_n", block_n),
        ("block_k", block_k),
        ("group_m", group_m),
    ):
        validate_block_size(name, value)
    if block_k < _MIN_BLOCK_K[a.dtype]:
        raise ValueError(f"block_k must be at least {_MIN_BLOCK_K[a.dtype]} for {a.dtype}")
    if dot_precision not in ("ieee", "tf32"):
        raise ValueError(f"dot_precision must be 'ieee' or 'tf32', got {dot_precision!r}")

    shape = (a.size(0), b.size(1))
    if out is None:
        out = torch.empty(shape, dtype=a.dtype, device=a.device)
    else:
        validate_output(out, shape, like=a, inputs=(a, b))

    _launch_tiled_matmul(
        a, b, out, block_m, block_n, block_k, group_m, dot_precision, num_warps, num_stages
    )
    return out
