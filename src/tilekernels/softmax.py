"""Row-wise softmax kernels, along the last dimension of a 2D tensor.

Why fuse? The unfused PyTorch pipeline runs max, subtract, exp, sum and divide as separate
GPU kernels, and each one reads from and writes to global memory. This kernel does all of
it in one program, with a single read and a single write per row. Everything in between
stays in registers.
"""

import torch
import triton
import triton.language as tl

from tilekernels.utils import (
    next_power_of_two,
    validate_block_size,
    validate_input,
    validate_output,
)

# Kernel arguments stay unannotated, except for tl.constexpr, because Triton's frontend parses
# them itself: pointers and integers get their types from the launch arguments. That is the
# reason for each targeted `type: ignore[no-untyped-def]` in this module.


@triton.jit
def _fused_softmax_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    y_ptr,
    n_cols,
    x_row_stride,
    x_col_stride,
    y_row_stride,
    y_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the softmax of one row. The grid launches one program per row.

    BLOCK_SIZE is a power of two that is at least n_cols, so the whole row fits in a
    single block. Columns past n_cols are masked and load as -inf. They then become
    exp(-inf) = 0, so they cannot affect the max or the sum.
    """
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    in_row = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols * x_col_stride, mask=in_row, other=-float("inf"))
    # Accumulate in float32 so that a float16 row does not lose precision in the sum.
    x = x.to(tl.float32)

    # Subtracting the row max first keeps exp() in range. The largest exponent is exactly 0,
    # so exp() cannot overflow to inf, and the sum is between 1 and n_cols.
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)

    y_ptrs = y_ptr + row * y_row_stride + cols * y_col_stride
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=in_row)


@triton.jit
def _online_softmax_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    y_ptr,
    n_cols,
    x_row_stride,
    x_col_stride,
    y_row_stride,
    y_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the softmax of one row that may be wider than one block.

    One block cannot hold the row, so the row is folded over column blocks. After block i,
    with m_i the running max and l_i the running sum of exp(x - m_i) over the columns seen:

        m_i = max(m_{i-1}, max(block_i))
        l_i = l_{i-1} * exp(m_{i-1} - m_i) + sum(exp(block_i - m_i))

    The first term re-expresses the old sum relative to the new max. Its exponent is never
    positive and every term of l_i is at most 1, so l_i stays between 0 and n_cols. That
    is the same stability property as the single block version.

    The result needs the final m and l, so a second pass over the row computes
    exp(x - m) / l. This costs a second read of x but still only one write of y.
    """
    row = tl.program_id(axis=0)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    for first_col in range(0, n_cols, BLOCK_SIZE):
        cols = first_col + tl.arange(0, BLOCK_SIZE)
        in_row = cols < n_cols
        x = tl.load(x_row + cols * x_col_stride, mask=in_row, other=-float("inf")).to(tl.float32)

        new_max = tl.maximum(running_max, tl.max(x, axis=0))
        # While only -inf has been seen, new_max is -inf too and the exponents below would be
        # -inf - (-inf) = NaN. A -inf column must add nothing to the sum rather than poison it,
        # because the row may still hold finite values in a later block. Both guards are
        # written so that a real NaN in x still propagates, as it does in torch.softmax.
        rescale = tl.where(new_max > running_max, tl.exp(running_max - new_max), 1.0)
        block_terms = tl.where(x == -float("inf"), 0.0, tl.exp(x - new_max))
        running_sum = running_sum * rescale + tl.sum(block_terms, axis=0)
        running_max = new_max

    for first_col in range(0, n_cols, BLOCK_SIZE):
        cols = first_col + tl.arange(0, BLOCK_SIZE)
        in_row = cols < n_cols
        x = tl.load(x_row + cols * x_col_stride, mask=in_row, other=-float("inf")).to(tl.float32)
        y = tl.exp(x - running_max) / running_sum
        tl.store(y_row + cols * y_col_stride, y.to(y_ptr.dtype.element_ty), mask=in_row)


def _launch_fused_softmax(
    x: torch.Tensor, y: torch.Tensor, block_size: int, num_warps: int
) -> None:
    """Launch one fused softmax program per row of x, writing into y."""
    n_rows, n_cols = x.shape
    _fused_softmax_kernel[(n_rows,)](
        x,
        y,
        n_cols,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def _launch_online_softmax(
    x: torch.Tensor, y: torch.Tensor, block_size: int, num_warps: int, num_stages: int
) -> None:
    """Launch one online softmax program per row of x, writing into y."""
    n_rows, n_cols = x.shape
    _online_softmax_kernel[(n_rows,)](
        x,
        y,
        n_cols,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def fused_softmax(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_size: int | None = None,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute softmax along the last dimension of a 2D tensor, in one fused kernel.

    Args:
        x: A 2D float32 or float16 tensor. Any strides are accepted.
        out: Optional output tensor with the same shape and dtype as x. It may have any
            strides, but must not overlap x. A new tensor is allocated if this is None.
        block_size: Column block size. It must be a power of two and at least n_cols, and
            defaults to the smallest valid value. Use online_softmax for rows wider than
            the largest allowed block.
        num_warps: Number of warps that execute each program.

    Returns:
        The output tensor. It has the same shape and dtype as x.

    Raises:
        TypeError: If a dtype is not supported or does not match.
        ValueError: If a shape, device, stride or block size is not valid.
    """
    validate_input("x", x)
    n_cols = x.size(1)
    block_size = next_power_of_two(n_cols) if block_size is None else block_size
    validate_block_size("block_size", block_size)
    if block_size < n_cols:
        raise ValueError(
            f"block_size={block_size} is smaller than n_cols={n_cols}. "
            "Use a larger block or online_softmax."
        )

    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    else:
        validate_output(out, tuple(x.shape), like=x, inputs=(x,))

    _launch_fused_softmax(x, out, block_size, num_warps)
    return out


def online_softmax(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute softmax along the last dimension of a 2D tensor, for rows of any width.

    The row is processed in column blocks with a running max and a running sum. See
    _online_softmax_kernel for the update rule, and docs/KERNEL_NOTES.md for its derivation.

    Args:
        x: A 2D float32 or float16 tensor. Any strides are accepted.
        out: Optional output tensor with the same shape and dtype as x. It may have any
            strides, but must not overlap x. A new tensor is allocated if this is None.
        block_size: Column block size. It must be a power of two, but unlike with
            fused_softmax it may be smaller than n_cols.
        num_warps: Number of warps that execute each program.
        num_stages: Number of software pipeline stages for the load in each loop.

    Returns:
        The output tensor. It has the same shape and dtype as x.

    Raises:
        TypeError: If a dtype is not supported or does not match.
        ValueError: If a shape, device, stride or block size is not valid.
    """
    validate_input("x", x)
    validate_block_size("block_size", block_size)

    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    else:
        validate_output(out, tuple(x.shape), like=x, inputs=(x,))

    _launch_online_softmax(x, out, block_size, num_warps, num_stages)
    return out
