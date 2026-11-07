"""Do the masking tests mean anything? Break a mask on purpose and show that they notice.

Each mutant below is a copy of a real kernel with one mask changed, marked with MUTATION. A
test that compares it against PyTorch must then fail, while the real kernel passes on exactly
the same input. If a mutant ever survived, the masking tests would be proving nothing.

Mutants can read or write out of bounds, which would corrupt memory if nothing was there. So
each test places its tensors inside a larger parent, which keeps those accesses inside a
real allocation.
"""

import pytest
import torch
import triton
import triton.language as tl

from kernel_testing import assert_close_for_dtype, assert_matmul_close, seeded_randn
from tilekernels import fused_softmax, tiled_matmul


@triton.jit
def _softmax_with_zero_padding(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    row_ptr_offset = tl.program_id(axis=0) * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    in_row = cols < n_cols
    # MUTATION: the real kernel loads -inf here. Zero is a plausible padding value, but each
    # padded column then adds exp(0 - max) to the denominator and can even become the max.
    x = tl.load(x_ptr + row_ptr_offset + cols, mask=in_row, other=0.0)
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(y_ptr + row_ptr_offset + cols, y, mask=in_row)


@triton.jit
def _softmax_without_load_mask(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    row_ptr_offset = tl.program_id(axis=0) * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    # MUTATION: the load has no mask, so it also reads the columns past n_cols.
    x = tl.load(x_ptr + row_ptr_offset + cols)
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(y_ptr + row_ptr_offset + cols, y, mask=cols < n_cols)


@triton.jit
def _softmax_without_store_mask(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    row_ptr_offset = tl.program_id(axis=0) * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row_ptr_offset + cols, mask=cols < n_cols, other=-float("inf"))
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    # MUTATION: the store has no mask, so it also writes the columns past n_cols.
    tl.store(y_ptr + row_ptr_offset + cols, y)


@triton.jit
def _matmul_without_k_mask(a_ptr, b_ptr, c_ptr, K, row_stride, BLOCK_K: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments.
    """One tile, BLOCK_K x BLOCK_K, in a single program, to keep the mutant small."""
    rows = tl.arange(0, BLOCK_K)[:, None]
    cols = tl.arange(0, BLOCK_K)[None, :]
    accumulator = tl.zeros((BLOCK_K, BLOCK_K), tl.float32)
    for first_k in range(0, K, BLOCK_K):
        a_k = first_k + cols
        b_k = first_k + rows
        # MUTATION: neither load masks K, so the last iteration reads past the end of K.
        a = tl.load(a_ptr + rows * row_stride + a_k)
        b = tl.load(b_ptr + b_k * row_stride + cols)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    tl.store(c_ptr + rows * row_stride + cols, accumulator)


def _x_in_padded_parent(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x (3 rows, 5 columns) as a view of a parent whose rows hold 8 columns.

    The three extra columns are large positive values. Any kernel that reads them gets a
    visibly wrong max and sum.
    """
    parent = seeded_randn((3, 8), torch.float32, device)
    parent[:, 5:] = 50.0
    return parent, parent[:, :5]


@pytest.mark.parametrize("mutant", [_softmax_with_zero_padding, _softmax_without_load_mask])
def test_softmax_load_mutants_are_caught(mutant: object, device: str) -> None:
    parent, x = _x_in_padded_parent(device)
    expected = torch.softmax(x, dim=-1)

    # The real kernel passes on this exact input, so a failure below is due to the mutation.
    assert_close_for_dtype(fused_softmax(x, block_size=8), expected)

    out = torch.zeros(3, 8, device=device)
    mutant[(3,)](parent, out, 5, BLOCK_SIZE=8)  # type: ignore[index]  # A Triton kernel.
    with pytest.raises(AssertionError):
        assert_close_for_dtype(out[:, :5], expected)


def test_softmax_store_mutant_is_caught_by_the_gap_check(device: str) -> None:
    parent, x = _x_in_padded_parent(device)
    out = torch.full((3, 8), -1.0, device=device)
    _softmax_without_store_mask[(3,)](parent, out, 5, BLOCK_SIZE=8)

    # The values inside the shape are still right, so comparing them alone would pass.
    assert_close_for_dtype(out[:, :5], torch.softmax(x, dim=-1))
    # Only checking the elements around the output catches this bug, which is the reason the
    # stride tests in test_strides.py check that gaps stay untouched.
    assert not (out[:, 5:] == -1.0).all()


def test_matmul_k_mask_mutant_is_caught(device: str) -> None:
    block_k = 8
    k = 5
    # Both inputs sit inside 8 x 8 parents. The unused K columns hold large values.
    a_parent = seeded_randn((8, 8), torch.float32, device, label="a")
    b_parent = seeded_randn((8, 8), torch.float32, device, label="b")
    a_parent[:, k:] = 50.0
    b_parent[k:, :] = 50.0
    a, b = a_parent[:, :k], b_parent[:k, :]
    expected = torch.matmul(a, b)

    # The real kernel passes on this exact input, so a failure below is due to the mutation.
    assert_matmul_close(
        tiled_matmul(a, b, block_m=block_k, block_n=block_k, block_k=block_k), expected, k
    )

    out = torch.zeros(8, 8, device=device)
    _matmul_without_k_mask[(1,)](a_parent, b_parent, out, k, 8, BLOCK_K=block_k)
    with pytest.raises(AssertionError):
        assert_matmul_close(out, expected, k)
