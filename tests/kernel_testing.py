"""Helpers shared by the kernel tests: seeded inputs and tolerances per dtype."""

import zlib
from typing import NamedTuple

import torch


class Tolerance(NamedTuple):
    """Tolerances passed to torch.testing.assert_close."""

    rtol: float
    atol: float


# float32: machine epsilon is about 1.2e-7. A kernel chains a few operations, and the libraries
# on each side use different exp() implementations, so we allow roughly 100 epsilons of
# relative error. The absolute term only matters for values that are nearly zero.
#
# float16: machine epsilon is about 1e-3. Both sides are computed in float32 and then rounded
# to float16, so they can differ by about one unit in the last place, hence 2e-3.
TOLERANCES = {
    torch.float32: Tolerance(rtol=1e-5, atol=1e-6),
    torch.float16: Tolerance(rtol=2e-3, atol=1e-4),
}


def seeded_randn(
    shape: tuple[int, ...], dtype: torch.dtype, device: str, *, label: str = ""
) -> torch.Tensor:
    """Return normal random values, seeded from the arguments so that every run is identical."""
    seed = zlib.crc32(f"{label}{shape}{dtype}".encode())
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


def assert_close_for_dtype(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare two tensors with the tolerance for their dtype. NaNs must match position."""
    assert actual.dtype == expected.dtype
    rtol, atol = TOLERANCES[actual.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def assert_matmul_close(actual: torch.Tensor, expected: torch.Tensor, n_terms: int) -> None:
    """Compare two matmul results, allowing for rounding in a sum of n_terms products.

    A relative tolerance alone is not enough here. When the terms of a sum cancel, its result
    is near zero while its rounding error is still proportional to the terms themselves. Two
    libraries that add in a different order can then disagree by a large relative amount, so
    the absolute tolerance grows with n_terms, in line with the standard floating point bound
    on summation. The factor of 4 covers the size of products of standard normal inputs.
    """
    assert actual.dtype == expected.dtype
    rtol, atol = TOLERANCES[actual.dtype]
    if actual.dtype == torch.float32:
        atol = 4 * n_terms * torch.finfo(torch.float32).eps
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
