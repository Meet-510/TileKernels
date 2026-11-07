"""Smoke test: the smallest kernel that proves the toolchain works end to end."""

import pytest
import torch
import triton
import triton.language as tl

from tilekernels.utils import is_interpreter_mode


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):  # type: ignore[no-untyped-def]  # Triton parses kernel arguments itself, so they stay unannotated.
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_execution_mode_matches_flag(request: pytest.FixtureRequest) -> None:
    assert is_interpreter_mode() != request.config.getoption("--gpu")


@pytest.mark.parametrize("n_elements", [1, 15, 16, 17, 100])
def test_vector_add_matches_pytorch(n_elements: int, device: str) -> None:
    generator = torch.Generator(device).manual_seed(n_elements)
    x = torch.rand(n_elements, device=device, generator=generator)
    y = torch.rand(n_elements, device=device, generator=generator)
    out = torch.empty_like(x)

    block_size = 16
    grid = (triton.cdiv(n_elements, block_size),)
    _vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)

    torch.testing.assert_close(out, x + y)
