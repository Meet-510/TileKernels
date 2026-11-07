"""Check the softmax kernels against torch.softmax on seeded random inputs."""

import pytest
import torch

from kernel_testing import assert_close_for_dtype, seeded_randn
from tilekernels import fused_softmax, online_softmax
from tilekernels.utils import next_power_of_two

DTYPES = [torch.float32, torch.float16]

# Both exact powers of two and shapes on either side of one, so the tail mask gets exercised.
SHAPES = [(1, 1), (1, 7), (5, 1), (3, 8), (4, 16), (13, 7), (2, 31), (2, 33), (7, 127), (3, 129)]


def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute softmax in float32 and round to x's dtype, matching the kernel's accumulation."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_matches_pytorch(shape: tuple[int, int], dtype: torch.dtype, device: str) -> None:
    x = seeded_randn(shape, dtype, device)
    assert_close_for_dtype(fused_softmax(x), reference_softmax(x))


@pytest.mark.parametrize("scale", [2, 4, 16])
@pytest.mark.parametrize("n_cols", [3, 16, 17])
def test_fused_with_blocks_larger_than_the_row(n_cols: int, scale: int, device: str) -> None:
    x = seeded_randn((4, n_cols), torch.float32, device)
    block_size = next_power_of_two(n_cols) * scale
    assert_close_for_dtype(fused_softmax(x, block_size=block_size), reference_softmax(x))


@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_fused_num_warps_does_not_change_the_result(num_warps: int, device: str) -> None:
    x = seeded_randn((3, 100), torch.float32, device)
    assert_close_for_dtype(fused_softmax(x, num_warps=num_warps), reference_softmax(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_output_is_a_probability_distribution(dtype: torch.dtype, device: str) -> None:
    x = seeded_randn((6, 50), dtype, device)
    y = fused_softmax(x).float()
    assert (y >= 0).all()
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(6, device=device), atol=tolerance, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block_size", [1, 4, 16, 1024])
def test_online_matches_pytorch(
    shape: tuple[int, int], dtype: torch.dtype, block_size: int, device: str
) -> None:
    x = seeded_randn(shape, dtype, device)
    assert_close_for_dtype(online_softmax(x, block_size=block_size), reference_softmax(x))


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_online_options_do_not_change_the_result(
    num_warps: int, num_stages: int, device: str
) -> None:
    x = seeded_randn((3, 100), torch.float32, device)
    y = online_softmax(x, block_size=16, num_warps=num_warps, num_stages=num_stages)
    assert_close_for_dtype(y, reference_softmax(x))


@pytest.mark.slow
def test_fused_large_shape(device: str) -> None:
    x = seeded_randn((64, 2048), torch.float32, device)
    assert_close_for_dtype(fused_softmax(x), reference_softmax(x))


@pytest.mark.slow
def test_online_large_shape(device: str) -> None:
    x = seeded_randn((64, 2048), torch.float32, device)
    assert_close_for_dtype(online_softmax(x, block_size=32), reference_softmax(x))


class TestOnlineValidation:
    """Online softmax shares its checks with fused softmax, so these only cover its own paths."""

    @pytest.mark.parametrize("block_size", [0, 3, 12])
    def test_rejects_block_size_that_is_not_a_power_of_two(
        self, block_size: int, device: str
    ) -> None:
        with pytest.raises(ValueError, match="power of two"):
            online_softmax(torch.zeros(2, 4, device=device), block_size=block_size)

    def test_rejects_non_2d(self, device: str) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            online_softmax(torch.zeros(4, device=device))

    def test_writes_into_out(self, device: str) -> None:
        x = seeded_randn((2, 9), torch.float32, device)
        out = torch.zeros_like(x)
        assert online_softmax(x, out=out, block_size=4) is out
        assert_close_for_dtype(out, reference_softmax(x))

    def test_rejects_out_that_overlaps_the_input(self, device: str) -> None:
        x = torch.zeros(2, 4, device=device)
        with pytest.raises(ValueError, match="overlap"):
            online_softmax(x, out=x)


class TestFusedValidation:
    """The wrapper should reject bad inputs with a clear error, before launching anything."""

    def test_rejects_non_2d(self, device: str) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            fused_softmax(torch.zeros(4, device=device))

    def test_rejects_empty(self, device: str) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            fused_softmax(torch.zeros(0, 4, device=device))

    def test_rejects_integer_dtype(self, device: str) -> None:
        with pytest.raises(TypeError, match="float32 or float16"):
            fused_softmax(torch.zeros(2, 4, dtype=torch.int32, device=device))

    def test_rejects_requires_grad(self, device: str) -> None:
        with pytest.raises(ValueError, match="no backward pass"):
            fused_softmax(torch.zeros(2, 4, device=device, requires_grad=True))

    @pytest.mark.parametrize("block_size", [0, 3, 12, -8])
    def test_rejects_block_size_that_is_not_a_power_of_two(
        self, block_size: int, device: str
    ) -> None:
        with pytest.raises(ValueError, match="power of two"):
            fused_softmax(torch.zeros(2, 4, device=device), block_size=block_size)

    def test_rejects_block_size_smaller_than_the_row(self, device: str) -> None:
        with pytest.raises(ValueError, match="online_softmax"):
            fused_softmax(torch.zeros(2, 17, device=device), block_size=16)

    def test_rejects_block_size_above_the_triton_limit(self, device: str) -> None:
        with pytest.raises(ValueError, match="exceeds the limit"):
            fused_softmax(torch.zeros(2, 4, device=device), block_size=2**30)

    def test_rejects_out_with_the_wrong_shape(self, device: str) -> None:
        x = torch.zeros(2, 4, device=device)
        with pytest.raises(ValueError, match="must have shape"):
            fused_softmax(x, out=torch.zeros(4, 2, device=device))

    def test_rejects_out_with_the_wrong_dtype(self, device: str) -> None:
        x = torch.zeros(2, 4, device=device)
        with pytest.raises(TypeError, match="must have dtype"):
            fused_softmax(x, out=torch.zeros(2, 4, dtype=torch.float16, device=device))

    def test_rejects_out_that_overlaps_the_input(self, device: str) -> None:
        x = torch.zeros(2, 4, device=device)
        with pytest.raises(ValueError, match="overlap"):
            fused_softmax(x, out=x)

    def test_rejects_expanded_out(self, device: str) -> None:
        x = torch.zeros(2, 4, device=device)
        out = torch.zeros(1, 4, device=device).expand(2, 4)
        with pytest.raises(ValueError, match="zero stride"):
            fused_softmax(x, out=out)

    def test_accepts_out_in_a_disjoint_part_of_the_same_storage(self, device: str) -> None:
        storage = seeded_randn((4, 8), torch.float32, device)
        x, out = storage[:2], storage[2:]
        expected = reference_softmax(x)
        assert fused_softmax(x, out=out) is out
        assert_close_for_dtype(out, expected)
