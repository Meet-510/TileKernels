"""Edge cases that both softmax kernels must handle exactly like torch.softmax."""

from collections.abc import Callable

import pytest
import torch

from kernel_testing import assert_close_for_dtype, seeded_randn
from tilekernels import fused_softmax, online_softmax

INF = float("inf")
NAN = float("nan")


def _fused(x: torch.Tensor) -> torch.Tensor:
    return fused_softmax(x)


def _online_small_blocks(x: torch.Tensor) -> torch.Tensor:
    """A block of 2 makes almost every row span several blocks, which stresses the update rule."""
    return online_softmax(x, block_size=2)


def _online_default(x: torch.Tensor) -> torch.Tensor:
    return online_softmax(x)


KERNELS = [
    pytest.param(_fused, id="fused"),
    pytest.param(_online_small_blocks, id="online_block2"),
    pytest.param(_online_default, id="online_default"),
]
Kernel = Callable[[torch.Tensor], torch.Tensor]


def check_against_pytorch(kernel: Kernel, row: list[float], device: str) -> None:
    """Run the kernel on one row and compare with float32 torch.softmax."""
    x = torch.tensor([row], dtype=torch.float32, device=device)
    assert_close_for_dtype(kernel(x), torch.softmax(x, dim=-1))


@pytest.mark.parametrize("kernel", KERNELS)
class TestSpecialRows:
    def test_very_large_values_do_not_overflow(self, kernel: Kernel, device: str) -> None:
        # A naive exp(x) would overflow to inf here, and the row would turn into NaN.
        check_against_pytorch(kernel, [1000.0, 1001.0, 999.0, 1000.5], device)

    def test_very_negative_values_do_not_underflow(self, kernel: Kernel, device: str) -> None:
        # A naive exp(x) would underflow to 0 here, and the row would divide 0 by 0.
        check_against_pytorch(kernel, [-1000.0, -1001.0, -999.0, -1000.5], device)

    def test_mixed_extreme_magnitudes(self, kernel: Kernel, device: str) -> None:
        check_against_pytorch(kernel, [1e30, -1e30, 0.0, 1e-30], device)

    def test_identical_values_become_uniform(self, kernel: Kernel, device: str) -> None:
        x = torch.full((2, 7), 3.5, device=device)
        expected = torch.full((2, 7), 1 / 7, device=device)
        assert_close_for_dtype(kernel(x), expected)

    def test_row_containing_negative_infinity(self, kernel: Kernel, device: str) -> None:
        check_against_pytorch(kernel, [1.0, -INF, 2.0], device)

    def test_negative_infinity_in_a_whole_leading_block(self, kernel: Kernel, device: str) -> None:
        # With block_size=2, the first blocks are entirely -inf. This is the case that the NaN
        # guards in the online kernel exist for: the finite values only appear later.
        check_against_pytorch(kernel, [-INF, -INF, -INF, -INF, 1.0, 2.0, -INF], device)

    def test_negative_infinity_in_a_whole_trailing_block(
        self, kernel: Kernel, device: str
    ) -> None:
        check_against_pytorch(kernel, [1.0, 2.0, -INF, -INF, -INF], device)

    def test_all_negative_infinity_is_nan(self, kernel: Kernel, device: str) -> None:
        check_against_pytorch(kernel, [-INF, -INF, -INF], device)

    def test_positive_infinity_is_nan(self, kernel: Kernel, device: str) -> None:
        check_against_pytorch(kernel, [1.0, INF, 2.0], device)

    def test_nan_in_the_first_block_spreads_to_the_row(self, kernel: Kernel, device: str) -> None:
        check_against_pytorch(kernel, [NAN, 1.0, 2.0, 3.0, 4.0], device)

    def test_nan_in_the_last_block_spreads_to_the_row(self, kernel: Kernel, device: str) -> None:
        check_against_pytorch(kernel, [1.0, 2.0, 3.0, 4.0, NAN], device)

    def test_single_column_is_always_one(self, kernel: Kernel, device: str) -> None:
        x = seeded_randn((5, 1), torch.float32, device)
        assert_close_for_dtype(kernel(x), torch.ones(5, 1, device=device))

    def test_special_rows_do_not_affect_their_neighbours(
        self, kernel: Kernel, device: str
    ) -> None:
        # Each row is its own program, so a NaN row must stay contained.
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [NAN, 2.0, 3.0], [-INF, -INF, -INF], [1.0, -INF, 3.0]],
            device=device,
        )
        assert_close_for_dtype(kernel(x), torch.softmax(x, dim=-1))


class TestOnlineUpdateRule:
    """Rows chosen to exercise the running max and running sum in the online kernel."""

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
    def test_max_moves_in_every_block(self, block_size: int, device: str) -> None:
        # With an ascending row, each block has a new max and the running sum must be rescaled.
        x = torch.arange(100, dtype=torch.float32, device=device).reshape(2, 50) / 5
        assert_close_for_dtype(online_softmax(x, block_size=block_size), torch.softmax(x, -1))

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
    def test_max_stays_in_the_first_block(self, block_size: int, device: str) -> None:
        x = torch.arange(100, 0, -1, dtype=torch.float32, device=device).reshape(2, 50) / 5
        assert_close_for_dtype(online_softmax(x, block_size=block_size), torch.softmax(x, -1))

    @pytest.mark.parametrize("n_cols", [1, 7, 8, 9, 16, 17, 100])
    def test_row_wider_than_one_block(self, n_cols: int, device: str) -> None:
        x = seeded_randn((3, n_cols), torch.float32, device)
        assert_close_for_dtype(online_softmax(x, block_size=8), torch.softmax(x, -1))

    def test_matches_the_fused_kernel_when_both_apply(self, device: str) -> None:
        x = seeded_randn((4, 64), torch.float32, device)
        assert_close_for_dtype(online_softmax(x, block_size=8), fused_softmax(x))

    def test_large_magnitude_jump_between_blocks(self, device: str) -> None:
        # exp(m_old - m_new) underflows to exactly 0, so the old sum must be discarded cleanly.
        check_against_pytorch(
            lambda x: online_softmax(x, block_size=2), [-1000.0, -999.0, 1000.0, 999.0], device
        )
