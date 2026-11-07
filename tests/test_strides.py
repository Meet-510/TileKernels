"""Non-contiguous tensors: the kernels must follow strides for every input and output."""

from collections.abc import Callable

import pytest
import torch

from kernel_testing import assert_close_for_dtype, assert_matmul_close, seeded_randn
from tilekernels import fused_softmax, online_softmax, tiled_matmul

DTYPES = [torch.float32, torch.float16]

SoftmaxKernel = Callable[..., torch.Tensor]
SOFTMAX_KERNELS = [
    pytest.param(fused_softmax, id="fused"),
    pytest.param(lambda x, **kw: online_softmax(x, block_size=4, **kw), id="online"),
]


def _transpose(x: torch.Tensor) -> torch.Tensor:
    return x.t()


def _every_other_column(x: torch.Tensor) -> torch.Tensor:
    return x[:, ::2]


def _every_other_row(x: torch.Tensor) -> torch.Tensor:
    return x[::2, :]


def _every_third_row_and_column(x: torch.Tensor) -> torch.Tensor:
    return x[::3, ::3]


def _inner_slice(x: torch.Tensor) -> torch.Tensor:
    """A slice with a nonzero storage offset, so the data pointer is not the storage start."""
    return x[2:-1, 3:-2]


VIEWS = [
    _transpose,
    _every_other_column,
    _every_other_row,
    _every_third_row_and_column,
    _inner_slice,
]


def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def reference_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.float(), b.float()).to(a.dtype)


@pytest.mark.parametrize("kernel", SOFTMAX_KERNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("view", VIEWS)
class TestSoftmaxInputViews:
    def test_input_view_matches_pytorch(
        self,
        kernel: SoftmaxKernel,
        dtype: torch.dtype,
        view: Callable[[torch.Tensor], torch.Tensor],
        device: str,
    ) -> None:
        x = view(seeded_randn((13, 15), dtype, device))
        assert not x.is_contiguous()
        assert_close_for_dtype(kernel(x), reference_softmax(x))


@pytest.mark.parametrize("kernel", SOFTMAX_KERNELS)
class TestSoftmaxStrides:
    def test_expanded_input(self, kernel: SoftmaxKernel, device: str) -> None:
        # Expanding gives a row stride of 0, so every program reads the same row of memory.
        x = seeded_randn((1, 9), torch.float32, device).expand(4, 9)
        assert_close_for_dtype(kernel(x), reference_softmax(x))

    def test_strided_output_leaves_the_gaps_untouched(
        self, kernel: SoftmaxKernel, device: str
    ) -> None:
        x = seeded_randn((3, 5), torch.float32, device)
        parent = torch.full((6, 10), -1.0, device=device)
        out = parent[::2, ::2]
        assert kernel(x, out=out) is out
        assert_close_for_dtype(out, reference_softmax(x))
        # Every element that out does not cover must still hold the fill value.
        covered = torch.zeros_like(parent, dtype=torch.bool)
        covered[::2, ::2] = True
        assert (parent[~covered] == -1.0).all()

    def test_transposed_input_and_output_together(
        self, kernel: SoftmaxKernel, device: str
    ) -> None:
        x = seeded_randn((7, 6), torch.float32, device).t()
        out = torch.zeros(7, 6, device=device).t()
        assert_close_for_dtype(kernel(x, out=out), reference_softmax(x))

    def test_float16_strided_output(self, kernel: SoftmaxKernel, device: str) -> None:
        # Pointer arithmetic scales with the element size, which a float32 test cannot show.
        x = seeded_randn((3, 5), torch.float16, device)
        out = torch.zeros(6, 10, dtype=torch.float16, device=device)[::2, ::2]
        assert_close_for_dtype(kernel(x, out=out), reference_softmax(x))


@pytest.mark.parametrize("dtype", DTYPES)
class TestMatmulStrides:
    def test_transposed_a(self, dtype: torch.dtype, device: str) -> None:
        a = seeded_randn((9, 13), dtype, device, label="a").t()
        b = seeded_randn((9, 11), dtype, device, label="b")
        c = tiled_matmul(a, b, block_m=8, block_n=8, block_k=16)
        assert_matmul_close(c, reference_matmul(a, b), n_terms=9)

    def test_transposed_b(self, dtype: torch.dtype, device: str) -> None:
        a = seeded_randn((13, 9), dtype, device, label="a")
        b = seeded_randn((11, 9), dtype, device, label="b").t()
        c = tiled_matmul(a, b, block_m=8, block_n=8, block_k=16)
        assert_matmul_close(c, reference_matmul(a, b), n_terms=9)

    def test_both_transposed(self, dtype: torch.dtype, device: str) -> None:
        a = seeded_randn((9, 13), dtype, device, label="a").t()
        b = seeded_randn((11, 9), dtype, device, label="b").t()
        c = tiled_matmul(a, b, block_m=8, block_n=8, block_k=16)
        assert_matmul_close(c, reference_matmul(a, b), n_terms=9)

    def test_sliced_inputs_with_storage_offsets(self, dtype: torch.dtype, device: str) -> None:
        a = seeded_randn((20, 30), dtype, device, label="a")[1:18:2, 3:24:3]
        b = seeded_randn((30, 20), dtype, device, label="b")[2:9, 1:20:4]
        c = tiled_matmul(a, b, block_m=8, block_n=8, block_k=16)
        assert_matmul_close(c, reference_matmul(a, b), n_terms=7)

    def test_strided_output_leaves_the_gaps_untouched(
        self, dtype: torch.dtype, device: str
    ) -> None:
        a = seeded_randn((11, 7), dtype, device, label="a")
        b = seeded_randn((7, 9), dtype, device, label="b")
        parent = torch.full((22, 27), -1.0, dtype=dtype, device=device)
        out = parent[::2, ::3]
        assert tiled_matmul(a, b, out=out, block_m=8, block_n=8, block_k=16) is out
        assert_matmul_close(out, reference_matmul(a, b), n_terms=7)
        covered = torch.zeros_like(parent, dtype=torch.bool)
        covered[::2, ::3] = True
        assert (parent[~covered] == -1.0).all()

    def test_transposed_output(self, dtype: torch.dtype, device: str) -> None:
        a = seeded_randn((11, 7), dtype, device, label="a")
        b = seeded_randn((7, 9), dtype, device, label="b")
        out = torch.zeros(9, 11, dtype=dtype, device=device).t()
        tiled_matmul(a, b, out=out, block_m=8, block_n=8, block_k=16)
        assert_matmul_close(out, reference_matmul(a, b), n_terms=7)

    def test_expanded_input(self, dtype: torch.dtype, device: str) -> None:
        a = seeded_randn((1, 7), dtype, device, label="a").expand(5, 7)
        b = seeded_randn((7, 6), dtype, device, label="b")
        c = tiled_matmul(a, b, block_m=4, block_n=4, block_k=16)
        assert_matmul_close(c, reference_matmul(a, b), n_terms=7)


def test_matmul_accepts_the_same_tensor_as_both_inputs(device: str) -> None:
    # Reading the same memory twice is safe. Only overlap with the output is rejected.
    a = seeded_randn((9, 9), torch.float32, device)
    c = tiled_matmul(a, a, block_m=8, block_n=8, block_k=8)
    assert_matmul_close(c, reference_matmul(a, a), n_terms=9)
