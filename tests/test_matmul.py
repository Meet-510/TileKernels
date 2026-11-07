"""Check the tiled matmul kernel against torch.matmul on seeded random inputs."""

import pytest
import torch

from kernel_testing import assert_matmul_close, seeded_randn
from tilekernels import tiled_matmul

DTYPES = [torch.float32, torch.float16]


def reference_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply in float32 and round to a's dtype, matching the kernel's accumulation."""
    return torch.matmul(a.float(), b.float()).to(a.dtype)


def check_matmul(m: int, n: int, k: int, dtype: torch.dtype, device: str, **options: int) -> None:
    """Run the kernel on seeded (m, k) and (k, n) inputs, and compare with PyTorch."""
    a = seeded_randn((m, k), dtype, device, label="a")
    b = seeded_randn((k, n), dtype, device, label="b")
    c = tiled_matmul(a, b, **options)  # type: ignore[arg-type]  # Only int options are passed.
    assert_matmul_close(c, reference_matmul(a, b), n_terms=k)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 16, 16), (32, 16, 8), (16, 32, 32), (48, 32, 16)],
)
def test_exact_multiples_of_the_blocks(
    m: int, n: int, k: int, dtype: torch.dtype, device: str
) -> None:
    check_matmul(m, n, k, dtype, device, block_m=16, block_n=16, block_k=16)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 1, 1),
        (1, 7, 3),  # one row
        (7, 1, 3),  # one column
        (13, 7, 5),  # every dimension smaller than its block
        (17, 33, 9),  # one past a multiple of the block in every dimension
        (15, 31, 7),  # one short of a multiple of the block in every dimension
        (40, 23, 3),  # K much smaller than BLOCK_K
        (5, 6, 100),  # K much larger than M and N
    ],
)
def test_shapes_that_are_not_multiples_of_the_blocks(
    m: int, n: int, k: int, dtype: torch.dtype, device: str
) -> None:
    check_matmul(m, n, k, dtype, device, block_m=16, block_n=16, block_k=16)


@pytest.mark.parametrize(
    ("block_m", "block_n", "block_k"),
    [(1, 1, 8), (2, 4, 8), (4, 8, 16), (8, 8, 8), (16, 8, 32), (32, 32, 32), (64, 64, 64)],
)
def test_block_sizes_do_not_change_the_result(
    block_m: int, block_n: int, block_k: int, device: str
) -> None:
    check_matmul(
        21, 19, 11, torch.float32, device, block_m=block_m, block_n=block_n, block_k=block_k
    )


@pytest.mark.parametrize("k", [1, 2, 7, 8, 9])
def test_k_near_the_block_boundary(k: int, device: str) -> None:
    check_matmul(9, 9, k, torch.float32, device, block_m=8, block_n=8, block_k=8)


@pytest.mark.parametrize("group_m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    ("m", "n"),
    [(64, 64), (40, 24), (72, 8), (8, 72), (9, 9)],
)
def test_program_grouping_covers_every_tile_exactly_once(
    m: int, n: int, group_m: int, device: str
) -> None:
    # Grouping only changes which program computes which tile. A wrong mapping would leave some
    # tiles of the output unwritten and would visit others twice, so the result would change.
    # NaNs are written first, so a tile left unwritten cannot pass by accident.
    a = seeded_randn((m, 6), torch.float32, device, label="a")
    b = seeded_randn((6, n), torch.float32, device, label="b")
    out = torch.full((m, n), float("nan"), device=device)
    tiled_matmul(a, b, out=out, block_m=8, block_n=8, block_k=8, group_m=group_m)
    assert_matmul_close(out, reference_matmul(a, b), n_terms=6)


@pytest.mark.parametrize("num_warps", [1, 4])
@pytest.mark.parametrize("num_stages", [1, 3, 4])
def test_scheduling_options_do_not_change_the_result(
    num_warps: int, num_stages: int, device: str
) -> None:
    check_matmul(
        20, 20, 20, torch.float32, device, block_m=16, block_n=16, block_k=8,
        num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip


def test_tf32_stays_within_its_own_accuracy(device: str) -> None:
    # TF32 keeps 10 mantissa bits, so on a GPU its relative error is around 1e-3. The interpreter
    # ignores input_precision, so there this test only proves that the option is accepted.
    a = seeded_randn((16, 16), torch.float32, device, label="a")
    b = seeded_randn((16, 16), torch.float32, device, label="b")
    c = tiled_matmul(a, b, block_m=16, block_n=16, block_k=16, dot_precision="tf32")
    torch.testing.assert_close(c, torch.matmul(a, b), rtol=5e-2, atol=5e-2)


def test_float16_output_matches_the_input_dtype(device: str) -> None:
    a = seeded_randn((4, 4), torch.float16, device)
    assert tiled_matmul(a, a.clone()).dtype == torch.float16


def test_float16_inputs_accumulate_in_float32(device: str) -> None:
    # The sum is 30 * 30 + 63 * 0.01, about 900.63. Spacing between float16 values near 900 is
    # 0.5, so a float16 accumulator would round away each 0.01 and stay at exactly 900. In
    # float32 the small terms survive, and the final rounding to float16 gives 900.5.
    k = 64
    a = torch.full((1, k), 1.0, dtype=torch.float16, device=device)
    b = torch.full((k, 1), 1e-2, dtype=torch.float16, device=device)
    a[0, 0] = b[0, 0] = 30.0
    c = tiled_matmul(a, b, block_m=1, block_n=1, block_k=16)
    assert c.item() == 900.5


def test_identity_matrix_returns_the_other_input(device: str) -> None:
    a = seeded_randn((11, 11), torch.float32, device)
    identity = torch.eye(11, device=device)
    torch.testing.assert_close(tiled_matmul(a, identity, block_m=4, block_n=4, block_k=8), a)


def test_writes_into_out(device: str) -> None:
    a = seeded_randn((5, 6), torch.float32, device, label="a")
    b = seeded_randn((6, 7), torch.float32, device, label="b")
    out = torch.zeros(5, 7, device=device)
    assert tiled_matmul(a, b, out=out) is out
    assert_matmul_close(out, reference_matmul(a, b), n_terms=6)


@pytest.mark.slow
@pytest.mark.parametrize("group_m", [1, 4])
def test_large_shape_that_is_not_a_multiple(group_m: int, device: str) -> None:
    check_matmul(
        127, 129, 65, torch.float32, device, block_m=16, block_n=16, block_k=16, group_m=group_m
    )


@pytest.mark.slow
def test_large_float16_shape(device: str) -> None:
    check_matmul(127, 129, 65, torch.float16, device, block_m=32, block_n=32, block_k=16)


class TestValidation:
    """The wrapper should reject bad inputs with a clear error, before launching anything."""

    def test_rejects_incompatible_shapes(self, device: str) -> None:
        a, b = torch.zeros(2, 3, device=device), torch.zeros(4, 5, device=device)
        with pytest.raises(ValueError, match="cannot be multiplied"):
            tiled_matmul(a, b)

    def test_rejects_non_2d(self, device: str) -> None:
        with pytest.raises(ValueError, match="b must be 2D"):
            tiled_matmul(torch.zeros(2, 3, device=device), torch.zeros(3, device=device))

    def test_rejects_mixed_dtypes(self, device: str) -> None:
        a = torch.zeros(2, 3, device=device)
        b = torch.zeros(3, 2, dtype=torch.float16, device=device)
        with pytest.raises(TypeError, match="same dtype"):
            tiled_matmul(a, b)

    def test_rejects_float64(self, device: str) -> None:
        a = torch.zeros(2, 2, dtype=torch.float64, device=device)
        with pytest.raises(TypeError, match="float32 or float16"):
            tiled_matmul(a, a.clone())

    @pytest.mark.parametrize("option", ["block_m", "block_n", "block_k", "group_m"])
    def test_rejects_sizes_that_are_not_powers_of_two(self, option: str, device: str) -> None:
        a = torch.zeros(4, 4, device=device)
        with pytest.raises(ValueError, match=f"{option} must be a power of two"):
            tiled_matmul(a, a.clone(), **{option: 3})  # type: ignore[arg-type]  # Value is an int.

    @pytest.mark.parametrize(
        ("dtype", "block_k", "minimum"), [(torch.float32, 4, 8), (torch.float16, 8, 16)]
    )
    def test_rejects_block_k_that_cannot_compile(
        self, dtype: torch.dtype, block_k: int, minimum: int, device: str
    ) -> None:
        a = torch.zeros(4, 4, dtype=dtype, device=device)
        with pytest.raises(ValueError, match=f"block_k must be at least {minimum}"):
            tiled_matmul(a, a.clone(), block_k=block_k)

    def test_rejects_an_unknown_precision(self, device: str) -> None:
        a = torch.zeros(4, 4, device=device)
        with pytest.raises(ValueError, match="dot_precision"):
            tiled_matmul(a, a.clone(), dot_precision="bad")  # type: ignore[arg-type]  # On purpose.

    def test_rejects_out_with_the_wrong_shape(self, device: str) -> None:
        a = torch.zeros(2, 3, device=device)
        b = torch.zeros(3, 5, device=device)
        with pytest.raises(ValueError, match="must have shape"):
            tiled_matmul(a, b, out=torch.zeros(2, 3, device=device))

    @pytest.mark.parametrize("overlaps", ["a", "b"])
    def test_rejects_out_that_overlaps_an_input(self, overlaps: str, device: str) -> None:
        # C is written tile by tile while later programs are still reading A and B.
        a = torch.zeros(4, 4, device=device)
        b = torch.zeros(4, 4, device=device)
        with pytest.raises(ValueError, match="overlap"):
            tiled_matmul(a, b, out=a if overlaps == "a" else b)
