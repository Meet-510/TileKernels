"""Unit tests for the validation paths that the kernel tests do not reach on their own."""

import pytest
import torch

from tilekernels import fused_softmax, tiled_matmul, utils


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (127, 128), (128, 128), (129, 256)],
)
def test_next_power_of_two(value: int, expected: int) -> None:
    assert utils.next_power_of_two(value) == expected


@pytest.mark.parametrize("value", [0, -1])
def test_next_power_of_two_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        utils.next_power_of_two(value)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes"])
def test_interpreter_mode_recognises_enabled_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRITON_INTERPRET", value)
    assert utils.is_interpreter_mode()


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_interpreter_mode_recognises_disabled_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRITON_INTERPRET", value)
    assert not utils.is_interpreter_mode()


def test_compiled_mode_rejects_a_cpu_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the validation runs here: it raises before anything reaches the GPU runtime.
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    with pytest.raises(ValueError, match="need a CUDA tensor"):
        fused_softmax(torch.zeros(2, 4, device="cpu"))


class TestDeviceMismatch:
    """Meta tensors stand in for a second device, so these tests need no GPU."""

    @pytest.fixture(autouse=True)
    def _interpreter_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Validation in compiled mode would reject the meta tensors for the wrong reason.
        monkeypatch.setenv("TRITON_INTERPRET", "1")

    def test_softmax_output_on_another_device(self) -> None:
        with pytest.raises(ValueError, match="out must be on"):
            fused_softmax(torch.zeros(2, 4), out=torch.zeros(2, 4, device="meta"))

    def test_matmul_inputs_on_different_devices(self) -> None:
        with pytest.raises(ValueError, match="same device"):
            tiled_matmul(torch.zeros(2, 2), torch.zeros(2, 2, device="meta"))


def test_offsets_that_overflow_i32_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real tensor this large would need many gigabytes, so the limit is lowered instead.
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setattr(utils, "_MAX_I32_OFFSET", 7)
    with pytest.raises(ValueError, match="more than 2\\*\\*31 elements"):
        fused_softmax(torch.zeros(3, 3))


def test_transposed_tensor_uses_its_full_memory_range() -> None:
    x = torch.zeros(2, 3)
    assert utils._last_element_offset(x) == utils._last_element_offset(x.t()) == 5


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((0, 2), (2, 4), False),  # adjacent rows do not overlap
        ((0, 3), (2, 4), True),  # row 2 is shared
        ((1, 2), (0, 4), True),  # one range contains the other
    ],
)
def test_overlap_detection_on_one_storage(
    first: tuple[int, int], second: tuple[int, int], expected: bool
) -> None:
    storage = torch.zeros(4, 3)
    a = storage[first[0] : first[1]]
    b = storage[second[0] : second[1]]
    assert utils._memory_ranges_overlap(a, b) is expected
    assert utils._memory_ranges_overlap(b, a) is expected


def test_separate_allocations_never_overlap() -> None:
    assert not utils._memory_ranges_overlap(torch.zeros(2, 2), torch.zeros(2, 2))
