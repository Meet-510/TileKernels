"""Helpers shared by the kernel wrappers: mode detection, block sizes and input validation."""

import os

import torch
import triton.language as tl

SUPPORTED_DTYPES = (torch.float32, torch.float16)

# Kernels compute memory offsets with i32 arithmetic, because tl.program_id and the integer
# kernel arguments are i32. An offset that does not fit would silently wrap around.
_MAX_I32_OFFSET = 2**31 - 1


def is_interpreter_mode() -> bool:
    """Return True when Triton will run kernels in its CPU interpreter.

    This reads the environment variable that Triton itself reads, so the answer
    stays correct across Triton versions.
    """
    return os.environ.get("TRITON_INTERPRET", "0").lower() in {"1", "true", "on", "y", "yes"}


def is_power_of_two(value: int) -> bool:
    """Return True if value is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two that is greater than or equal to value."""
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def validate_block_size(name: str, block_size: int) -> None:
    """Check that a block size is one Triton can compile.

    Triton's tl.arange only accepts power-of-two ranges, and block tensors have a
    hard upper limit on their element count.
    """
    if not is_power_of_two(block_size):
        raise ValueError(f"{name} must be a power of two, got {block_size}")
    if block_size > tl.TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"{name}={block_size} exceeds the limit of {tl.TRITON_MAX_TENSOR_NUMEL}")


def validate_input(name: str, tensor: torch.Tensor) -> None:
    """Check that a tensor can be read by a kernel, and raise a clear error if it cannot.

    Strides are unrestricted, because every kernel takes them as arguments, so
    non-contiguous inputs such as transposes and slices are accepted.
    """
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty, got shape {tuple(tensor.shape)}")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"{name} must be float32 or float16, got {tensor.dtype}")
    if tensor.requires_grad:
        # Kernels write through raw pointers, so autograd would silently lose the graph.
        raise ValueError(f"{name} requires grad, but kernels have no backward pass; detach it")
    if not is_interpreter_mode() and tensor.device.type != "cuda":
        raise ValueError(
            f"{name} is on {tensor.device}, but compiled kernels need a CUDA tensor. "
            "Move it to a GPU, or set TRITON_INTERPRET=1 before importing triton."
        )
    _validate_i32_addressable(name, tensor)


def validate_output(
    out: torch.Tensor, shape: tuple[int, ...], like: torch.Tensor, inputs: tuple[torch.Tensor, ...]
) -> None:
    """Check that a caller-provided output tensor can safely be written by a kernel."""
    validate_input("out", out)
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}, got {tuple(out.shape)}")
    if out.dtype != like.dtype:
        raise TypeError(f"out must have dtype {like.dtype}, got {out.dtype}")
    if out.device != like.device:
        raise ValueError(f"out must be on {like.device}, got {out.device}")
    if any(out.size(dim) > 1 and out.stride(dim) == 0 for dim in range(out.ndim)):
        # An expanded tensor maps several output elements onto a single memory location.
        raise ValueError("out must not have a zero stride, because elements would overwrite")
    for tensor in inputs:
        if _memory_ranges_overlap(out, tensor):
            raise ValueError("out must not overlap an input tensor")


def _validate_i32_addressable(name: str, tensor: torch.Tensor) -> None:
    """Reject tensors whose furthest element offset does not fit in a signed 32-bit int."""
    if _last_element_offset(tensor) > _MAX_I32_OFFSET:
        raise ValueError(f"{name} spans more than 2**31 elements, which kernels cannot address")


def _last_element_offset(tensor: torch.Tensor) -> int:
    """Return the distance in elements from the first element to the furthest element."""
    return sum((tensor.size(dim) - 1) * tensor.stride(dim) for dim in range(tensor.ndim))


def _memory_ranges_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return True if two tensors touch overlapping bytes of the same storage.

    This compares the outer [start, end) byte ranges. Interleaved views of one
    storage count as overlapping too, which is stricter than needed but always safe.
    """
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return False
    first_start, first_end = _byte_range(first)
    second_start, second_end = _byte_range(second)
    return first_start < second_end and second_start < first_end


def _byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the [start, end) byte range of a tensor, relative to its storage."""
    item_size = tensor.element_size()
    start = int(tensor.storage_offset()) * item_size
    return start, start + (_last_element_offset(tensor) + 1) * item_size
