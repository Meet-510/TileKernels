"""Compile a Triton kernel for an explicit GPU target, and inspect the result.

None of this needs a GPU. triton.compile only needs a GPUTarget, and the Triton wheel bundles
ptxas, so every stage down to the cubin can be produced on any machine.
"""

import contextlib
import io
import os
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

# Field offsets and formats from the ELF64 specification, for a little-endian file.
_ELF_MAGIC = b"\x7fELF"
_ELF_SECTION_TABLE_OFFSET = 0x28
_ELF_SECTION_COUNTS_OFFSET = 0x3A
_ELF_SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")


@dataclass(frozen=True)
class PtxasStats:
    """Register use, as reported by ptxas when it assembled the kernel."""

    registers: int
    spill_store_bytes: int
    spill_load_bytes: int


@dataclass(frozen=True)
class CompiledResult:
    """A compiled kernel and what the assembler reported about it."""

    compiled: Any
    stats: PtxasStats


def compile_kernel(
    kernel: Any,
    constexprs: dict[str, int | str],
    *,
    dtype: str = "fp32",
    arch: int = 86,
    num_warps: int = 4,
    num_stages: int = 3,
) -> CompiledResult:
    """Compile a kernel from scratch for a CUDA architecture, such as 86 for sm_86.

    The signature is built from the kernel's argument names: *_ptr arguments are pointers to
    dtype, names in constexprs are compile-time constants, and everything else is an i32.
    That covers every kernel in this repo.

    The returned .compiled is Triton's compiled kernel. Its .asm dict maps stage names to
    generated code, and .metadata holds the options and the shared memory it needs.
    """
    signature = {}
    for name in kernel.arg_names:
        if name in constexprs:
            signature[name] = "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = f"*{dtype}"
        else:
            signature[name] = "i32"
    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)

    log = io.StringIO()
    with _fresh_compile_with_ptxas_log(), contextlib.redirect_stdout(log):
        compiled = triton.compile(
            source,
            target=GPUTarget("cuda", arch, 32),
            options={"num_warps": num_warps, "num_stages": num_stages},
        )
    return CompiledResult(compiled, parse_ptxas_log(log.getvalue()))


@contextlib.contextmanager
def _fresh_compile_with_ptxas_log() -> Iterator[None]:
    """Make Triton skip its cache and print the ptxas log, then restore the environment.

    A cached kernel never runs ptxas, so it would have no log. Compiling afresh also keeps
    stale source paths from an earlier checkout out of the generated code.
    """
    settings = {"TRITON_ALWAYS_COMPILE": "1", "TRITON_DUMP_PTXAS_LOG": "1"}
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value


def parse_ptxas_log(log: str) -> PtxasStats:
    """Extract register and spill counts from the output of `ptxas -v`."""
    registers = re.search(r"Used (\d+) registers", log)
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if registers is None or spills is None:
        raise ValueError(f"unexpected ptxas log: {log!r}")
    return PtxasStats(int(registers.group(1)), int(spills.group(1)), int(spills.group(2)))


def read_code_size(cubin: bytes, kernel_name: str) -> int:
    """Return the size in bytes of a kernel's machine code, from the cubin's ELF section table."""
    if cubin[:4] != _ELF_MAGIC:
        raise ValueError("not an ELF file")
    table_offset = struct.unpack_from("<Q", cubin, _ELF_SECTION_TABLE_OFFSET)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", cubin, _ELF_SECTION_COUNTS_OFFSET)
    headers = [
        _ELF_SECTION_HEADER.unpack_from(cubin, table_offset + index * entry_size)
        for index in range(count)
    ]
    names_offset = headers[names_index][4]

    target = f".text.{kernel_name}"
    for header in headers:
        name_start = names_offset + header[0]
        if cubin[name_start : cubin.index(b"\0", name_start)].decode() == target:
            return int(header[5])
    raise ValueError(f"no section named {target}")
