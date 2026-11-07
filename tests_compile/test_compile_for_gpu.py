"""Compile the kernels for real GPU targets, with no GPU needed.

The interpreter never runs the compiler, so a kernel can pass every interpreter test and still
fail to compile. These tests cover that gap, and pin the facts about generated code that
docs/COMPILER_DUMPS.md relies on. They do not prove that the code computes the right answer.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts import dump_ir, ir_tools
from triton.compiler.errors import CompilationError

from tilekernels import matmul, softmax

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHS = [75, 80, 86, 90, 100]
ALL_STAGES = {"ttir", "ttgir", "llir", "ptx", "cubin"}


def matmul_constexprs(
    *, block_m: int = 64, block_n: int = 64, block_k: int = 32, group_m: int = 1,
    precision: str = "ieee",
) -> dict[str, int | str]:  # fmt: skip
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "DOT_PRECISION": precision,
    }


def code_of(result: ir_tools.CompiledResult) -> str:
    """Return the PTX kernel body, without the debug sections that follow it."""
    return str(result.compiled.asm["ptx"]).split(".section", 1)[0]


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
class TestEveryKernelCompiles:
    @pytest.mark.parametrize(
        "kernel", [softmax._fused_softmax_kernel, softmax._online_softmax_kernel]
    )
    @pytest.mark.parametrize("block_size", [1, 64, 1024])
    def test_softmax(self, kernel: Any, block_size: int, dtype: str, arch: int) -> None:
        result = ir_tools.compile_kernel(kernel, {"BLOCK_SIZE": block_size}, dtype=dtype, arch=arch)
        assert set(result.compiled.asm) >= ALL_STAGES

    @pytest.mark.parametrize("group_m", [1, 8])
    def test_matmul(self, group_m: int, dtype: str, arch: int) -> None:
        constexprs = matmul_constexprs(group_m=group_m)
        result = ir_tools.compile_kernel(
            matmul._tiled_matmul_kernel, constexprs, dtype=dtype, arch=arch
        )
        assert set(result.compiled.asm) >= ALL_STAGES


class TestDotLowering:
    """What tl.dot becomes on each target. docs/RUNNING_ON_A_GPU.md has this as a table."""

    @pytest.mark.parametrize("arch", ARCHS)
    def test_ieee_float32_is_an_fma_loop_everywhere(self, arch: int) -> None:
        result = ir_tools.compile_kernel(
            matmul._tiled_matmul_kernel, matmul_constexprs(precision="ieee"), arch=arch
        )
        code = code_of(result)
        assert re.search(r"\bfma\.", code)
        assert not re.search(r"\bmma\.|wgmma\.|tcgen05\.mma", code)

    def test_t4_has_no_tensor_cores_even_with_tf32(self) -> None:
        result = ir_tools.compile_kernel(
            matmul._tiled_matmul_kernel, matmul_constexprs(precision="tf32"), arch=75
        )
        assert re.search(r"\bfma\.", code_of(result))
        assert not re.search(r"\bmma\.", code_of(result))

    @pytest.mark.parametrize(
        ("arch", "instruction"),
        [(80, r"\bmma\.sync"), (86, r"\bmma\.sync"), (90, r"wgmma\."), (100, r"tcgen05\.mma")],
    )
    @pytest.mark.parametrize(("dtype", "precision"), [("fp32", "tf32"), ("fp16", "ieee")])
    def test_newer_targets_use_tensor_cores(
        self, arch: int, instruction: str, dtype: str, precision: str
    ) -> None:
        result = ir_tools.compile_kernel(
            matmul._tiled_matmul_kernel,
            matmul_constexprs(precision=precision),
            dtype=dtype,
            arch=arch,
        )
        assert re.search(instruction, code_of(result))
        assert not re.search(r"\bfma\.", code_of(result))


class TestMinimumBlockK:
    """The limit that matmul._MIN_BLOCK_K enforces comes from the compiler. Check it still holds."""

    @pytest.mark.parametrize(("dtype", "too_small"), [("fp32", 4), ("fp16", 8)])
    def test_smaller_block_k_does_not_compile(self, dtype: str, too_small: int) -> None:
        with pytest.raises(CompilationError, match="K >= "):
            ir_tools.compile_kernel(
                matmul._tiled_matmul_kernel,
                matmul_constexprs(block_k=too_small),
                dtype=dtype,
            )

    @pytest.mark.parametrize(("dtype", "minimum"), [("fp32", 8), ("fp16", 16)])
    def test_the_minimum_compiles_with_the_smallest_tile(self, dtype: str, minimum: int) -> None:
        # M and N have no such limit, and tests rely on BLOCK_M = BLOCK_N = 1.
        ir_tools.compile_kernel(
            matmul._tiled_matmul_kernel,
            matmul_constexprs(block_m=1, block_n=1, block_k=minimum),
            dtype=dtype,
        )


class TestNumStages:
    def test_has_no_effect_on_t4(self) -> None:
        # Below sm_80 the backend does not run the software pipeliner at all.
        constexprs = matmul_constexprs(precision="tf32")
        codes = {
            code_of(
                ir_tools.compile_kernel(
                    matmul._tiled_matmul_kernel, constexprs, arch=75, num_stages=stages
                )
            )
            for stages in (1, 4)
        }
        assert len(codes) == 1

    def test_buffers_more_tiles_in_shared_memory_on_newer_targets(self) -> None:
        constexprs = matmul_constexprs(precision="tf32")
        shared = [
            ir_tools.compile_kernel(
                matmul._tiled_matmul_kernel, constexprs, arch=86, num_stages=stages
            ).compiled.metadata.shared
            for stages in (2, 3, 4)
        ]
        # One tile each of A and B is 64*32*4 + 32*64*4 = 16384 bytes. Each stage adds one set.
        assert shared == [16384, 32768, 49152]


class TestSoftmaxCode:
    def test_mask_becomes_a_predicate_on_the_load(self) -> None:
        result = ir_tools.compile_kernel(softmax._fused_softmax_kernel, {"BLOCK_SIZE": 64})
        assert re.search(r"@%p\d+ ld\.global\.b32", code_of(result))

    def test_reductions_become_warp_shuffles(self) -> None:
        result = ir_tools.compile_kernel(softmax._fused_softmax_kernel, {"BLOCK_SIZE": 64})
        assert "shfl.sync.bfly" in code_of(result)

    def test_one_warp_needs_no_shared_memory_or_barrier(self) -> None:
        result = ir_tools.compile_kernel(
            softmax._fused_softmax_kernel, {"BLOCK_SIZE": 64}, num_warps=1
        )
        assert result.compiled.metadata.shared == 0
        assert "bar.sync" not in code_of(result)

    def test_several_warps_combine_reductions_through_shared_memory(self) -> None:
        result = ir_tools.compile_kernel(
            softmax._fused_softmax_kernel, {"BLOCK_SIZE": 64}, num_warps=4
        )
        assert result.compiled.metadata.shared > 0
        assert "bar.sync" in code_of(result)
        assert "st.shared" in code_of(result)

    def test_a_large_block_loads_several_elements_per_thread(self) -> None:
        # 1024 elements on 128 threads is 8 per thread, each with its own predicated load.
        result = ir_tools.compile_kernel(softmax._fused_softmax_kernel, {"BLOCK_SIZE": 1024})
        assert len(re.findall(r"@%p\d+ ld\.global\.b32", code_of(result))) == 8


class TestTools:
    def test_ptxas_log_is_parsed(self) -> None:
        log = (
            "ptxas info    : Function properties for kernel\n"
            "    0 bytes stack frame, 12 bytes spill stores, 8 bytes spill loads\n"
            "ptxas info    : Used 15 registers, used 1 barriers, 408 bytes cmem[0]\n"
        )
        stats = ir_tools.parse_ptxas_log(log)
        assert stats == ir_tools.PtxasStats(15, 12, 8)

    def test_unexpected_ptxas_log_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unexpected ptxas log"):
            ir_tools.parse_ptxas_log("nothing useful")

    def test_registers_are_reported_on_every_target(self) -> None:
        # The register field of the ELF section is empty on sm_90 and newer. A count of zero
        # would be wrong, which is why the count comes from the ptxas log.
        for arch in ARCHS:
            result = ir_tools.compile_kernel(
                softmax._fused_softmax_kernel, {"BLOCK_SIZE": 64}, arch=arch
            )
            assert result.stats.registers > 0

    def test_environment_is_restored_after_compiling(self) -> None:
        import os

        before = os.environ.get("TRITON_DUMP_PTXAS_LOG")
        ir_tools.compile_kernel(softmax._fused_softmax_kernel, {"BLOCK_SIZE": 16})
        assert os.environ.get("TRITON_DUMP_PTXAS_LOG") == before

    def test_code_size_of_a_cubin(self) -> None:
        kernel = softmax._fused_softmax_kernel
        cubin = ir_tools.compile_kernel(kernel, {"BLOCK_SIZE": 16}).compiled.asm["cubin"]
        assert ir_tools.read_code_size(cubin, kernel.__name__) > 0

    def test_code_size_rejects_other_files(self) -> None:
        with pytest.raises(ValueError, match="not an ELF"):
            ir_tools.read_code_size(b"not a cubin", "kernel")

    def test_code_size_rejects_a_missing_kernel(self) -> None:
        kernel = softmax._fused_softmax_kernel
        cubin = ir_tools.compile_kernel(kernel, {"BLOCK_SIZE": 16}).compiled.asm["cubin"]
        with pytest.raises(ValueError, match="no section named"):
            ir_tools.read_code_size(cubin, "another_kernel")


class TestDumpScript:
    def test_writes_every_stage_and_metadata(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/dump_ir.py", "--kernel", "fused_softmax",
             "--set", "BLOCK_SIZE=32", "--num-warps", "2", "--arch", "90",
             "--output", str(tmp_path)],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )  # fmt: skip
        assert result.returncode == 0, result.stderr
        folder = tmp_path / "fused_softmax"
        for stage in ALL_STAGES:
            assert (folder / f"fused_softmax.{stage}").stat().st_size > 0
        metadata = json.loads((folder / "metadata.json").read_text())
        assert metadata["constexprs"] == {"BLOCK_SIZE": 32}
        assert metadata["target"] == "sm_90"
        assert metadata["num_warps"] == 2
        assert metadata["registers"] > 0
        assert set(metadata["stages_produced"]) - {"sass"} == ALL_STAGES

    def test_dumps_reveal_no_local_paths(self, tmp_path: Path) -> None:
        dump_ir.dump_kernel(
            "fused_softmax", tmp_path, overrides={}, dtype="fp32", arch=86, num_warps=4,
            num_stages=3,
        )  # fmt: skip
        text = (tmp_path / "fused_softmax" / "fused_softmax.ttir").read_text()
        assert str(REPO_ROOT) not in text
        assert str(REPO_ROOT).replace("\\", "\\\\") not in text
        assert "<repo>/src/tilekernels/softmax.py" in text

    def test_ptx_keeps_line_info_but_not_the_dwarf_tables(self, tmp_path: Path) -> None:
        # The DWARF tables hold the checkout path as raw bytes, where text replacement misses it.
        dump_ir.dump_kernel(
            "fused_softmax", tmp_path, overrides={}, dtype="fp32", arch=86, num_warps=4,
            num_stages=3,
        )  # fmt: skip
        ptx = (tmp_path / "fused_softmax" / "fused_softmax.ptx").read_text()
        assert ".section" not in ptx
        assert "DW_AT_comp_dir" not in ptx
        assert ".loc" in ptx
        assert "ld.global.b32" in ptx

    @pytest.mark.parametrize(
        ("argument", "expected"),
        [("BLOCK_SIZE=64", ("BLOCK_SIZE", 64)), ("DOT_PRECISION=tf32", ("DOT_PRECISION", "tf32"))],
    )
    def test_constexpr_values_are_parsed(self, argument: str, expected: tuple[str, object]) -> None:
        assert dump_ir.parse_constexpr(argument) == expected

    def test_a_malformed_constexpr_is_rejected(self) -> None:
        with pytest.raises(Exception, match="NAME=VALUE"):
            dump_ir.parse_constexpr("BLOCK_SIZE")
