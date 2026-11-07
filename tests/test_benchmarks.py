"""Test the benchmark plumbing without a GPU.

These tests use a fake timer, so they produce no real timings and nothing they compute is
ever written to results/. They check that the sweeps run real kernels, verify before timing,
record invalid configurations, and generate their output files.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from benchmarks import bench_matmul, bench_softmax, common
from benchmarks.common import STATUS_OK, STATUS_SKIPPED, STATUS_WRONG, Case, Environment, Row

ENV = Environment("Test GPU", "1.0", "2.0", "3.0", "3.12")
REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_timer(fn: object) -> float:
    return 2.0


def _case(output: torch.Tensor, reference: torch.Tensor) -> Case:
    return Case({"kernel": "k"}, lambda: output, reference, rtol=1e-5, atol=1e-6)


class TestMeasure:
    def test_correct_output_is_timed(self) -> None:
        result = common.measure(_case(torch.ones(2), torch.ones(2)), fake_timer)
        assert result.status == STATUS_OK
        assert result.time_ms == 2.0

    def test_wrong_output_is_never_timed(self) -> None:
        def must_not_run(fn: object) -> float:
            raise AssertionError("a wrong kernel was timed")

        result = common.measure(_case(torch.zeros(2), torch.ones(2)), must_not_run)
        assert result.status == STATUS_WRONG
        assert result.time_ms is None

    def test_invalid_configuration_is_recorded_with_its_reason(self) -> None:
        def out_of_resources() -> torch.Tensor:
            raise common.OutOfResources(100, 50, "shared memory")

        case = Case({}, out_of_resources, torch.ones(1), rtol=0, atol=0)
        result = common.measure(case, fake_timer)
        assert result.status == STATUS_SKIPPED
        assert "shared memory" in result.reason

    def test_unexpected_errors_are_not_hidden(self) -> None:
        def broken() -> torch.Tensor:
            raise RuntimeError("a real bug")

        case = Case({}, broken, torch.ones(1), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="a real bug"):
            common.measure(case, fake_timer)


class TestRefusals:
    """Both scripts must refuse to produce timings that mean nothing."""

    @pytest.mark.parametrize("script", ["benchmarks.bench_softmax", "benchmarks.bench_matmul"])
    def test_refuses_interpreter_mode(self, script: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", script, "--quick"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "TRITON_INTERPRET": "1"},
            check=False,
        )
        assert result.returncode != 0
        assert "Refusing to benchmark" in result.stderr

    def test_refuses_without_a_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRITON_INTERPRET", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(SystemExit, match="is_available"):
            common.require_compiled_gpu()


class TestSoftmaxSweep:
    def test_runs_real_kernels_and_verifies_them(self, device: str) -> None:
        rows = bench_softmax.run_sweep([(4, 64)], quick=True, device=device, timer=fake_timer)
        assert {row["kernel"] for row in rows} == {"torch.softmax", "fused", "online"}
        assert all(row["status"] == STATUS_OK for row in rows)

    def test_online_blocks_wider_than_the_row_are_left_out(self, device: str) -> None:
        rows = bench_softmax.run_sweep([(2, 8)], quick=True, device=device, timer=fake_timer)
        assert "online" not in {row["kernel"] for row in rows}

    def test_bandwidth_formula(self) -> None:
        # 1000 x 1000 float32, read and written, is 8 MB. In 1 ms that is 8 GB/s.
        assert bench_softmax.gigabytes_per_second(1000, 1000, 1.0) == pytest.approx(8.0)


class TestMatmulSweep:
    def test_runs_real_kernels_and_verifies_them(self, device: str) -> None:
        sweep = bench_matmul.Sweep(32, (16,), (16,), (4,), (3,), (1, 8))
        rows = bench_matmul.run_sweep(sweep, device=device, timer=fake_timer)
        assert all(row["status"] == STATUS_OK for row in rows)
        kernels = [row["kernel"] for row in rows]
        assert kernels[0] == "torch.matmul"
        # Grid: one configuration per precision. Scheduling stage: GROUP_M=8 per precision.
        assert len(kernels) == 1 + 2 + 2

    def test_no_configuration_is_timed_twice(self, device: str) -> None:
        sweep = bench_matmul.Sweep(16, (16,), (16,), (4,), (3,), (1,))
        rows = bench_matmul.run_sweep(sweep, device=device, timer=fake_timer)
        configs = [str(row["dot_precision"]) for row in rows if row["kernel"] == "tiled"]
        assert sorted(configs) == ["ieee", "tf32"]

    def test_tflops_formula(self) -> None:
        # 1000**3 * 2 operations in 1 ms is 2 TFLOPS.
        assert bench_matmul.tflops(1000, 1.0) == pytest.approx(2.0)


def _softmax_rows(fused_ms: float) -> list[Row]:
    base = {"n_rows": 4, "n_cols": 64, "status": STATUS_OK, "reason": ""}
    return [
        {**base, "kernel": "torch.softmax", "block_size": 0, "num_warps": 0,
         "time_ms": 2.0, "gb_per_s": 1.0},
        {**base, "kernel": "fused", "block_size": 64, "num_warps": 4,
         "time_ms": fused_ms, "gb_per_s": 2.0},
        {**base, "kernel": "online", "block_size": 16, "num_warps": 4,
         "time_ms": 3.0, "gb_per_s": 0.5},
        {**base, "kernel": "fused", "block_size": 128, "num_warps": 4, "status": STATUS_SKIPPED,
         "reason": "out of memory"},
    ]  # fmt: skip


def _matmul_rows(tiled_ms: float) -> list[Row]:
    size = {"m": 64, "n": 64, "k": 64, "status": STATUS_OK, "reason": ""}
    tiled = {
        **size, "kernel": "tiled", "block_m": 16, "block_n": 16, "block_k": 16, "group_m": 1,
        "num_warps": 4, "num_stages": 3,
    }  # fmt: skip
    return [
        {**size, "kernel": "torch.matmul", "time_ms": 2.0, "tflops": 1.0},
        {**tiled, "dot_precision": "ieee", "time_ms": tiled_ms, "tflops": 0.5},
        {**tiled, "dot_precision": "tf32", "time_ms": 1.0, "tflops": 2.0},
    ]


class TestSummaries:
    """The analysis must follow the numbers, in whichever direction they go."""

    def test_softmax_says_when_fused_wins(self) -> None:
        text = bench_softmax.build_summary(_softmax_rows(fused_ms=1.0), ENV)
        assert "fused kernel beat torch.softmax by 2.00x" in text
        assert "fewer round trips" in text
        assert "Test GPU" in text

    def test_softmax_says_when_pytorch_wins(self) -> None:
        text = bench_softmax.build_summary(_softmax_rows(fused_ms=4.0), ENV)
        assert "torch.softmax beat the fused kernel by 2.00x" in text

    def test_matmul_says_when_cublas_wins(self) -> None:
        text = bench_matmul.build_summary(_matmul_rows(tiled_ms=4.0), ENV)
        assert "torch.matmul (cuBLAS) beat this kernel by 2.00x" in text
        assert "no tuned software pipelining" in text

    def test_matmul_says_when_the_kernel_wins(self) -> None:
        text = bench_matmul.build_summary(_matmul_rows(tiled_ms=1.0), ENV)
        assert "this kernel beat torch.matmul (cuBLAS) by 2.00x" in text

    def test_tf32_is_flagged_as_not_like_for_like(self) -> None:
        text = bench_matmul.build_summary(_matmul_rows(tiled_ms=4.0), ENV)
        assert "not like for like" in text

    def test_summary_counts_skipped_configurations(self) -> None:
        text = bench_softmax.build_summary(_softmax_rows(fused_ms=1.0), ENV)
        assert "3 configurations were timed, 1 were skipped" in text


class TestOutputFiles:
    def test_csv_starts_with_the_environment(self, tmp_path: Path) -> None:
        path = tmp_path / "out" / "softmax.csv"
        common.write_csv(path, _softmax_rows(1.0), bench_softmax.FIELDNAMES, ENV)
        lines = path.read_text().splitlines()
        assert lines[0] == "# gpu: Test GPU"
        assert lines[1] == "# torch: 1.0"
        assert lines[5].startswith("kernel,n_rows")
        assert len(lines) == 5 + 1 + 4

    def test_softmax_plots_are_written(self, tmp_path: Path) -> None:
        block_panels, warp_panels = bench_softmax.build_panels(_softmax_rows(1.0))
        common.plot_panels(tmp_path / "block.png", block_panels)
        common.plot_panels(tmp_path / "warps.png", warp_panels)
        assert (tmp_path / "block.png").stat().st_size > 0
        assert (tmp_path / "warps.png").stat().st_size > 0

    def test_matmul_plots_are_written(self, tmp_path: Path) -> None:
        block_panels, schedule_panels = bench_matmul.build_panels(_matmul_rows(1.0))
        common.plot_panels(tmp_path / "block.png", block_panels)
        common.plot_panels(tmp_path / "schedule.png", schedule_panels)
        assert (tmp_path / "block.png").stat().st_size > 0

    def test_full_sweep_series_fit_the_palette(self) -> None:
        # A plot fails on more series than it can color, so the sweep values must fit.
        assert len(bench_matmul.FULL_SWEEP.blocks_k) <= len(common.SERIES_COLORS)
        assert len(bench_matmul.FULL_SWEEP.stages) <= len(common.SERIES_COLORS)

    def test_too_many_series_are_refused(self, tmp_path: Path) -> None:
        panel = common.Panel("t", "x", "y", {f"s{i}": [(1.0, 1.0)] for i in range(5)})
        with pytest.raises(ValueError, match="can be told apart"):
            common.plot_panels(tmp_path / "x.png", [panel])

    def test_summary_sections_are_independent(self, tmp_path: Path) -> None:
        path = tmp_path / "SUMMARY.md"
        common.update_summary(path, "softmax", "first")
        common.update_summary(path, "matmul", "second")
        common.update_summary(path, "softmax", "updated")
        text = path.read_text()
        assert "updated" in text
        assert "second" in text
        assert "first" not in text

    def test_series_sort_numerically(self) -> None:
        labels = ["K=128", "K=16", "K=32"]
        assert sorted(labels, key=common._natural_key) == ["K=16", "K=32", "K=128"]
