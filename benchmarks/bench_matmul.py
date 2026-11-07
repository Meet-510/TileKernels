"""Benchmark the tiled matmul kernel against torch.matmul on a GPU.

The sweep has two stages. The first is a grid over BLOCK_M, BLOCK_N, BLOCK_K and tl.dot
precision. The second takes the fastest blocks at each precision, then varies num_warps,
num_stages and GROUP_M. A full grid of everything would need hundreds of compilations.

Every configuration is verified against PyTorch before it is timed. Results go to
results/matmul.csv, two plots and a section of results/SUMMARY.md.

    python -m benchmarks.bench_matmul          # the full sweep
    python -m benchmarks.bench_matmul --quick  # a small sweep, to check that everything works
"""

import argparse
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from benchmarks.common import (
    Case,
    Environment,
    Panel,
    Row,
    Timer,
    count_not_timed,
    do_bench_timer,
    float_value,
    markdown_table,
    measure,
    plot_panels,
    read_environment,
    require_compiled_gpu,
    successful,
    update_summary,
    write_csv,
)
from tilekernels import tiled_matmul
from tilekernels.matmul import DotPrecision

FIELDNAMES = (
    "kernel", "m", "n", "k", "block_m", "block_n", "block_k", "group_m", "dot_precision",
    "num_warps", "num_stages", "status", "reason", "time_ms", "tflops",
)  # fmt: skip

TORCH_KERNEL = "torch.matmul"
TILED_KERNEL = "tiled"
PRECISIONS: tuple[DotPrecision, ...] = ("ieee", "tf32")

# TF32 keeps 10 mantissa bits, so its products are only accurate to about 1e-3. The IEEE
# tolerance allows float32 rounding in a sum of K products, as explained in tests/kernel_testing.
TF32_TOLERANCE = 5e-2
IEEE_RTOL = 1e-3


class Sweep(NamedTuple):
    """The values one run explores."""

    size: int
    blocks_mn: tuple[int, ...]
    blocks_k: tuple[int, ...]
    warps: tuple[int, ...]
    stages: tuple[int, ...]
    groups: tuple[int, ...]


FULL_SWEEP = Sweep(2048, (32, 64, 128, 256), (16, 32, 64, 128), (2, 4, 8), (2, 3, 4), (1, 8))
QUICK_SWEEP = Sweep(512, (64, 128), (32,), (4,), (3,), (1,))


class Config(NamedTuple):
    """One configuration of the tiled kernel."""

    block_m: int
    block_n: int
    block_k: int
    group_m: int = 1
    dot_precision: DotPrecision = "ieee"
    num_warps: int = 4
    num_stages: int = 3


class Workload:
    """Fixed square inputs and an output, shared by the baseline and every configuration."""

    def __init__(self, size: int, device: str) -> None:
        generator = torch.Generator("cpu").manual_seed(0)
        self.size = size
        self.a = torch.randn(size, size, generator=generator).to(device)
        self.b = torch.randn(size, size, generator=generator).to(device)
        # The baseline and every kernel write here, so allocation is left out of the timings.
        self.out = torch.empty_like(self.a)
        self.reference = torch.matmul(self.a, self.b)

    def _base_params(self) -> Row:
        return {"m": self.size, "n": self.size, "k": self.size}

    def baseline_case(self) -> Case:
        """Return the torch.matmul case. Its output is its own reference."""
        return Case(
            {**self._base_params(), "kernel": TORCH_KERNEL},
            lambda: torch.matmul(self.a, self.b, out=self.out),
            self.reference,
            IEEE_RTOL,
            self._ieee_atol(),
        )

    def kernel_case(self, config: Config) -> Case:
        """Return the case for one configuration of the tiled kernel."""
        is_tf32 = config.dot_precision == "tf32"
        return Case(
            {**self._base_params(), "kernel": TILED_KERNEL, **config._asdict()},
            lambda: tiled_matmul(self.a, self.b, out=self.out, **config._asdict()),
            self.reference,
            TF32_TOLERANCE if is_tf32 else IEEE_RTOL,
            TF32_TOLERANCE if is_tf32 else self._ieee_atol(),
        )

    def _ieee_atol(self) -> float:
        return 4 * self.size * torch.finfo(torch.float32).eps


def tflops(size: int, time_ms: float) -> float:
    """Return floating point throughput. A square matmul does 2 * size**3 operations."""
    return 2 * size**3 / (time_ms * 1e-3) / 1e12


def _record(case: Case, size: int, timer: Timer) -> Row:
    """Measure one case and turn it into a CSV row."""
    result = measure(case, timer)
    row = dict(case.params, status=result.status, reason=result.reason)
    if result.time_ms is not None:
        row["time_ms"] = round(result.time_ms, 4)
        row["tflops"] = round(tflops(size, result.time_ms), 4)
    print(row, flush=True)
    return row


def run_sweep(sweep: Sweep, *, device: str, timer: Timer) -> list[Row]:
    """Measure the baseline, then the block grid, then the scheduling sweep."""
    workload = Workload(sweep.size, device)
    rows = [_record(workload.baseline_case(), sweep.size, timer)]

    for block_m, block_n, block_k, precision in itertools.product(
        sweep.blocks_mn, sweep.blocks_mn, sweep.blocks_k, PRECISIONS
    ):
        config = Config(block_m, block_n, block_k, dot_precision=precision)
        rows.append(_record(workload.kernel_case(config), sweep.size, timer))

    seen = {_config_of(row) for row in rows if row["kernel"] == TILED_KERNEL}
    for precision in PRECISIONS:
        best = _fastest(rows, precision)
        if best is None:
            continue
        for num_warps, num_stages, group_m in itertools.product(
            sweep.warps, sweep.stages, sweep.groups
        ):
            config = Config(
                int(float_value(best, "block_m")),
                int(float_value(best, "block_n")),
                int(float_value(best, "block_k")),
                group_m,
                precision,
                num_warps,
                num_stages,
            )
            if config not in seen:
                seen.add(config)
                rows.append(_record(workload.kernel_case(config), sweep.size, timer))
    return rows


def _config_of(row: Row) -> Config:
    """Read the Config back out of a row."""
    values = {name: row[name] for name in Config._fields}
    return Config(**values)  # type: ignore[arg-type]  # The row was built from a Config.


def _fastest(rows: Sequence[Row], precision: str) -> Row | None:
    """Return the fastest timed configuration at one precision, if there is one."""
    matching = [r for r in successful(rows) if r.get("dot_precision") == precision]
    return min(matching, key=lambda r: float_value(r, "time_ms"), default=None)


def _baseline(rows: Sequence[Row]) -> Row | None:
    """Return the timed torch.matmul row, if there is one."""
    return next((r for r in successful(rows) if r["kernel"] == TORCH_KERNEL), None)


def build_panels(rows: Sequence[Row]) -> tuple[list[Panel], list[Panel]]:
    """Build the block size figure and the scheduling figure, with one panel per precision."""
    baseline = _baseline(rows)
    baseline_tflops = float_value(baseline, "tflops") if baseline else None
    block_panels, schedule_panels = [], []
    for precision in PRECISIONS:
        timed = [r for r in successful(rows) if r.get("dot_precision") == precision]
        if not timed:
            continue
        block_panel = Panel(
            f"{precision}: best by tile size",
            "BLOCK_M = BLOCK_N",
            "Throughput (TFLOPS)",
            baseline=baseline_tflops,
            baseline_label=TORCH_KERNEL,
        )
        schedule_panel = Panel(
            f"{precision}: best by warps",
            "num_warps",
            "Throughput (TFLOPS)",
            baseline=baseline_tflops,
            baseline_label=TORCH_KERNEL,
        )
        for row in timed:
            if row["block_m"] == row["block_n"]:
                _keep_highest(block_panel, f"BLOCK_K={row['block_k']}", row, "block_m")
            _keep_highest(schedule_panel, f"num_stages={row['num_stages']}", row, "num_warps")
        block_panels.append(block_panel)
        schedule_panels.append(schedule_panel)
    return block_panels, schedule_panels


def _keep_highest(panel: Panel, label: str, row: Row, x_key: str) -> None:
    """Add a row to a series, keeping only the highest throughput at each x value."""
    x, value = float_value(row, x_key), float_value(row, "tflops")
    points = dict(panel.series.get(label, []))
    points[x] = max(value, points.get(x, value))
    panel.series[label] = list(points.items())


def build_summary(rows: Sequence[Row], env: Environment) -> str:
    """Write the matmul section of the summary, with analysis that follows the numbers."""
    baseline = _baseline(rows)
    best_by_precision = {p: _fastest(rows, p) for p in PRECISIONS}
    table: list[tuple[object, ...]] = []
    if baseline:
        table.append((TORCH_KERNEL, "-", "-", "-", "-", baseline["time_ms"], baseline["tflops"]))
    for precision, row in best_by_precision.items():
        if row:
            blocks = f"{row['block_m']}x{row['block_n']}x{row['block_k']}"
            schedule = f"{row['num_warps']} / {row['num_stages']}"
            table.append(
                (TILED_KERNEL, precision, blocks, row["group_m"], schedule,
                 row["time_ms"], row["tflops"])
            )  # fmt: skip
    headers = ("kernel", "precision", "M x N x K", "GROUP_M", "warps / stages", "ms", "TFLOPS")
    parts = [
        "## Matmul",
        f"Multiplying two {rows[0]['m']} x {rows[0]['m']} float32 matrices.",
        "Measured on: " + "; ".join(env.as_lines()),
        markdown_table(headers, table),
        count_not_timed(rows),
    ]
    best = best_by_precision["ieee"]
    if baseline and best:
        parts.append(_ieee_verdict(baseline, best))
    tf32 = best_by_precision["tf32"]
    if baseline and tf32:
        parts.append(_tf32_note(baseline, tf32))
    return "\n\n".join(parts)


def _ieee_verdict(baseline: Row, best: Row) -> str:
    """Compare the like-for-like case, since torch.matmul does not use TF32 by default."""
    baseline_ms = float_value(baseline, "time_ms")
    best_ms = float_value(best, "time_ms")
    if best_ms >= baseline_ms:
        return (
            f"At full float32 precision, torch.matmul (cuBLAS) beat this kernel by "
            f"{best_ms / baseline_ms:.2f}x. That is the expected outcome. Likely reasons: "
            "this kernel has no tuned software pipelining, no layout chosen for tensor cores "
            "and no autotuned configurations, while cuBLAS is tuned by NVIDIA for each GPU."
        )
    return (
        f"At full float32 precision, this kernel beat torch.matmul (cuBLAS) by "
        f"{baseline_ms / best_ms:.2f}x on this GPU. That is a surprising result, so check it "
        "with larger sizes and repeated runs before drawing conclusions from it."
    )


def _tf32_note(baseline: Row, tf32: Row) -> str:
    """Compare against TF32 while making clear that it is not a like-for-like comparison."""
    ratio = float_value(baseline, "time_ms") / float_value(tf32, "time_ms")
    return (
        f"The fastest TF32 configuration ran at {ratio:.2f}x the speed of torch.matmul, but "
        "this is not like for like: TF32 trades accuracy for tensor core throughput, while "
        "torch.matmul keeps full float32 precision by default."
    )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--quick", action="store_true", help="run a small sweep")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    """Run the sweep and write the CSV, plots and summary."""
    args = parse_args()
    require_compiled_gpu()
    env = read_environment()
    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    rows = run_sweep(sweep, device="cuda", timer=do_bench_timer)

    write_csv(args.output_dir / "matmul.csv", rows, FIELDNAMES, env)
    block_panels, schedule_panels = build_panels(rows)
    plot_panels(args.output_dir / "matmul_block_size.png", block_panels)
    plot_panels(args.output_dir / "matmul_schedule.png", schedule_panels)
    update_summary(args.output_dir / "SUMMARY.md", "matmul", build_summary(rows, env))
    print(f"Wrote results to {args.output_dir}. Timed: {len(successful(rows))}")


if __name__ == "__main__":
    main()
