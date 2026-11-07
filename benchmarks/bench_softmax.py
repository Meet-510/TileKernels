"""Benchmark the softmax kernels against torch.softmax on a GPU.

Sweeps BLOCK_SIZE and num_warps, verifies each configuration against PyTorch before timing
it, and writes results/softmax.csv, two plots and a section of results/SUMMARY.md.

    python -m benchmarks.bench_softmax          # the full sweep
    python -m benchmarks.bench_softmax --quick  # a small sweep, to check that everything works
"""

import argparse
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from pathlib import Path

import torch

from benchmarks.common import (
    STATUS_OK,
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
from tilekernels import fused_softmax, online_softmax
from tilekernels.utils import next_power_of_two

FIELDNAMES = (
    "kernel", "n_rows", "n_cols", "block_size", "num_warps",
    "status", "reason", "time_ms", "gb_per_s",
)  # fmt: skip

FULL_SHAPES = ((4096, 256), (4096, 1024), (1024, 4096))
QUICK_SHAPES = ((1024, 256),)
FULL_WARPS = (1, 2, 4, 8, 16)
QUICK_WARPS = (4, 8)
ONLINE_BLOCKS = (64, 256, 1024)

# float32 rounding, with room for the different exp() implementations on each side.
RTOL = 1e-4
ATOL = 1e-5
TORCH_KERNEL = "torch.softmax"


def gigabytes_per_second(n_rows: int, n_cols: int, time_ms: float) -> float:
    """Effective bandwidth: the tensor read once and written once, divided by the time.

    This uses the same formula for every kernel, so it compares fairly. The online kernel
    really reads twice, so it moves more data than this figure counts.
    """
    bytes_moved = 2 * n_rows * n_cols * torch.finfo(torch.float32).bits // 8
    return bytes_moved / (time_ms * 1e-3) / 1e9


def build_cases(shape: tuple[int, int], *, quick: bool, device: str) -> Iterator[Case]:
    """Yield the PyTorch baseline for a shape, followed by every kernel configuration."""
    generator = torch.Generator("cpu").manual_seed(0)
    x = torch.randn(shape, generator=generator).to(device)
    out = torch.empty_like(x)
    reference = torch.softmax(x, dim=-1)
    n_rows, n_cols = shape

    def params(kernel: str, block_size: int, num_warps: int) -> Row:
        return {
            "kernel": kernel,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "block_size": block_size,
            "num_warps": num_warps,
        }

    def case(parameters: Row, run: Callable[[], torch.Tensor]) -> Case:
        return Case(parameters, run, reference, RTOL, ATOL)

    # All the kernels and the baseline write into the same preallocated tensor, so allocation
    # time is left out of every measurement.
    yield case(params(TORCH_KERNEL, 0, 0), lambda: torch.softmax(x, dim=-1, out=out))

    smallest = next_power_of_two(n_cols)
    for scale in (1, 2) if quick else (1, 2, 4):
        for num_warps in QUICK_WARPS if quick else FULL_WARPS:
            block = smallest * scale
            yield case(
                params("fused", block, num_warps),
                partial(fused_softmax, x, out=out, block_size=block, num_warps=num_warps),
            )

    for block in ONLINE_BLOCKS[:1] if quick else ONLINE_BLOCKS:
        if block > n_cols:
            continue
        for num_warps in QUICK_WARPS:
            yield case(
                params("online", block, num_warps),
                partial(online_softmax, x, out=out, block_size=block, num_warps=num_warps),
            )


def run_sweep(
    shapes: Sequence[tuple[int, int]], *, quick: bool, device: str, timer: Timer
) -> list[Row]:
    """Measure every case for every shape, and return one row per case."""
    rows = []
    for shape in shapes:
        for case in build_cases(shape, quick=quick, device=device):
            result = measure(case, timer)
            row = dict(case.params, status=result.status, reason=result.reason)
            if result.time_ms is not None:
                row["time_ms"] = round(result.time_ms, 4)
                row["gb_per_s"] = round(gigabytes_per_second(*shape, result.time_ms), 3)
            print(row, flush=True)
            rows.append(row)
    return rows


def _best(rows: Sequence[Row], kernel: str, n_cols: object) -> Row | None:
    """Return the fastest timed row for one kernel at one width, if there is one."""
    matching = [r for r in successful(rows) if r["kernel"] == kernel and r["n_cols"] == n_cols]
    return min(matching, key=lambda r: float_value(r, "time_ms"), default=None)


def build_panels(rows: Sequence[Row]) -> tuple[list[Panel], list[Panel]]:
    """Build the block size and num_warps figures, with one panel per row width."""
    block_panels, warp_panels = [], []
    for n_cols in sorted({int(float_value(r, "n_cols")) for r in rows}):
        baseline = _best(rows, TORCH_KERNEL, n_cols)
        baseline_ms = float_value(baseline, "time_ms") if baseline else None
        block_panel = Panel(
            f"Best time by block, {n_cols} columns",
            "BLOCK_SIZE",
            "Time (ms)",
            baseline=baseline_ms,
            baseline_label=TORCH_KERNEL,
        )
        warp_panel = Panel(
            f"Best time by warps, {n_cols} columns",
            "num_warps",
            "Time (ms)",
            baseline=baseline_ms,
            baseline_label=TORCH_KERNEL,
        )
        for kernel in ("fused", "online"):
            kernel_rows = [
                r for r in successful(rows) if r["kernel"] == kernel and r["n_cols"] == n_cols
            ]
            if kernel_rows:
                block_panel.series[kernel] = _lowest_per_x(kernel_rows, "block_size")
                warp_panel.series[kernel] = _lowest_per_x(kernel_rows, "num_warps")
        block_panels.append(block_panel)
        warp_panels.append(warp_panel)
    return block_panels, warp_panels


def _lowest_per_x(rows: Sequence[Row], x_key: str) -> list[tuple[float, float]]:
    """Reduce rows to the lowest time at each x value."""
    lowest: dict[float, float] = {}
    for row in rows:
        x, time_ms = float_value(row, x_key), float_value(row, "time_ms")
        lowest[x] = min(time_ms, lowest.get(x, time_ms))
    return list(lowest.items())


def build_summary(rows: Sequence[Row], env: Environment) -> str:
    """Write the softmax section of the summary, with analysis that follows the numbers."""
    table = []
    verdicts = []
    for n_cols in sorted({int(float_value(r, "n_cols")) for r in rows}):
        baseline = _best(rows, TORCH_KERNEL, n_cols)
        fused = _best(rows, "fused", n_cols)
        for row in (baseline, fused, _best(rows, "online", n_cols)):
            if row is not None:
                table.append(
                    (n_cols, row["kernel"], row["block_size"], row["num_warps"],
                     row["time_ms"], row["gb_per_s"])
                )  # fmt: skip
        if baseline and fused:
            verdicts.append(_verdict(n_cols, baseline, fused))
    headers = ("n_cols", "kernel", "BLOCK_SIZE", "num_warps", "time (ms)", "GB/s")
    return "\n\n".join(
        [
            "## Softmax",
            "Measured on: " + "; ".join(env.as_lines()),
            markdown_table(headers, table),
            count_not_timed(rows),
            *verdicts,
        ]
    )


def _verdict(n_cols: int, baseline: Row, fused: Row) -> str:
    """Say plainly which kernel won at this width, and the likely reason."""
    baseline_ms = float_value(baseline, "time_ms")
    fused_ms = float_value(fused, "time_ms")
    if fused_ms < baseline_ms:
        return (
            f"At {n_cols} columns the fused kernel beat torch.softmax by "
            f"{baseline_ms / fused_ms:.2f}x. The likely reason is fewer round trips to global "
            "memory: it does one read and one write per row and keeps everything else in "
            "registers, if PyTorch runs the steps as separate kernels. This run does not "
            "measure memory traffic directly, so treat that as an explanation, not a result."
        )
    return (
        f"At {n_cols} columns torch.softmax beat the fused kernel by "
        f"{fused_ms / baseline_ms:.2f}x. Likely reasons: PyTorch's implementation has been "
        "tuned for this hardware, and this sweep only explores BLOCK_SIZE and num_warps."
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
    shapes = QUICK_SHAPES if args.quick else FULL_SHAPES
    rows = run_sweep(shapes, quick=args.quick, device="cuda", timer=do_bench_timer)

    write_csv(args.output_dir / "softmax.csv", rows, FIELDNAMES, env)
    block_panels, warp_panels = build_panels(rows)
    plot_panels(args.output_dir / "softmax_block_size.png", block_panels)
    plot_panels(args.output_dir / "softmax_num_warps.png", warp_panels)
    update_summary(args.output_dir / "SUMMARY.md", "softmax", build_summary(rows, env))
    print(f"Wrote results to {args.output_dir}. {STATUS_OK}: {len(successful(rows))}")


if __name__ == "__main__":
    main()
