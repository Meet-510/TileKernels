"""Shared benchmark tools: GPU checks, verified timing, CSV output, plots and the summary."""

import csv
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Benchmarks run headless on Colab and Kaggle, so never open a window.
import matplotlib.pyplot as plt
import torch
import triton
from matplotlib.ticker import ScalarFormatter
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources, PTXASError

from tilekernels.utils import is_interpreter_mode

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_WRONG = "verification_failed"

# A configuration can be invalid for reasons known only once it compiles or loads on the
# device, such as needing more shared memory than the GPU has. Those are results to record.
INVALID_CONFIG_ERRORS = (CompilationError, OutOfResources, PTXASError)

# A fixed hue order, so that a series keeps its color across figures. Checked with the
# palette validator for colorblind separation. The baseline is neutral ink, not a series color.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
BASELINE_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"

Row = dict[str, object]
Timer = Callable[[Callable[[], object]], float]


@dataclass(frozen=True)
class Environment:
    """What a result was measured on. Written into every CSV and into the summary."""

    gpu_name: str
    torch_version: str
    triton_version: str
    cuda_version: str
    python_version: str

    def as_lines(self) -> list[str]:
        """Return one `name: value` line per field."""
        return [
            f"gpu: {self.gpu_name}",
            f"torch: {self.torch_version}",
            f"triton: {self.triton_version}",
            f"cuda: {self.cuda_version}",
            f"python: {self.python_version}",
        ]


@dataclass(frozen=True)
class Case:
    """One thing to time, together with the answer it has to produce first."""

    params: Row
    run: Callable[[], torch.Tensor]
    reference: torch.Tensor
    rtol: float
    atol: float


@dataclass(frozen=True)
class Measurement:
    """The outcome of one case. time_ms is None unless the status is STATUS_OK."""

    status: str
    reason: str = ""
    time_ms: float | None = None


@dataclass
class Panel:
    """One subplot: a line per series, plus an optional horizontal baseline."""

    title: str
    xlabel: str
    ylabel: str
    series: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    baseline: float | None = None
    baseline_label: str = ""


def require_compiled_gpu() -> None:
    """Exit with a clear message unless timings from this process would mean something."""
    if is_interpreter_mode():
        raise SystemExit(
            "Refusing to benchmark: TRITON_INTERPRET is set. The interpreter runs kernels one "
            "program at a time on the CPU, so its timings say nothing about the GPU. "
            "Unset TRITON_INTERPRET and run on a machine with a CUDA GPU."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "Refusing to benchmark: torch.cuda.is_available() is False. These benchmarks need "
            "a CUDA GPU. See docs/RUNNING_ON_A_GPU.md for steps on a free Colab or Kaggle GPU."
        )


def read_environment() -> Environment:
    """Describe the current GPU and library versions."""
    return Environment(
        gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        cuda_version=str(torch.version.cuda),
        python_version=platform.python_version(),
    )


def do_bench_timer(fn: Callable[[], object]) -> float:
    """Return the median time of fn in milliseconds, using Triton's own benchmark helper."""
    # Imported here because triton.testing is only meaningful with a GPU present.
    from triton.testing import do_bench

    return float(do_bench(fn, warmup=25, rep=100, return_mode="median"))


def measure(case: Case, timer: Timer) -> Measurement:
    """Run a case once and verify its output. Only time it if the output is right.

    A fast wrong answer must not appear in the results. The first run also triggers
    compilation, so the compile time stays out of the timing.
    """
    try:
        output = case.run()
    except INVALID_CONFIG_ERRORS as error:
        return Measurement(STATUS_SKIPPED, reason=_short_reason(error))
    try:
        torch.testing.assert_close(output, case.reference, rtol=case.rtol, atol=case.atol)
    except AssertionError as error:
        return Measurement(STATUS_WRONG, reason=_short_reason(error))
    return Measurement(STATUS_OK, time_ms=timer(case.run))


def _short_reason(error: Exception) -> str:
    """Compress an error to the last line of its message, which is what explains it."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else type(error).__name__


def write_csv(path: Path, rows: Sequence[Row], fieldnames: Sequence[str], env: Environment) -> None:
    """Write rows to a CSV file whose `#` header lines record the environment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        for line in env.as_lines():
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_panels(path: Path, panels: Sequence[Panel]) -> None:
    """Save side-by-side line charts that share one color for each series label."""
    labels = sorted({label for panel in panels for label in panel.series}, key=_natural_key)
    if len(labels) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} series can be told apart, got {labels}")
    colors = dict(zip(labels, SERIES_COLORS, strict=False))

    figure, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False, layout="constrained"
    )
    for axis, panel in zip(axes[0], panels, strict=True):
        _draw_panel(axis, panel, colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_panel(axis: plt.Axes, panel: Panel, colors: Mapping[str, str]) -> None:
    """Draw one panel: thin lines, visible markers, a recessive grid and a legend."""
    for label, points in sorted(panel.series.items(), key=lambda item: _natural_key(item[0])):
        ordered = sorted(points)
        axis.plot(
            [x for x, _ in ordered],
            [y for _, y in ordered],
            color=colors[label],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            label=label,
        )
    if panel.baseline is not None:
        axis.axhline(
            panel.baseline, color=BASELINE_COLOR, linewidth=2, linestyle="--",
            label=panel.baseline_label,
        )  # fmt: skip
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_title(panel.title, loc="left", fontsize=11)
    axis.set_xlabel(panel.xlabel)
    axis.set_ylabel(panel.ylabel)
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9)


def _natural_key(label: str) -> list[int | str]:
    """Sort labels so that 'K=16' comes before 'K=128'."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label)]


def update_summary(path: Path, section: str, body: str) -> None:
    """Replace one section of the summary file, keeping what the other benchmark wrote.

    Each benchmark script owns a section, so they can run separately in either order.
    """
    start, end = f"<!-- {section} start -->", f"<!-- {section} end -->"
    block = f"{start}\n{body.strip()}\n{end}"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Benchmark summary\n"
    if start in text:
        pattern = re.escape(start) + r".*?" + re.escape(end)
        text = re.sub(pattern, lambda _: block, text, flags=re.DOTALL)
    else:
        text = f"{text.rstrip()}\n\n{block}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def successful(rows: Sequence[Row]) -> list[Row]:
    """Return the rows that passed verification and were timed."""
    return [row for row in rows if row["status"] == STATUS_OK]


def count_not_timed(rows: Sequence[Row]) -> str:
    """Describe how many configurations were skipped or failed verification."""
    skipped = sum(row["status"] == STATUS_SKIPPED for row in rows)
    wrong = sum(row["status"] == STATUS_WRONG for row in rows)
    return (
        f"{len(successful(rows))} configurations were timed, {skipped} were skipped as invalid "
        f"on this GPU, and {wrong} failed verification against PyTorch. "
        "The reason for each is in the CSV."
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Format a small Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def float_value(row: Row, key: str) -> float:
    """Read a numeric field from a row."""
    value = row[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} is not numeric in {row}")
    return float(value)
