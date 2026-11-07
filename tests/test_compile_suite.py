"""Run the compile tests from the main suite, so that a plain `pytest` covers both modes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_kernels_compile_for_gpu_targets() -> None:
    # This needs its own process. See tests_compile/conftest.py for the reason.
    env = {key: value for key, value in os.environ.items() if key != "TRITON_INTERPRET"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_compile", "-q", "--no-header"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
