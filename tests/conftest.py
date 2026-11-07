"""Pytest configuration: run in Triton's interpreter unless --gpu is given.

Nothing in this file imports triton at module level. Test modules import it during
collection, and pytest_configure runs before collection, so the environment variable
is already set by the time Triton first looks at it.
"""

import os

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --gpu, which runs the same tests compiled on a CUDA device."""
    parser.addoption(
        "--gpu",
        action="store_true",
        help="run kernels compiled on a CUDA GPU instead of in the Triton interpreter",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Choose the execution mode before any test module imports triton."""
    if not config.getoption("--gpu"):
        os.environ["TRITON_INTERPRET"] = "1"
        return
    if not torch.cuda.is_available():
        raise pytest.UsageError("--gpu was given, but torch.cuda.is_available() is False")
    os.environ.pop("TRITON_INTERPRET", None)


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    """The device every test should allocate its tensors on."""
    return "cuda" if request.config.getoption("--gpu") else "cpu"
