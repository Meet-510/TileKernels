"""Keep the documentation honest: excerpts must exist, and results must not be invented."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = sorted(
    [REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md", *(REPO_ROOT / "docs").glob("*.md")]
)

# An excerpt is a fenced block, directly after a comment that names the file it came from.
EXCERPT = re.compile(r"<!-- excerpt: (\S+) -->\n```text\n(.*?)\n```", re.DOTALL)
LINK = re.compile(r"\]\((?!https?://|#)([^)#]+)")


def _excerpts() -> list[tuple[str, str]]:
    text = (REPO_ROOT / "docs" / "COMPILER_DUMPS.md").read_text(encoding="utf-8")
    return [
        (source, line.strip())
        for source, block in EXCERPT.findall(text)
        for line in block.splitlines()
        if line.strip()
    ]


def test_the_walkthrough_has_excerpts() -> None:
    assert len(_excerpts()) > 20


@pytest.mark.parametrize(("source", "line"), _excerpts())
def test_each_excerpt_line_is_in_its_dump(source: str, line: str) -> None:
    # A substring match, because excerpts may leave off the trailing debug location.
    assert line in (REPO_ROOT / source).read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda path: path.name)
def test_relative_links_point_at_real_files(doc: Path) -> None:
    for target in LINK.findall(doc.read_text(encoding="utf-8")):
        assert (doc.parent / target).exists(), f"{doc.name} links to a missing file: {target}"


def test_readme_quickstart_runs() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    snippet = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert snippet is not None
    result = subprocess.run(
        [sys.executable, "-c", snippet.group(1)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_results_are_marked_pending_until_measured() -> None:
    # There must be no results section with numbers until a real GPU run has produced them.
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    measured = (REPO_ROOT / "results" / "softmax.csv").exists()
    assert measured or "Pending: run on a GPU to fill this in" in readme


def test_dumps_contain_no_local_paths() -> None:
    for path in (REPO_ROOT / "dumps").rglob("*"):
        if path.is_file() and path.suffix != ".cubin":
            text = path.read_text(encoding="utf-8")
            assert "site-packages" not in text, path
            assert not re.search(r"[A-Za-z]:[\\/]+Users", text), path
