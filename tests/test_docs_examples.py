"""Run the Python examples in the documentation.

Every block fenced as ``python`` on a page runs, in page order, in one namespace per
page, so later blocks can use what earlier ones defined. Blocks fenced as ``bash``,
``text`` or ``pycon`` are shown but not run. Keeping the examples executable is what
stops the documentation drifting from the code.
"""

import pathlib
import re

import matplotlib
import pytest

matplotlib.use("Agg")

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
PYTHON_BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
PAGES = sorted(
    page
    for page in DOCS.rglob("*.md")
    if PYTHON_BLOCK.search(page.read_text(encoding="utf-8"))
)


def test_documentation_has_examples():
    assert PAGES, f"no Python examples found under {DOCS}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(DOCS).as_posix())
def test_page_examples_run(page, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    # Examples that write files write them into a scratch directory.
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__docs_example__"}
    try:
        for number, block in enumerate(
            PYTHON_BLOCK.findall(page.read_text(encoding="utf-8")), start=1
        ):
            code = compile(block, f"{page.relative_to(DOCS)} (block {number})", "exec")
            exec(code, namespace)
    finally:
        plt.close("all")
