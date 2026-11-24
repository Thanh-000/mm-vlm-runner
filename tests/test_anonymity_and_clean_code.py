"""Automated checks for double-blind anonymity, clean-code limits, and zero forbidden chars."""

from __future__ import annotations

from pathlib import Path


def _forbidden_tokens() -> list[str]:
    return [
        "C:" + "\\Users",
        "C:/" + "Users",
        "One" + "Drive",
        "drive.usercontent." + "google.com",
        "1wOJzIez" + "lrH",
        "Anti" + "gravity",
        "Users/" + "Admin",
        "Users\\" + "Admin",
    ]


def test_anonymity_and_clean_code_gates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forbidden = _forbidden_tokens()

    check_files: list[Path] = []
    for rel_dir in ("src", "tests", "configs", "scripts", "colab"):
        d = repo_root / rel_dir
        if d.is_dir():
            for fp in sorted(d.rglob("*")):
                if fp.is_file() and "__pycache__" not in fp.parts and fp.suffix in (".py", ".yaml", ".yml", ".md", ".ipynb"):
                    check_files.append(fp)
    for top_fn in ("README.md", "LICENSE", "pyproject.toml", "Makefile"):
        fp = repo_root / top_fn
        if fp.is_file():
            check_files.append(fp)

    for fp in check_files:
        text = fp.read_text(encoding="utf8", errors="replace")
        for tok in forbidden:
            assert tok not in text, f"Forbidden identity/path token {tok!r} in {fp.relative_to(repo_root)}"
        if fp.suffix in (".md", ".yaml", ".yml", ".ipynb"):
            assert "$" not in text, f"Forbidden '$' character found in {fp.relative_to(repo_root)}"

    for src_py in sorted((repo_root / "src" / "mnarm").rglob("*.py")):
        lines = src_py.read_text(encoding="utf8").splitlines()
        assert len(lines) <= 350, f"Module exceeds 350 LOC limit ({len(lines)}): {src_py.relative_to(repo_root)}"
        assert "sys.path.insert" not in "\n".join(lines), f"Forbidden sys.path hack in {src_py.relative_to(repo_root)}"
