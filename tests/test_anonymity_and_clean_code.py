"""Automated checks for double-blind anonymity, clean-code limits, and zero forbidden chars."""

from __future__ import annotations

import zipfile
from pathlib import Path


def _machine_identity_tokens() -> list[str]:
    return [
        "C:" + "\\Users",
        "C:/" + "Users",
        "One" + "Drive",
        "Anti" + "gravity",
        "Users/" + "Admin",
        "Users\\" + "Admin",
    ]


def _drive_url_tokens() -> list[str]:
    return [
        "drive.usercontent." + "google.com",
        "1wOJzIez" + "lrH",
        "1_1Bn0d" + "HKqZ",
        "1WUMyBQ" + "68cM",
        "1oknnyM" + "NKCXX",
    ]


def test_anonymity_and_clean_code_gates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    id_tokens = _machine_identity_tokens()
    drive_tokens = _drive_url_tokens()

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
        rel_p = fp.relative_to(repo_root)
        for tok in id_tokens:
            assert tok not in text, f"Forbidden identity/path token {tok!r} in {rel_p}"
        if rel_p.parts[0] not in ("colab", "scripts"):
            for tok in drive_tokens:
                assert tok not in text, f"Forbidden Drive token {tok!r} in core file {rel_p}"
        if fp.suffix in (".md", ".yaml", ".yml", ".ipynb"):
            assert "$" not in text, f"Forbidden '$' character found in {rel_p}"

    for src_py in sorted((repo_root / "src" / "mnarm").rglob("*.py")):
        lines = src_py.read_text(encoding="utf8").splitlines()
        assert len(lines) <= 350, f"Module exceeds 350 LOC limit ({len(lines)}): {src_py.relative_to(repo_root)}"
        assert "sys.path.insert" not in "\n".join(lines), f"Forbidden sys.path hack in {src_py.relative_to(repo_root)}"

    zp = repo_root / "colab" / "mnarm_iccv_same_bank_code.zip"
    if zp.is_file():
        with zipfile.ZipFile(zp) as zf:
            znames = set(zf.namelist())
        for src_py in sorted((repo_root / "src" / "mnarm").rglob("*.py")):
            rel = src_py.relative_to(repo_root).as_posix()
            assert rel in znames, f"Stale Colab zip missing {rel}; rebuild via build_iccv_same_bank_colab.py"
