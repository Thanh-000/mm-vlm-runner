"""Pack lightweight SHA-verified CrisisMMD P1-P4 contract bundle (~5.8 MB LZMA)."""

from __future__ import annotations

import argparse
import json
import site
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(ROOT / "src"))

from mnarm.core.io import sha256_bytes  # noqa: E402
from mnarm.data.crisismmd import assert_zero_split_overlap, extract_contracts_from_a2z_zip  # noqa: E402

REQUIRED_FILES = ("manifest.json", "PLANS.json", "partition_contract.json", "splits.zip")
A2Z_PREFIX = "dense_bundle/kit/inputs/"


def pack_contracts_lite(a2z_zip: Path, out_zip: Path) -> Path:
    """Extract the 4 contract files from JCMIL_A2Z_INPUTS.zip and pack via LZMA."""
    if not a2z_zip.is_file():
        raise FileNotFoundError(f"Input A2Z ZIP not found: {a2z_zip}")
    if not zipfile.is_zipfile(a2z_zip):
        raise zipfile.BadZipFile(f"Input is not a valid ZIP: {a2z_zip}")

    blobs: dict[str, bytes] = {}
    shas: dict[str, str] = {}
    with zipfile.ZipFile(a2z_zip) as zf:
        bm = json.loads(zf.read("BUNDLE_MANIFEST.json"))
        for fname in REQUIRED_FILES:
            rel = f"{A2Z_PREFIX}{fname}"
            meta = bm["files"].get(rel)
            if meta is None:
                raise FileNotFoundError(f"Missing {rel} in {a2z_zip}")
            raw = zf.read(f"blobs/{meta['sha256']}")
            actual_sha = sha256_bytes(raw)
            if actual_sha != meta["sha256"]:
                raise ValueError(f"SHA mismatch on {rel}: expected {meta['sha256']}, got {actual_sha}")
            blobs[fname] = raw
            shas[fname] = actual_sha

    manifest = json.loads(blobs["manifest.json"])
    audit = assert_zero_split_overlap(manifest["rows"])
    if not all(v == 0 for v in audit.values()):
        raise ValueError(f"Unexpected split overlap in manifest: {audit}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_zip.with_name(out_zip.name + ".tmp")
    sha_manifest = json.dumps(shas, indent=2, sort_keys=True).encode("utf8")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_LZMA) as out:
        out.writestr("CONTRACT_SHA256.json", sha_manifest)
        for fname in REQUIRED_FILES:
            out.writestr(fname, blobs[fname])
    tmp_zip.replace(out_zip)

    if not zipfile.is_zipfile(out_zip):
        raise zipfile.BadZipFile(f"Round-trip is_zipfile failed on {out_zip}")
    with tempfile.TemporaryDirectory() as tmpd:
        extract_contracts_from_a2z_zip(out_zip, Path(tmpd))
    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack lightweight SHA-verified CrisisMMD P1-P4 contract ZIP.")
    parser.add_argument("--a2z", type=Path, required=True, help="Path to full JCMIL_A2Z_INPUTS.zip.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "colab" / "crisismmd_p1_p4_contracts.zip",
        help="Destination path for lightweight contract ZIP.",
    )
    args = parser.parse_args()
    out_path = pack_contracts_lite(args.a2z, args.out)
    print(f"Built {out_path} ({out_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
