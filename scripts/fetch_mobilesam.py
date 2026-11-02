"""Download the MobileSAM ONNX models the fabric check uses, pinned and hash-checked.

    uv run python scripts/fetch_mobilesam.py

Source: https://huggingface.co/Acly/MobileSAM (MIT licence), an ONNX export of MobileSAM
(https://github.com/ChaoningZhang/MobileSAM). Pinned to one commit, and every file's SHA-256
is checked against the one Hugging Face lists for that commit, so what lands on disk is
exactly what was reviewed - a file that does not match is deleted, not kept.

About 45 MB in total. Files already present with the right hash are left alone.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paths  # noqa: E402

REPOSITORY = "https://huggingface.co/Acly/MobileSAM/resolve/0d3b403339b4674a82493d5e97964dd78089ddc8"

# name, bytes, sha256 - from the repository's file listing at the pinned commit.
FILES = (
    ("mobile_sam_image_encoder.onnx", 28157093, "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749"),
    ("sam_mask_decoder_multi.onnx",   16496559, "8976b90a87ba50a6a72217a5ff994f7d25ce16f2229fcc1ed259e1294c622ffe"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, size: int, expected: str) -> None:
    target = paths.MOBILESAM_DIR / name
    if target.is_file() and sha256(target) == expected:
        print(f"{name}: already here, hash matches")
        return

    partial = target.with_suffix(target.suffix + ".part")
    print(f"{name}: downloading {size / 1e6:.1f} MB from {REPOSITORY}/{name}")
    urllib.request.urlretrieve(f"{REPOSITORY}/{name}", partial)

    found = sha256(partial)
    if found != expected:
        partial.unlink()
        raise RuntimeError(f"{name}: SHA-256 {found} does not match the pinned {expected} - deleted, nothing kept")
    partial.replace(target)
    print(f"{name}: hash verified")


def main() -> None:
    paths.MOBILESAM_DIR.mkdir(parents=True, exist_ok=True)
    for name, size, expected in FILES:
        fetch(name, size, expected)
    print(f"MobileSAM ready in {paths.MOBILESAM_DIR}")


if __name__ == "__main__":
    main()
