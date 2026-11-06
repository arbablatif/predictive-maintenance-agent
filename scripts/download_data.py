"""Download the CWRU bearing files and NASA C-MAPSS FD001 into data/raw/ (gitignored).

    python scripts/download_data.py            # both datasets, skips what is already there
    python scripts/download_data.py --only cmapss

Writes data/raw/manifest.json with the size and SHA-256 of every file, so a later
run (or another person) can confirm they are analysing the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pm_agent.config import (CMAPSS_DIR, CMAPSS_FILES, CMAPSS_URL, CWRU_DIR, CWRU_URL,  # noqa: E402
                             DATA_DIR, RECORDINGS)

USER_AGENT = "predictive-maintenance-agent/1.0"
RETRIES = 3
# Certificates are always verified. certifi's fixed root bundle is used because the trust
# store Python reads from Windows failed intermittently for the CWRU host.
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch(url: str) -> bytes:
    """Download one URL with a short retry for transient network errors."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=120, context=TLS_CONTEXT) as response:
                expected = response.headers.get("Content-Length")
                body = response.read()
            if expected and len(body) != int(expected):
                raise OSError(f"truncated download: {len(body)} of {expected} bytes")
            return body
        except OSError as exc:
            if attempt == RETRIES - 1:
                raise
            print(f"  retrying after error: {exc}")
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def download_cwru() -> None:
    CWRU_DIR.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(RECORDINGS, 1):
        path = CWRU_DIR / f"{rec.file_id}.mat"
        if path.exists() and path.stat().st_size > 0:
            continue  # resume: a partial file is never left behind (see the rename below)
        print(f"[{i}/{len(RECORDINGS)}] {rec.file_id}.mat ({rec.name})")
        temp = path.with_suffix(".part")
        temp.write_bytes(fetch(CWRU_URL.format(file_id=rec.file_id)))
        temp.replace(path)


def _find_member(archive: zipfile.ZipFile, name: str) -> str:
    matches = [m for m in archive.namelist() if m.replace("\\", "/").rsplit("/", 1)[-1] == name]
    if not matches:
        raise FileNotFoundError(f"{name} not found in archive")
    return matches[0]


def download_cmapss() -> None:
    """The NASA zip holds a second zip. Only the three FD001 files are extracted, and they
    are written under fixed names, so paths inside the archive are never trusted."""
    CMAPSS_DIR.mkdir(parents=True, exist_ok=True)
    if all((CMAPSS_DIR / f).exists() for f in CMAPSS_FILES):
        return
    print("C-MAPSS archive")
    with zipfile.ZipFile(io.BytesIO(fetch(CMAPSS_URL))) as outer:
        inner_name = next((m for m in outer.namelist() if m.lower().endswith("cmapssdata.zip")), None)
        if inner_name is None:  # some mirrors ship the files without the extra layer
            inner_bytes = None
        else:
            inner_bytes = outer.read(inner_name)
        archive = zipfile.ZipFile(io.BytesIO(inner_bytes)) if inner_bytes else outer
        for name in CMAPSS_FILES:
            (CMAPSS_DIR / name).write_bytes(archive.read(_find_member(archive, name)))


def write_manifest() -> None:
    entries = {}
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries[path.relative_to(DATA_DIR).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (DATA_DIR / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"manifest: {len(entries)} files, {sum(e['bytes'] for e in entries.values()) / 1e6:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("cwru", "cmapss"), help="fetch just one dataset")
    args = parser.parse_args()
    if args.only != "cmapss":
        download_cwru()
    if args.only != "cwru":
        download_cmapss()
    write_manifest()


if __name__ == "__main__":
    main()
