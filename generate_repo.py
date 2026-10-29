#!/usr/bin/env python3
"""Generate a flat Sileo/Zebra APT repository from pool/*.deb."""
from __future__ import annotations
import bz2
import gzip
import hashlib
import html
import lzma
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
POOL = BASE / "pool"
SITE = BASE / "site"
APT_FILES = ("Packages", "Packages.bz2", "Packages.gz", "Packages.xz")
EXPECTED_ID = "com.anth.pixelguard"
EXPECTED_ARCH = "iphoneos-arm64"
URL = "https://anthh26.github.io/PixelGuardx/"

def run(*args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout.decode("utf-8")

def field(deb, name):
    return run("dpkg-deb", "--field", str(deb), name).strip()

def hashline(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return f" {h.hexdigest()} {path.stat().st_size:16} {path.name}\n"

def main():
    packages = sorted(POOL.glob("*.deb"))
    if not packages:
        print("No package yet; building an empty Sileo APT feed.")
    for deb in packages:
        if field(deb, "Package") != EXPECTED_ID:
            sys.exit(f"Unexpected package identifier: {deb.name}")
        if field(deb, "Architecture") != EXPECTED_ARCH:
            sys.exit(f"Unexpected package architecture: {deb.name}")
        if deb.stat().st_size < 100000:
            sys.exit(f"Possible placeholder .deb refused: {deb.name}")
        version = field(deb, "Version")
        source_path = BASE / "source" / (
            f"PixelGuard-v{version.split('-')[0].rsplit('.',1)[0]}-source.zip"
        )
        if not source_path.is_file():
            sys.exit(f"Missing matching source archive: {source_path}")
        print(f"Indexing: {deb.name} version {version}")

    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "pool").mkdir(parents=True)
    (SITE / "source").mkdir(parents=True)
    for deb in packages:
        shutil.copy2(deb, SITE / "pool" / deb.name)
    for archive in (BASE / "source").glob("*.zip"):
        shutil.copy2(archive, SITE / "source" / archive.name)

    packages_text = run("dpkg-scanpackages", "-m", "pool", "/dev/null") if packages else ""
    if packages and f"Package: {EXPECTED_ID}" not in packages_text:
        sys.exit("Package metadata generation failed.")
    blob = packages_text.encode("utf-8")
    (SITE / "Packages").write_bytes(blob)
    (SITE / "Packages.bz2").write_bytes(bz2.compress(blob, compresslevel=9))
    with (SITE / "Packages.gz").open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", compresslevel=9, mtime=0) as z:
            z.write(blob)
    (SITE / "Packages.xz").write_bytes(lzma.compress(blob, preset=6))

    headers = [
        "Origin: PixelGuardx", "Label: PixelGuardx",
        "Suite: stable", "Codename: ios",
        "Architectures: iphoneos-arm64", "Components: main",
        "Description: Experimental AI image pixelation for Dopamine rootless.",
        "Date: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
    ]
    for algorithm, title in [("md5", "MD5Sum"), ("sha256", "SHA256"), ("sha512", "SHA512")]:
        headers.append(title + ":\n" + "".join(
            hashline(SITE / name, algorithm) for name in APT_FILES
        ).rstrip("\n"))
    (SITE / "Release").write_text("\n".join(headers) + "\n", encoding="utf-8")

    latest = html.escape(field(packages[-1], "Version")) if packages else "Not published yet"
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PixelGuardx</title><style>body{font:18px system-ui;background:#111827;color:#f3f4f6;max-width:680px;margin:8vh auto;padding:22px;line-height:1.5}a{color:#93c5fd}code{word-break:break-all}</style></head><body><h1>PixelGuardx</h1><p>Sileo APT source for experimental PixelGuard AI.</p><p>Latest version: <code>__VERSION__</code></p><p>Add this in Sileo → Sources → +</p><p><code>__URL__</code></p><p><a href="sileo://source/__URL__">Open in Sileo</a></p><p>Compiled packages and <a href="./source/">corresponding source archives</a> are hosted here. Sileo offers updates when refreshed; it does not silently install packages.</p><p>This tweak is not reliable across all applications, videos or image renderers. This repository is unsigned and should be treated as experimental.</p><p>Model attribution: <a href="https://github.com/notAI-tech/NudeNet">NudeNet (AGPL-3.0)</a>.</p></body></html>"""
    (SITE / "index.html").write_text(
        page.replace("__VERSION__", latest).replace("__URL__", URL), encoding="utf-8"
    )
    (SITE / ".nojekyll").write_text("")
    print("Ready for GitHub Pages:", URL)

if __name__ == "__main__":
    main()
