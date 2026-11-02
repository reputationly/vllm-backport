#!/usr/bin/env python3
"""Generate a PEP 503 simple index over a local wheel/sdist directory.

Writes `<dir>/simple/index.html` and `<dir>/simple/<project>/index.html`
with relative links back to the package files in `<dir>`. When
`--cuda-suffix` is given (e.g. `cu130`), it also writes
`<dir>/simple/<cuda-suffix>/<project>/index.html`, so a single base URL
`http://<host>:<port>/simple` can serve both `PIP_INDEX_URL` (trailing
slash) and `PYTORCH_CUDA_INDEX_BASE_URL`/`FLASHINFER_INDEX_BASE_URL`,
which the Dockerfile appends `cuXX` to.
"""

import argparse
import html
import os
import re

_SDIST_SUFFIXES = (".tar.gz", ".zip")


def _package_name(filename: str) -> str:
    if filename.endswith(".whl"):
        stem = filename.split("-")[0]
    else:
        for suffix in _SDIST_SUFFIXES:
            if filename.endswith(suffix):
                stem = re.sub(r"-\d.*$", "", filename[: -len(suffix)])
                break
        else:
            stem = filename
    return re.sub(r"[-_.]+", "-", stem).lower()


def _collect(dirpath: str) -> dict[str, list[str]]:
    pkgs: dict[str, list[str]] = {}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith((".whl", *_SDIST_SUFFIXES)):
            continue
        pkgs.setdefault(_package_name(fname), []).append(fname)
    return pkgs


def _write_index(outdir: str, files: list[str], depth: int) -> None:
    os.makedirs(outdir, exist_ok=True)
    prefix = "../" * depth
    links = "".join(
        f'<a href="{prefix}{html.escape(f.replace("+", "%2B"))}">'
        f"{html.escape(f)}</a><br>\n"
        for f in files
    )
    with open(os.path.join(outdir, "index.html"), "w") as fh:
        fh.write("<html><body>\n" + links + "</body></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", help="directory containing wheels/sdists")
    parser.add_argument(
        "--cuda-suffix",
        metavar="cuNNN",
        help="also mirror the index under simple/<cuNNN>/ for the "
        "PYTORCH/FLASHINFER *_INDEX_BASE_URL args",
    )
    args = parser.parse_args()

    pkgs = _collect(args.dir)
    roots = [(os.path.join(args.dir, "simple"), 2)]
    if args.cuda_suffix:
        roots.append((os.path.join(roots[0][0], args.cuda_suffix), 3))

    for root, depth in roots:
        for name, files in pkgs.items():
            _write_index(os.path.join(root, name), files, depth)
        with open(os.path.join(root, "index.html"), "w") as fh:
            fh.write(
                "<html><body>\n"
                + "".join(f'<a href="{n}/">{n}</a><br>\n' for n in pkgs)
                + "</body></html>\n"
            )

    print(f"{len(pkgs)} projects indexed under {roots[0]}")


if __name__ == "__main__":
    main()
