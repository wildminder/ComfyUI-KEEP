"""P5.0 - the provenance of the weight manifest (one verified download each).

This host has **no** KEEP weights (P0.2: 0 of 8 present), so the sha256 values
of the manifest cannot come from a local inventory.  They come from a **one
time verified download**: every upstream URL is downloaded exactly once, the
sha256 of the downloaded bytes is computed and cross-checked against the
upstream ``-<8 hex>`` file name convention, and the result is recorded in
``docs/audit/2026-10-22-weight-manifest-sources.md``.

The script is

* **env-gated** - the network is only touched when ``KEEP_FETCH_WEIGHT_HASHES=1``
  (or ``--fetch``) is given; without it the script runs **offline** and renders
  the document from the committed cache.  An offline run without a cache entry
  fails with an actionable message instead of silently inventing a hash;
* **idempotent** - a fetch that finds a verified cache entry re-uses it (no
  second download), and the document is rendered from the cache, so a re-run
  produces byte-identical output;
* **404-safe** - a URL that answers ``404`` is recorded as *unavailable*
  (``sha256: null``, ``unverified: true``) and **no download is attempted**.

The transport is injectable (:class:`HttpTransport`), so the behaviour above is
unit-testable without a network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The machine-readable cache (the state of the one time fetch).
DEFAULT_CACHE = REPO_ROOT / "tools" / "weight_hashes.json"

#: The provenance document the script writes.
DEFAULT_DOC = REPO_ROOT / "docs" / "audit" / "2026-10-22-weight-manifest-sources.md"

#: The environment variable that allows the script to touch the network.
FETCH_ENV = "KEEP_FETCH_WEIGHT_HASHES"

#: The upstream release the weights come from.
RELEASE = "v1.0.0"
RELEASE_BASE = f"https://github.com/jnjaby/KEEP/releases/download/{RELEASE}"

#: The truthy values of :data:`FETCH_ENV`.
TRUTHY = frozenset({"1", "true", "yes", "on"})

#: The upstream file name convention: ``<name>-<8 hex chars>.pth``.
NAME_PREFIX_PATTERN = re.compile(r"[-_]([0-9a-f]{8})\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class WeightSource:
    """One upstream weight file.

    ``filename_prefix`` is the ``-<8 hex>`` part of the upstream file name - the
    upstream convention is that it is the **first 8 hex chars of the sha256** of
    the file.  It is the cross-check of the provenance, not a hash.
    """

    name: str
    dest_folder: str
    subdir: str
    required: bool
    filename_prefix: str | None = None

    @property
    def url(self) -> str:
        return f"{RELEASE_BASE}/{self.name}"


#: The eight manifest entries (the legacy on-disk layout is kept exactly:
#: ``keep_models`` + the ``KEEP`` sub-directory, ``facedetection``, ``gmflow``).
WEIGHT_SOURCES: tuple[WeightSource, ...] = (
    WeightSource("KEEP-b76feb75.pth", "keep_models", "KEEP", True, "b76feb75"),
    # Not published upstream (HTTP 404 - P0.2 §3 / P0.4 §4.1).  The name prefix
    # is the only provenance the file name carries.
    WeightSource("KEEP_Asian-4765ebe0.pth", "keep_models", "KEEP", False, "4765ebe0"),
    WeightSource("detection_Resnet50_Final.pth", "facedetection", "", True),
    WeightSource("detection_mobilenet0.25_Final.pth", "facedetection", "", True),
    WeightSource("yolov5n-face.pth", "facedetection", "", True),
    WeightSource("yolov5l-face.pth", "facedetection", "", True),
    WeightSource("parsing_parsenet.pth", "facedetection", "", True),
    # The GMFlow checkpoint is defensive (P0.4 §2): the KEEP checkpoint carries
    # the trained flownet weights.
    WeightSource("gmflow_sintel-0c07dcb3.pth", "gmflow", "", False, "0c07dcb3"),
)


class FetchError(RuntimeError):
    """Raised when a weight cannot be fetched or verified."""


class HttpTransport:
    """The network transport (stdlib only - no torch, no requests)."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self.user_agent = "ComfyUI-KEEP/2.0 (weight provenance fetch)"

    def head(self, url: str) -> tuple[int, int | None]:
        """``(status, content_length)`` of the URL (404 is not an error)."""
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": self.user_agent}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                length = response.headers.get("Content-Length")
                return response.status, int(length) if length is not None else None
        except urllib.error.HTTPError as exc:
            return exc.code, None

    def download(
        self,
        url: str,
        dest: Path,
        attempts: int = 3,
        expected_size: int | None = None,
    ) -> tuple[str, int]:
        """Download ``url`` to ``dest``; returns ``(sha256, size)``.

        A partial download is kept between the attempts and continued with an
        HTTP ``Range`` request (the KEEP checkpoint is ~634 MB - a restart from
        zero would be expensive).
        """
        part = dest.with_name(dest.name + ".part")
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._download_once(url, part, dest, expected_size)
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                print(
                    f"  attempt {attempt}/{attempts} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt < attempts:
                    time.sleep(2 * attempt)
        raise FetchError(f"the download of {url} failed: {last_error}")

    def _download_once(
        self,
        url: str,
        part: Path,
        dest: Path,
        expected_size: int | None,
    ) -> tuple[str, int]:
        already = part.stat().st_size if part.exists() else 0
        if expected_size is not None and already == expected_size:
            # the partial file is already complete - hash it, do not fetch it again
            return self._finish(part, dest)

        headers = {"User-Agent": self.user_agent}
        if already:
            headers["Range"] = f"bytes={already}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and already:
                # the server refuses the range: the partial file is complete
                return self._finish(part, dest)
            raise
        with response:
            append = already > 0 and response.status == 206
            digest = hashlib.sha256()
            if append:
                with part.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(1 << 20), b""):
                        digest.update(chunk)
            else:
                already = 0
            with part.open("ab" if append else "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
        size = part.stat().st_size
        part.replace(dest)
        return digest.hexdigest(), size

    @staticmethod
    def _finish(part: Path, dest: Path) -> tuple[str, int]:
        """Hash a complete partial file and move it to its destination."""
        digest = hashlib.sha256()
        with part.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        size = part.stat().st_size
        part.replace(dest)
        return digest.hexdigest(), size


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fetch_entry(
    source: WeightSource, transport: HttpTransport, workdir: Path
) -> dict:
    """Fetch one weight and return its record.

    A URL that answers ``404`` is recorded as *unavailable* and **no download
    is attempted** - the file does not exist upstream, so there is nothing to
    hash.
    """
    status, length = transport.head(source.url)
    if status == 404:
        return {
            "name": source.name,
            "url": source.url,
            "dest_folder": source.dest_folder,
            "subdir": source.subdir,
            "required": source.required,
            "available": False,
            "unverified": True,
            "sha256": None,
            "size": None,
            "http_status": 404,
            "filename_prefix": source.filename_prefix,
            "prefix_match": None,
            "fetched_at": _utc_now(),
            "note": "not published upstream (HTTP 404) - no hash is obtainable",
        }
    if status != 200:
        raise FetchError(
            f"the URL {source.url} answered HTTP {status} (expected 200 or 404)"
        )

    target = workdir / source.name
    sha256, size = transport.download(source.url, target, expected_size=length)
    if length is not None and size != length:
        raise FetchError(
            f"the download of {source.url} is incomplete: {size} of {length} bytes"
        )
    if target.exists():
        target.unlink()

    prefix_match: bool | None = None
    if source.filename_prefix:
        prefix_match = sha256.startswith(source.filename_prefix)
        if not prefix_match:
            # Not fatal - the recorded sha256 is the truth - but it is a
            # finding that must be visible.
            print(
                f"WARNING: the sha256 of {source.name} does not start with the "
                f"file name prefix {source.filename_prefix!r} - the upstream "
                "renaming convention does not hold for this asset",
                file=sys.stderr,
            )

    return {
        "name": source.name,
        "url": source.url,
        "dest_folder": source.dest_folder,
        "subdir": source.subdir,
        "required": source.required,
        "available": True,
        "unverified": False,
        "sha256": sha256,
        "size": size,
        "http_status": 200,
        "filename_prefix": source.filename_prefix,
        "prefix_match": prefix_match,
        "fetched_at": _utc_now(),
        "note": "",
    }


def load_cache(path: Path) -> dict:
    """The committed cache (an empty cache when the file does not exist)."""
    if not path.is_file():
        return {"release": RELEASE, "entries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    return data


def write_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_cache(
    fetch: bool,
    cache: dict,
    transport: HttpTransport,
    workdir: Path,
    cache_path: Path | None = None,
) -> dict:
    """The cache with every entry fetched (or taken from the cache).

    The cache is written after **every** entry, so an interrupted run (the
    KEEP checkpoint is ~634 MB) keeps the hashes it already verified.
    """
    entries = dict(cache.get("entries", {}))
    for source in WEIGHT_SOURCES:
        recorded = entries.get(source.name)
        if recorded and recorded.get("url") == source.url:
            if recorded.get("sha256") or recorded.get("http_status") == 404:
                print(f"  {source.name}: cached - no second download")
                continue
        if not fetch:
            raise FetchError(
                f"no cached record for {source.name} and the network is disabled. "
                f"Re-run with {FETCH_ENV}=1 (or --fetch) to download it once."
            )
        print(f"  {source.name}: downloading from {source.url}", flush=True)
        entries[source.name] = fetch_entry(source, transport, workdir)
        if cache_path is not None:
            write_cache(cache_path, {"release": RELEASE, "entries": entries})
    return {"release": RELEASE, "entries": entries}


def render_doc(cache: dict) -> str:
    """The provenance document (deterministic - a function of the cache)."""
    entries = [cache["entries"][source.name] for source in WEIGHT_SOURCES]
    verified = [e for e in entries if e.get("sha256")]
    unavailable = [e for e in entries if not e.get("available")]

    lines: list[str] = []
    lines.append("# P5.0 - Weight manifest provenance: one verified download per weight")
    lines.append("")
    lines.append(
        "- Date of record: 2026-10-22 (plan date) - executed 2026-10-23\n"
        "- Script: `tools/fetch_weight_hashes.py` (env-gated, idempotent)\n"
        "- Machine-readable cache: `tools/weight_hashes.json`\n"
        f"- Upstream release: `jnjaby/KEEP {RELEASE}`"
    )
    lines.append("")
    lines.append("## 1. Method")
    lines.append("")
    lines.append(
        "This host has **no** KEEP weights (P0.2: 0 of 8 present), so the sha256\n"
        "values of the manifest cannot come from a local inventory.  Every URL was\n"
        "therefore downloaded **once**; the sha256 of the downloaded bytes was\n"
        "computed and cross-checked against the upstream file name convention\n"
        "(``-<8 hex chars>`` = the first 8 hex chars of the sha256)."
    )
    lines.append("")
    lines.append(
        "The script is **env-gated** (`KEEP_FETCH_WEIGHT_HASHES=1`), **idempotent**\n"
        "(a verified cache entry is never downloaded twice) and **404-safe** (a URL\n"
        "that answers 404 is recorded as unavailable and *no download is attempted*)."
    )
    lines.append("")
    lines.append(
        f"**Result: {len(verified)} of {len(entries)} weights verified, "
        f"{len(unavailable)} unavailable upstream.**"
    )
    lines.append("")
    lines.append("## 2. The eight entries")
    lines.append("")
    lines.append(
        "| # | File | Destination | Required | HTTP | Size (bytes) | sha256 | "
        "Name prefix | Prefix matches |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for index, entry in enumerate(entries, start=1):
        destination = entry["dest_folder"] + (
            "/" + entry["subdir"] if entry["subdir"] else ""
        )
        sha = entry.get("sha256") or "`null`"
        prefix = entry.get("filename_prefix") or "-"
        match = entry.get("prefix_match")
        if match is None:
            match_text = "-"
        else:
            match_text = "yes" if match else "**no**"
        lines.append(
            f"| {index} | `{entry['name']}` | `{destination}` | "
            f"{'yes' if entry['required'] else 'no'} | {entry.get('http_status')} | "
            f"{entry.get('size') if entry.get('size') is not None else '-'} | "
            f"`{sha}` | `{prefix}` | {match_text} |"
        )
    lines.append("")
    lines.append("## 3. The machine-readable record")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "release": cache.get("release", RELEASE),
                "entries": entries,
                "verified_count": len(verified),
                "unavailable_count": len(unavailable),
                "total_count": len(entries),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("## 4. The unavailable asset")
    lines.append("")
    for entry in unavailable:
        lines.append(
            f"`{entry['name']}` is **not published upstream** (HTTP "
            f"{entry.get('http_status')}; the release API of `jnjaby/KEEP "
            f"{RELEASE}` does not list it - P0.2 section 3 / P0.4 section 4.1).\n"
            "No hash is obtainable, so the manifest entry carries `sha256: null`,\n"
            "`unverified: true` and `available: false`.  The entry is **not**\n"
            "required: a required entry would break the *default* KEEP model too\n"
            "(both checkpoints share the `keep_models/KEEP` destination).  Selecting\n"
            "the `Asian` option raises an actionable error instead (P5.3)."
        )
    lines.append("")
    lines.append("## 5. How the manifest uses this record")
    lines.append("")
    lines.append(
        "* the **7** verified entries carry their `sha256` in `WEIGHT_MANIFEST`\n"
        "  (`src/keep_model_configs.py`) and the downloader verifies every download\n"
        "  against it;\n"
        "* the **Asian** entry carries `sha256: null` + `unverified: true` +\n"
        "  `available: false` + `sha256_prefix: \"4765ebe0\"` (the only provenance the\n"
        "  file name carries) - the downloader **skips the verification** for it and\n"
        "  warns loudly."
    )
    lines.append("")
    return "\n".join(lines)


def write_doc(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"allow the network (same as {FETCH_ENV}=1)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network (the default without the env variable)",
    )
    args = parser.parse_args(argv)

    env_fetch = (os.environ.get(FETCH_ENV) or "").strip().lower() in TRUTHY
    fetch = env_fetch or (args.fetch and not args.offline)
    if args.offline:
        fetch = False

    cache = load_cache(args.cache)
    transport = HttpTransport()
    workdir = Path(tempfile.mkdtemp(prefix="keep-weight-hashes-"))
    try:
        cache = build_cache(fetch, cache, transport, workdir, args.cache)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    write_cache(args.cache, cache)
    write_doc(args.doc, render_doc(cache))
    print(f"wrote {args.cache}")
    print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
