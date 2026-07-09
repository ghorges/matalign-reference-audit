from __future__ import annotations

from hashlib import md5
import json
import bz2
import gzip
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

from .config import Settings
from .io_utils import write_json


FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
USER_AGENT = "materials-fairness-audit/0.1"


def normalize_download_url(url: str) -> str:
    if "figshare.com/files/" in url:
        file_id = url.rstrip("/").split("/")[-1]
        return f"https://ndownloader.figshare.com/files/{file_id}"
    return url


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_figshare_article(article_id: int) -> dict[str, Any]:
    return fetch_json(FIGSHARE_API.format(article_id=article_id))


def save_figshare_manifest(article_id: int, output_dir: Path) -> Path:
    metadata = fetch_figshare_article(article_id)
    manifest_path = output_dir / f"figshare_article_{article_id}.json"
    write_json(metadata, manifest_path)
    return manifest_path


def _eligible(name: str, suffixes: tuple[str, ...]) -> bool:
    lower_name = name.lower()
    return any(lower_name.endswith(suffix.lower()) for suffix in suffixes)


def select_article_files(article: dict[str, Any], suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    return [item for item in article.get("files", []) if _eligible(item["name"], suffixes)]


def download_file(url: str, destination: Path, expected_md5: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_md5 is not None and hash_file(destination) == expected_md5:
            return destination
        if expected_md5 is None and file_is_readable(destination):
            return destination

    resolved_url = normalize_download_url(url)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        try:
            request = Request(resolved_url, headers={"User-Agent": USER_AGENT})
            with urlopen(request) as response, tmp_path.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            os.replace(tmp_path, destination)
            break
        except (ConnectionResetError, TimeoutError, URLError):
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))

    if expected_md5 and hash_file(destination) != expected_md5:
        raise ValueError(f"Checksum mismatch for {destination}")
    if expected_md5 is None and not file_is_readable(destination):
        raise ValueError(f"Unreadable download for {destination}")
    return destination


def hash_file(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_is_readable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    suffixes = "".join(path.suffixes).lower()
    try:
        if suffixes.endswith(".csv.gz") or suffixes.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                while handle.read(1024 * 1024):
                    pass
            return True
        if suffixes.endswith(".csv.bz2") or suffixes.endswith(".json.bz2"):
            with bz2.open(path, "rt", encoding="utf-8") as handle:
                while handle.read(1024 * 1024):
                    pass
            return True
        if suffixes.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                return archive.testzip() is None
        with path.open("rb") as handle:
            return bool(handle.read(16))
    except Exception:
        return False


def clone_or_update_repo(settings: Settings) -> Path:
    repo_dir = settings.paths.official_repo
    if repo_dir.exists() and (repo_dir / ".git").exists():
        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", settings.phase0.matbench_repo_ref],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", settings.phase0.matbench_repo_ref],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", settings.phase0.matbench_repo_ref],
                check=True,
            )
        except subprocess.CalledProcessError:
            return repo_dir
        return repo_dir

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            settings.phase0.matbench_repo_ref,
            settings.phase0.matbench_repo_url,
            str(repo_dir),
        ],
        check=True,
    )
    return repo_dir
