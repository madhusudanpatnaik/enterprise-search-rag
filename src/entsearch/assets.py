"""Everything this project downloads, in one list — fetched on the machine that runs it, never on
import, in tests, or while building.

`entsearch download --list` shows each model, its size and what needs it; `entsearch download
[NAME ...]` fetches them into the Hugging Face cache (`HF_HOME`, or `--cache-dir`), which
sentence-transformers reads, so later runs start without a download. `entsearch download --beir
scifact` fetches a BEIR benchmark into `data/beir/` in the layout `entsearch eval --beir` reads.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Asset:
    name: str
    repo_id: str
    size: str  # approximate, for planning disk
    needed_by: str


ASSETS: tuple[Asset, ...] = (
    Asset("encoder", "BAAI/bge-small-en-v1.5", "~130 MB", "--encoder st"),
    Asset("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", "~90 MB", "--rerank cross"),
    Asset("nli", "cross-encoder/nli-deberta-v3-base", "~0.7 GB", "--verify nli"),
    Asset("minicheck", "lytang/MiniCheck-Flan-T5-Large", "~3 GB", "--verify minicheck"),
)


def find(names: Sequence[str]) -> list[Asset]:
    """The named assets (all of them when no name is given)."""
    if not names:
        return list(ASSETS)
    by_name = {a.name: a for a in ASSETS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown asset(s) {unknown}; choose from {sorted(by_name)}")
    return [by_name[n] for n in names]


def describe(assets: Sequence[Asset]) -> list[str]:
    rows = [f"{'name':<10} {'size':<8} {'repository':<42} needed by"]
    rows += [f"{a.name:<10} {a.size:<8} {a.repo_id:<42} {a.needed_by}" for a in assets]
    return rows


def download(
    assets: Sequence[Asset],
    *,
    cache_dir: str | None = None,
    fetch_model: Callable[..., Any] | None = None,
) -> list[str]:
    if fetch_model is None:
        from huggingface_hub import snapshot_download

        fetch_model = snapshot_download
    return [
        f"{a.name}: {fetch_model(repo_id=a.repo_id, repo_type='model', cache_dir=cache_dir)}"
        for a in assets
    ]


# --------------------------------------------------------------------------- BEIR

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def _fetch_to(url: str, path: Path) -> None:
    with urllib.request.urlopen(url) as response, path.open("wb") as out:  # noqa: S310 - fixed host
        shutil.copyfileobj(response, out)


def download_beir(
    name: str, dest: str | Path = "data/beir", *, fetch: Callable[[str, Path], None] = _fetch_to
) -> Path:
    """Fetch and unpack one BEIR dataset (corpus.jsonl, queries.jsonl, qrels/). Skips the
    download when it is already there. Archive members that would land outside `dest` are
    refused rather than extracted."""
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"not a BEIR dataset name: {name!r}")
    root = Path(dest).resolve()
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / f"{name}.zip"
        fetch(BEIR_URL.format(name=name), archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise ValueError(f"refusing archive member outside {root}: {member!r}")
            zf.extractall(root)
    if not (target / "corpus.jsonl").exists():
        raise ValueError(f"{name}.zip did not contain {name}/corpus.jsonl")
    return target
