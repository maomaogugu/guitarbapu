"""Content-addressed disk cache for separated audio stems."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

from .separator import SeparationError, SeparationResult, Stem


def default_separation_cache_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "GuitarBapu" / "separation"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "GuitarBapu" / "Cache" / "separation"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "guitarbapu" / "separation"


class SeparationCache:
    """Store stems outside the repository and reuse them by audio content."""

    METADATA_NAME = "metadata.json"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or default_separation_cache_root()).expanduser()

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def key_for(
        self,
        source_path: str | Path,
        configuration: Mapping[str, Any],
    ) -> str:
        source = Path(source_path).expanduser().resolve(strict=True)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(
            json.dumps(
                dict(configuration),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def entry_path(self, cache_key: str) -> Path:
        if len(cache_key) != 64 or any(
            character not in "0123456789abcdef" for character in cache_key
        ):
            raise ValueError("cache_key must be a lowercase SHA-256 hex digest")
        return self.root / cache_key

    def load(self, cache_key: str) -> SeparationResult | None:
        entry = self.entry_path(cache_key)
        metadata_path = entry / self.METADATA_NAME
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stems = tuple(
                Stem(
                    name=str(item["name"]),
                    path=(entry / str(item["file"])).resolve(strict=True),
                    sample_rate=int(item["sample_rate"]),
                    channels=int(item["channels"]),
                    duration=float(item["duration"]),
                )
                for item in metadata["stems"]
            )
            return SeparationResult(
                source_path=Path(metadata["source_path"]),
                model_name=str(metadata["model_name"]),
                device=str(metadata["device"]),
                cache_key=cache_key,
                stems=stems,
                from_cache=True,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def create_work_directory(self, cache_key: str) -> Path:
        root = self.ensure_root()
        return Path(tempfile.mkdtemp(prefix=f".{cache_key[:12]}-", dir=root))

    def commit(
        self,
        cache_key: str,
        work_directory: str | Path,
        *,
        source_path: str | Path,
        model_name: str,
        device: str,
        stems: tuple[Stem, ...],
    ) -> SeparationResult:
        work = Path(work_directory).resolve(strict=True)
        root = self.ensure_root().resolve(strict=True)
        if work.parent != root or not work.name.startswith(f".{cache_key[:12]}-"):
            raise SeparationError("拒绝提交缓存目录之外的临时分离结果")
        metadata = {
            "source_path": str(Path(source_path).expanduser().resolve(strict=False)),
            "model_name": model_name,
            "device": device,
            "stems": [
                {
                    "name": stem.name,
                    "file": stem.path.name,
                    "sample_rate": stem.sample_rate,
                    "channels": stem.channels,
                    "duration": stem.duration,
                }
                for stem in stems
            ],
        }
        (work / self.METADATA_NAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        target = self.entry_path(cache_key)
        if target.exists():
            cached = self.load(cache_key)
            if cached is not None:
                shutil.rmtree(work, ignore_errors=True)
                return cached
            # A previous crashed commit can leave an unreadable entry behind.
            # Remove that incomplete entry instead of blocking this audio forever.
            shutil.rmtree(target, ignore_errors=True)
        try:
            work.replace(target)
        except OSError as exc:
            cached = self.load(cache_key)
            if cached is not None:
                shutil.rmtree(work, ignore_errors=True)
                return cached
            if target.exists():
                # Windows refuses to replace an existing directory; if the winner
                # is still unreadable, clear it and retry this commit once.
                shutil.rmtree(target, ignore_errors=True)
                try:
                    work.replace(target)
                except OSError as retry_exc:
                    raise SeparationError("分离缓存提交失败") from retry_exc
            else:
                shutil.rmtree(work, ignore_errors=True)
                raise SeparationError("分离缓存提交失败") from exc
        loaded = self.load(cache_key)
        if loaded is None:
            raise SeparationError("分离缓存提交后无法读取")
        return SeparationResult(
            source_path=loaded.source_path,
            model_name=loaded.model_name,
            device=loaded.device,
            cache_key=loaded.cache_key,
            stems=loaded.stems,
            from_cache=False,
        )

    @staticmethod
    def discard_work_directory(work_directory: str | Path) -> None:
        work = Path(work_directory)
        if work.exists() and work.name.startswith("."):
            shutil.rmtree(work)


__all__ = ["SeparationCache", "default_separation_cache_root"]
