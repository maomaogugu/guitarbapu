"""Status and explicit download preparation for optional Demucs models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repository: str
    required_files: tuple[str, ...]
    approximate_size_mb: int


@dataclass(frozen=True)
class ModelStatus:
    spec: ModelSpec
    dependencies_available: bool
    cached: bool
    cached_files: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.dependencies_available and self.cached and self.error is None

    @property
    def summary(self) -> str:
        if self.error:
            return f"状态检查失败：{self.error}"
        if not self.dependencies_available:
            return "未安装 Demucs/PyTorch 可选依赖"
        if self.cached:
            return f"模型已缓存（约 {self.spec.approximate_size_mb} MB）"
        return f"模型尚未下载（约 {self.spec.approximate_size_mb} MB）"


HTDEMUCS_6S = ModelSpec(
    name="htdemucs_6s",
    repository="adefossez/HTDemucs-6s",
    required_files=("htdemucs_6s.yaml", "5c90dfd2.safetensors"),
    approximate_size_mb=52,
)


class OptionalModelManager:
    """Inspect Hugging Face cache and download only after explicit user action."""

    def __init__(
        self,
        spec: ModelSpec = HTDEMUCS_6S,
        *,
        cache_dir: str | Path | None = None,
        downloader: Callable[..., str] | None = None,
    ) -> None:
        self.spec = spec
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._downloader = downloader

    @staticmethod
    def dependencies_available() -> bool:
        return bool(
            importlib.util.find_spec("torch")
            and importlib.util.find_spec("demucs")
            and importlib.util.find_spec("huggingface_hub")
        )

    def _cached_files(self) -> tuple[str, ...]:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound

        try:
            cache = scan_cache_dir(self.cache_dir)
        except CacheNotFound:
            return ()
        for repository in cache.repos:
            if repository.repo_id != self.spec.repository:
                continue
            files = {
                item.file_name
                for revision in repository.revisions
                for item in revision.files
            }
            return tuple(sorted(files))
        return ()

    def status(self) -> ModelStatus:
        dependencies = self.dependencies_available()
        if not dependencies:
            return ModelStatus(self.spec, False, False)
        try:
            files = self._cached_files()
        except Exception as exc:
            return ModelStatus(self.spec, True, False, error=str(exc))
        cached = all(name in files for name in self.spec.required_files)
        return ModelStatus(self.spec, True, cached, files)

    def prepare(self) -> ModelStatus:
        """Download required files and validate that the cache is complete."""

        if not self.dependencies_available():
            raise RuntimeError(
                "未安装 Demucs/PyTorch；请先安装 requirements-separation.txt"
            )
        status = self.status()
        if status.ready:
            return status
        if self._downloader is None:
            from huggingface_hub import snapshot_download

            downloader = snapshot_download
        else:
            downloader = self._downloader
        downloader(
            repo_id=self.spec.repository,
            allow_patterns=list(self.spec.required_files),
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
        )
        status = self.status()
        if not status.ready:
            raise RuntimeError(status.error or "模型下载完成但缓存校验未通过")
        return status


__all__ = [
    "HTDEMUCS_6S",
    "ModelSpec",
    "ModelStatus",
    "OptionalModelManager",
]
