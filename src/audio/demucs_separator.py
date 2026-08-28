"""Optional Demucs implementation of the source-separation contract."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .separation_cache import SeparationCache
from .separator import (
    ProgressCallback,
    SeparationCancelled,
    SeparationError,
    SeparationProgress,
    SeparationResult,
    Stem,
)


@dataclass(frozen=True)
class DemucsConfig:
    model_name: str = "htdemucs_6s"
    requested_stems: tuple[str, ...] = ("guitar",)
    device: str = "auto"
    shifts: int = 0
    overlap: float = 0.25
    segment: int | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.requested_stems:
            raise ValueError("requested_stems must not be empty")
        if self.shifts < 0:
            raise ValueError("shifts must be non-negative")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be between 0 and 1")


class DemucsSeparator:
    """Run a guitar-capable Demucs model and cache requested WAV stems."""

    def __init__(
        self,
        config: DemucsConfig | None = None,
        *,
        cache: SeparationCache | None = None,
        api_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or DemucsConfig()
        self.cache = cache or SeparationCache()
        self._api_factory = api_factory
        self._separator = None
        self._device: str | None = None

    @staticmethod
    def is_available() -> bool:
        return bool(
            importlib.util.find_spec("torch")
            and importlib.util.find_spec("demucs")
        )

    @staticmethod
    def available_device() -> str:
        if not DemucsSeparator.is_available():
            return "cpu"
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _resolved_device(self) -> str:
        return (
            self.available_device()
            if self.config.device == "auto"
            else self.config.device
        )

    def _configuration(self) -> dict[str, Any]:
        version = "unavailable"
        if self.is_available():
            import demucs

            version = demucs.__version__
        return {
            "backend": "demucs",
            "backend_version": version,
            "model_name": self.config.model_name,
            "requested_stems": list(self.config.requested_stems),
            "shifts": self.config.shifts,
            "overlap": self.config.overlap,
            "segment": self.config.segment,
        }

    def _emit(
        self,
        callback: ProgressCallback | None,
        stage: str,
        fraction: float | None,
        message: str,
    ) -> None:
        if callback is not None:
            callback(SeparationProgress(stage, fraction, message))

    @staticmethod
    def _check_cancel(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SeparationCancelled("音源分离已取消")

    def _load_api(
        self,
        *,
        progress_callback: ProgressCallback | None,
        cancel_event: Event | None,
    ):
        def demucs_callback(state: dict) -> None:
            self._check_cancel(cancel_event)
            audio_length = max(1, int(state.get("audio_length", 1)))
            offset = max(0, int(state.get("segment_offset", 0)))
            fraction = min(0.98, offset / audio_length)
            self._emit(
                progress_callback,
                "separating",
                fraction,
                "正在分离吉他音轨…",
            )

        if self._separator is not None:
            update_parameter = getattr(self._separator, "update_parameter", None)
            if update_parameter is None:
                # Third-party/test factories may not support callback updates.
                # Recreate them so a later job never inherits stale cancel state.
                self._separator = None
            else:
                update_parameter(callback=demucs_callback)
                return self._separator

        if self._api_factory is None:
            if not self.is_available():
                raise SeparationError(
                    "未安装可选的 Demucs/PyTorch；请安装 requirements-separation.txt"
                )
            from demucs.api import Separator as DemucsApiSeparator

            factory = DemucsApiSeparator
        else:
            factory = self._api_factory
        self._device = self._resolved_device()

        self._emit(
            progress_callback,
            "loading_model",
            None,
            f"正在加载 {self.config.model_name} 模型…",
        )
        try:
            self._separator = factory(
                model=self.config.model_name,
                device=self._device,
                shifts=self.config.shifts,
                overlap=self.config.overlap,
                split=True,
                segment=self.config.segment,
                jobs=0,
                progress=False,
                callback=demucs_callback,
            )
        except SeparationCancelled:
            raise
        except Exception as exc:
            raise SeparationError(f"无法加载 Demucs 模型：{exc}") from exc
        sources = tuple(self._separator.model.sources)
        missing = [name for name in self.config.requested_stems if name not in sources]
        if missing:
            raise SeparationError(
                f"模型 {self.config.model_name} 不包含 stem：{', '.join(missing)}"
            )
        return self._separator

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32)

    def separate(
        self,
        source_path: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> SeparationResult:
        source = Path(source_path).expanduser().resolve(strict=True)
        self._check_cancel(cancel_event)
        self._emit(progress_callback, "hashing", 0.0, "正在检查分离缓存…")
        cache_key = self.cache.key_for(source, self._configuration())
        cached = self.cache.load(cache_key)
        if cached is not None:
            self._emit(progress_callback, "cached", 1.0, "已复用吉他分离缓存")
            return cached

        separator = self._load_api(
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        self._check_cancel(cancel_event)
        work = self.cache.create_work_directory(cache_key)
        try:
            try:
                _, separated = separator.separate_audio_file(source)
            except KeyboardInterrupt as exc:
                raise SeparationCancelled("音源分离已取消") from exc
            except SeparationCancelled:
                raise
            except Exception as exc:
                raise SeparationError(f"Demucs 分离失败：{exc}") from exc
            self._check_cancel(cancel_event)
            stems: list[Stem] = []
            for name in self.config.requested_stems:
                if name not in separated:
                    raise SeparationError(f"分离结果缺少 {name!r} stem")
                samples = self._to_numpy(separated[name])
                if samples.ndim != 2:
                    raise SeparationError("Demucs stem 必须是 channels × samples")
                output = work / f"{name}.wav"
                sf.write(output, samples.T, separator.samplerate, subtype="PCM_16")
                info = sf.info(output)
                stems.append(
                    Stem(
                        name=name,
                        path=output,
                        sample_rate=info.samplerate,
                        channels=info.channels,
                        duration=info.duration,
                    )
                )
            self._emit(progress_callback, "saving", 0.99, "正在保存吉他 stem…")
            result = self.cache.commit(
                cache_key,
                work,
                source_path=source,
                model_name=self.config.model_name,
                device=self._device or "cpu",
                stems=tuple(stems),
            )
            self._emit(progress_callback, "complete", 1.0, "吉他音轨分离完成")
            return result
        except BaseException:
            self.cache.discard_work_directory(work)
            raise


__all__ = ["DemucsConfig", "DemucsSeparator"]
