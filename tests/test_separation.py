"""Tests for content-addressed separation and the optional Demucs backend."""

from pathlib import Path
from threading import Event

import numpy as np
import pytest

from src.audio.demucs_separator import DemucsConfig, DemucsSeparator
from src.audio.separation_cache import SeparationCache
from src.audio.separator import SeparationCancelled, SeparationError


class _FakeModel:
    sources = ("drums", "bass", "other", "vocals", "guitar", "piano")


class _FakeDemucsApi:
    samplerate = 44_100
    audio_channels = 2
    model = _FakeModel()

    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")

    def separate_audio_file(self, _source: Path):
        if self.callback is not None:
            self.callback(
                {
                    "audio_length": 100,
                    "segment_offset": 50,
                    "state": "start",
                    "models": 1,
                    "model_idx_in_bag": 0,
                    "shift_idx": 0,
                }
            )
        return np.zeros((2, 100), dtype=np.float32), {
            "guitar": np.linspace(-0.5, 0.5, 200, dtype=np.float32).reshape(2, 100)
        }


class _ReusableFakeDemucsApi(_FakeDemucsApi):
    def update_parameter(self, *, callback=None, **_kwargs):
        self.callback = callback


def _source(tmp_path: Path, content: bytes = b"audio") -> Path:
    source = tmp_path / "source.wav"
    source.write_bytes(content)
    return source


def test_cache_key_uses_audio_content_and_configuration(tmp_path):
    cache = SeparationCache(tmp_path / "cache")
    source = _source(tmp_path)

    first = cache.key_for(source, {"model": "a"})
    second = cache.key_for(source, {"model": "b"})
    source.write_bytes(b"changed")
    third = cache.key_for(source, {"model": "a"})

    assert len(first) == 64
    assert first != second
    assert first != third


def test_demucs_backend_writes_guitar_stem_and_reuses_cache(tmp_path):
    source = _source(tmp_path)
    cache = SeparationCache(tmp_path / "cache")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _FakeDemucsApi(**kwargs)

    progress = []
    separator = DemucsSeparator(cache=cache, api_factory=factory)

    first = separator.separate(source, progress_callback=progress.append)
    second = separator.separate(source, progress_callback=progress.append)

    guitar = first.stem("guitar")
    assert guitar.path.is_file()
    assert guitar.sample_rate == 44_100
    assert guitar.channels == 2
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.stem("guitar").path == guitar.path
    assert len(calls) == 1
    assert any(item.stage == "separating" for item in progress)
    assert progress[-1].stage == "cached"


def test_demucs_backend_requires_real_guitar_source(tmp_path):
    class NoGuitarApi(_FakeDemucsApi):
        class Model:
            sources = ("drums", "bass", "vocals", "other")

        model = Model()

    separator = DemucsSeparator(
        cache=SeparationCache(tmp_path / "cache"),
        api_factory=lambda **kwargs: NoGuitarApi(**kwargs),
    )

    with pytest.raises(SeparationError, match="不包含 stem"):
        separator.separate(_source(tmp_path))


def test_demucs_backend_honors_cancel_before_work(tmp_path):
    cancelled = Event()
    cancelled.set()
    separator = DemucsSeparator(
        cache=SeparationCache(tmp_path / "cache"),
        api_factory=lambda **kwargs: _FakeDemucsApi(**kwargs),
    )

    with pytest.raises(SeparationCancelled):
        separator.separate(_source(tmp_path), cancel_event=cancelled)


def test_explicit_wrong_stem_position_is_not_silently_substituted(tmp_path):
    separator = DemucsSeparator(
        DemucsConfig(requested_stems=("lead_guitar",)),
        cache=SeparationCache(tmp_path / "cache"),
        api_factory=lambda **kwargs: _FakeDemucsApi(**kwargs),
    )

    with pytest.raises(SeparationError, match="lead_guitar"):
        separator.separate(_source(tmp_path))


def test_reused_backend_refreshes_progress_and_cancel_callback(tmp_path):
    first_source = _source(tmp_path, b"first")
    second_source = tmp_path / "second.wav"
    second_source.write_bytes(b"second")
    cache = SeparationCache(tmp_path / "cache")
    api = _ReusableFakeDemucsApi()
    separator = DemucsSeparator(cache=cache, api_factory=lambda **_kwargs: api)
    first_progress = []
    separator.separate(first_source, progress_callback=first_progress.append)
    first_count = len(first_progress)
    cancelled = Event()

    def cancel_second_job(progress):
        if progress.stage == "separating":
            cancelled.set()

    with pytest.raises(SeparationCancelled):
        separator.separate(
            second_source,
            progress_callback=cancel_second_job,
            cancel_event=cancelled,
        )

    assert cancelled.is_set()
    assert len(first_progress) == first_count
