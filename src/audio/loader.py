"""Audio file loading and the common decoded-audio data structure."""

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class AudioData:
    """Decoded waveform and metadata shared by downstream audio stages."""

    waveform: object
    sample_rate: int
    duration: float
    channels: int


def load_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
) -> AudioData:
    """Decode an MP3, WAV, or FLAC file with ``soundfile``.

    ``soundfile`` preserves the source sample rate in this stage. The
    ``sample_rate`` argument is accepted for API compatibility and must be
    omitted or match the source rate; resampling belongs to a later pipeline
    stage. With ``mono=False``, the waveform retains one column per channel.
    """

    audio_path = Path(path)
    supported_formats = {".mp3", ".wav", ".flac"}
    if audio_path.suffix.lower() not in supported_formats:
        supported = ", ".join(sorted(supported_formats))
        raise ValueError(f"Unsupported audio format; expected one of {supported}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    waveform, actual_sample_rate = sf.read(
        str(audio_path),
        always_2d=True,
        dtype="float32",
    )
    channel_count = waveform.shape[1]
    if mono and channel_count > 1:
        waveform = waveform.mean(axis=1)
        channel_count = 1
    elif mono:
        waveform = waveform[:, 0]

    if sample_rate is not None and sample_rate != actual_sample_rate:
        raise ValueError(
            "Resampling is not implemented; sample_rate must match the source rate"
        )

    frame_count = waveform.shape[0]
    duration = frame_count / actual_sample_rate if actual_sample_rate else 0.0

    return AudioData(
        waveform=waveform,
        sample_rate=int(actual_sample_rate),
        duration=float(duration),
        channels=int(channel_count),
    )
