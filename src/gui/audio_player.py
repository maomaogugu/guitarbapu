"""Small Qt Multimedia playback service used by the desktop GUI."""

from pathlib import Path

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer


class AudioPlayer(QObject):
    """Expose file playback in seconds without leaking Qt units to widgets."""

    position_changed = pyqtSignal(float)
    duration_changed = pyqtSignal(float)
    playing_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(
            lambda value: self.position_changed.emit(value / 1000.0)
        )
        self.media_player.durationChanged.connect(
            lambda value: self.duration_changed.emit(value / 1000.0)
        )
        self.media_player.playbackStateChanged.connect(self._state_changed)
        self.media_player.errorOccurred.connect(self._error_changed)
        self.source_path: Path | None = None

    @property
    def has_source(self) -> bool:
        return self.source_path is not None

    @property
    def is_playing(self) -> bool:
        return (
            self.media_player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        )

    @property
    def position(self) -> float:
        return self.media_player.position() / 1000.0

    @property
    def duration(self) -> float:
        return self.media_player.duration() / 1000.0

    def set_source(self, path: str | Path | None) -> None:
        self.stop()
        if path is None:
            self.source_path = None
            self.media_player.setSource(QUrl())
            return
        source = Path(path).expanduser().resolve(strict=False)
        if not source.is_file():
            self.source_path = None
            self.media_player.setSource(QUrl())
            raise FileNotFoundError(f"音频文件不存在：{source}")
        self.source_path = source
        self.media_player.setSource(QUrl.fromLocalFile(str(source)))

    def play(self) -> None:
        if self.has_source:
            self.media_player.play()

    def pause(self) -> None:
        self.media_player.pause()

    def toggle(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        self.media_player.stop()

    def seek(self, seconds: float) -> None:
        if not self.has_source:
            return
        upper = self.duration if self.duration > 0 else float("inf")
        clamped = min(max(0.0, float(seconds)), upper)
        self.media_player.setPosition(round(clamped * 1000))

    def set_volume(self, volume: float) -> None:
        self.audio_output.setVolume(min(1.0, max(0.0, float(volume))))

    def _state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.playing_changed.emit(
            state == QMediaPlayer.PlaybackState.PlayingState
        )

    def _error_changed(self, _error, error_string: str) -> None:
        if error_string:
            self.error_occurred.emit(error_string)


def format_seconds(seconds: float) -> str:
    """Return a stable ``MM:SS.mmm`` label for playback controls."""

    milliseconds = max(0, round(float(seconds) * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


__all__ = ["AudioPlayer", "format_seconds"]
