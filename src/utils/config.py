"""Central application configuration.

Configuration loading (files, environment variables, and user overrides) can
be added later without changing the modules that consume these defaults.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Defaults shared by audio, music, and GUI layers."""

    sample_rate: int = 44_100
    audio_channels: int = 1
    input_device: int | str | None = None


def default_config() -> AppConfig:
    """Return a validated default configuration for local development."""

    return AppConfig()
