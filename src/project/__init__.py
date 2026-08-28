"""Versioned GuitarBapu project persistence."""

from .model import TranscriptionProject
from .track import TranscriptionTrack
from .serializer import (
    CURRENT_SCHEMA_VERSION,
    ProjectFormatError,
    load_project,
    project_to_dict,
    save_project,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ProjectFormatError",
    "TranscriptionProject",
    "TranscriptionTrack",
    "load_project",
    "project_to_dict",
    "save_project",
]
