"""Track-role labels shared by classifiers, projects, and the GUI."""

from enum import Enum


class TrackRole(str, Enum):
    UNKNOWN = "unknown"
    LEAD = "lead"
    RHYTHM = "rhythm"
    SOLO = "solo"

    @property
    def display_name(self) -> str:
        return {
            TrackRole.UNKNOWN: "未分类",
            TrackRole.LEAD: "主音候选",
            TrackRole.RHYTHM: "节奏候选",
            TrackRole.SOLO: "Solo",
        }[self]


__all__ = ["TrackRole"]
