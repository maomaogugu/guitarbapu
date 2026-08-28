"""Read-only runtime diagnostics for support and packaging verification."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import platform
import sys
from typing import Iterable

from src import __version__

from .model_manager import ModelStatus, OptionalModelManager
from .paths import user_cache_dir, user_data_dir, user_log_dir


@dataclass(frozen=True)
class DiagnosticItem:
    name: str
    status: str
    message: str


@dataclass(frozen=True)
class RuntimeDiagnostics:
    items: tuple[DiagnosticItem, ...]
    model_status: ModelStatus

    @property
    def healthy(self) -> bool:
        return all(item.status != "error" for item in self.items)

    def render(self) -> str:
        lines = [
            f"GuitarBapu {__version__} 系统诊断",
            "=" * 44,
        ]
        for item in self.items:
            lines.append(f"[{item.status.upper():8}] {item.name}: {item.message}")
        lines.extend(
            (
                "",
                "可选模型",
                f"- {self.model_status.spec.name}: {self.model_status.summary}",
                "",
                "说明：诊断报告不包含环境变量、音频内容或项目内容。",
            )
        )
        return "\n".join(lines)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_diagnostics(
    *,
    model_manager: OptionalModelManager | None = None,
    paths: Iterable[Path] | None = None,
) -> RuntimeDiagnostics:
    """Collect deterministic local checks without network access."""

    manager = model_manager or OptionalModelManager()
    items = [
        DiagnosticItem("应用版本", "ok", __version__),
        DiagnosticItem(
            "Python",
            "ok" if sys.version_info >= (3, 10) else "error",
            f"{platform.python_version()} · {sys.executable}",
        ),
        DiagnosticItem("操作系统", "ok", platform.platform()),
        DiagnosticItem(
            "运行模式",
            "ok",
            "安装包" if getattr(sys, "frozen", False) else "源码/开发环境",
        ),
    ]
    for distribution in ("numpy", "librosa", "soundfile", "music21", "PyQt6"):
        version = _version(distribution)
        items.append(
            DiagnosticItem(
                distribution,
                "ok" if version else "error",
                version or "未安装",
            )
        )
    for name in ("torch", "demucs"):
        version = _version(name)
        items.append(
            DiagnosticItem(
                name,
                "ok" if version else "optional",
                version or "未安装",
            )
        )
    checked_paths = tuple(paths or (user_data_dir(), user_cache_dir(), user_log_dir()))
    for path in checked_paths:
        parent = path if path.exists() else path.parent
        available = parent.exists() and parent.is_dir()
        items.append(
            DiagnosticItem(
                f"目录 {path}",
                "ok" if available else "warning",
                "可用" if available else "尚未创建；首次使用时创建",
            )
        )
    return RuntimeDiagnostics(tuple(items), manager.status())


def save_diagnostic_report(
    diagnostics: RuntimeDiagnostics,
    path: str | Path,
) -> Path:
    target = Path(path).expanduser().resolve(strict=False)
    target.write_text(diagnostics.render() + "\n", encoding="utf-8")
    return target


__all__ = [
    "DiagnosticItem",
    "RuntimeDiagnostics",
    "collect_diagnostics",
    "save_diagnostic_report",
]
