"""Tests for Phase 9D paths, logs, diagnostics, and model management."""

import logging

from src.utils.diagnostics import collect_diagnostics, save_diagnostic_report
from src.utils.logger import configure_logging, get_logger, write_crash_report
from src.utils.model_manager import (
    HTDEMUCS_6S,
    ModelStatus,
    OptionalModelManager,
)
from src.utils.paths import (
    crash_report_dir,
    user_cache_dir,
    user_data_dir,
    user_log_dir,
)


def test_cross_platform_user_paths_are_deterministic(tmp_path):
    environment = {
        "APPDATA": str(tmp_path / "Roaming"),
        "LOCALAPPDATA": str(tmp_path / "Local"),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
        "XDG_STATE_HOME": str(tmp_path / "xdg-state"),
    }

    assert user_data_dir(platform_name="darwin", home=tmp_path) == (
        tmp_path / "Library" / "Application Support" / "GuitarBapu"
    )
    assert user_log_dir(platform_name="darwin", home=tmp_path) == (
        tmp_path / "Library" / "Logs" / "GuitarBapu"
    )
    assert user_data_dir(
        platform_name="win32",
        home=tmp_path,
        environ=environment,
    ) == tmp_path / "Roaming" / "GuitarBapu"
    assert user_cache_dir(
        platform_name="linux",
        home=tmp_path,
        environ=environment,
    ) == tmp_path / "xdg-cache" / "GuitarBapu"
    assert crash_report_dir(platform_name="linux", home=tmp_path, environ={}) == (
        tmp_path / ".local" / "share" / "GuitarBapu" / "crashes"
    )


def test_rotating_log_and_crash_report_are_written_without_environment_dump(tmp_path):
    log_path = configure_logging(log_dir=tmp_path / "logs", level=logging.DEBUG)
    get_logger("test").error("example failure")
    for handler in get_logger().handlers:
        handler.flush()

    try:
        raise RuntimeError("synthetic crash")
    except RuntimeError as error:
        report = write_crash_report(
            type(error),
            error,
            error.__traceback__,
            report_dir=tmp_path / "crashes",
        )

    assert "example failure" in log_path.read_text(encoding="utf-8")
    content = report.read_text(encoding="utf-8")
    assert "synthetic crash" in content
    assert "Traceback:" in content
    assert "PATH=" not in content


def test_model_status_distinguishes_dependencies_cache_and_ready(monkeypatch):
    manager = OptionalModelManager()
    monkeypatch.setattr(
        OptionalModelManager,
        "dependencies_available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(manager, "_cached_files", lambda: ())

    missing = manager.status()

    monkeypatch.setattr(
        manager,
        "_cached_files",
        lambda: HTDEMUCS_6S.required_files,
    )
    ready = manager.status()

    assert missing.dependencies_available is True
    assert missing.cached is False
    assert ready.ready is True
    assert "已缓存" in ready.summary


def test_model_prepare_uses_explicit_downloader_and_validates_cache(monkeypatch):
    cached = []
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        cached.extend(HTDEMUCS_6S.required_files)
        return "/tmp/model"

    manager = OptionalModelManager(downloader=downloader)
    monkeypatch.setattr(
        OptionalModelManager,
        "dependencies_available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(manager, "_cached_files", lambda: tuple(cached))

    status = manager.prepare()

    assert status.ready is True
    assert calls[0]["repo_id"] == HTDEMUCS_6S.repository
    assert calls[0]["allow_patterns"] == list(HTDEMUCS_6S.required_files)


def test_diagnostics_render_and_save_without_network(tmp_path):
    class Manager:
        def status(self):
            return ModelStatus(HTDEMUCS_6S, True, True, HTDEMUCS_6S.required_files)

    diagnostics = collect_diagnostics(
        model_manager=Manager(),
        paths=(tmp_path,),
    )
    report = save_diagnostic_report(diagnostics, tmp_path / "diagnostics.txt")

    content = report.read_text(encoding="utf-8")
    assert "GuitarBapu 0.9.0 系统诊断" in content
    assert "htdemucs_6s" in content
    assert "环境变量" in content
