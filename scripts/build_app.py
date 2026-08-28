"""Build a local GuitarBapu application bundle with PyInstaller."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-separation",
        action="store_true",
        help="bundle optional PyTorch/Demucs dependencies (much larger build)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="reuse PyInstaller work files",
    )
    args = parser.parse_args(argv)

    if sys.version_info < (3, 10):
        parser.error("GuitarBapu packaging requires Python 3.10 or newer")
    if importlib.util.find_spec("PyInstaller") is None:
        parser.error("PyInstaller is missing; install requirements-dev.txt")

    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
    ]
    if not args.no_clean:
        command.append("--clean")
    command.append(str(project_root / "GuitarBapu.spec"))
    environment = dict(os.environ)
    environment.setdefault(
        "PYINSTALLER_CONFIG_DIR",
        str(Path(tempfile.gettempdir()) / "GuitarBapu-pyinstaller"),
    )
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "GuitarBapu-matplotlib"),
    )
    if args.with_separation:
        environment["GUITARBAPU_INCLUDE_SEPARATION"] = "1"
    else:
        environment.pop("GUITARBAPU_INCLUDE_SEPARATION", None)
    subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=True,
    )
    print(f"Build completed: {project_root / 'dist' / 'GuitarBapu'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
