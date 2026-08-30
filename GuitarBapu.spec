# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import collect_all


datas = [
    # Bundled alphaTab renderer + fonts + soundfont for the in-app player view
    ("src/gui/web", "src/gui/web"),
]
binaries = []
hiddenimports = []

for package in ("librosa", "music21", "soundfile"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

include_separation = os.environ.get("GUITARBAPU_INCLUDE_SEPARATION") == "1"
if include_separation:
    for package in ("torch", "demucs", "huggingface_hub"):
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden

analysis = Analysis(
    ["src/gui/app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[] if include_separation else ["torch", "demucs", "huggingface_hub"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="GuitarBapu",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="GuitarBapu",
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name="GuitarBapu.app",
        icon=None,
        bundle_identifier="com.maomaogugu.guitarbapu",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": (
                "GuitarBapu can use the microphone for optional recording."
            ),
        },
    )
