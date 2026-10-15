# PyInstaller spec for the portable one-file SunoCheck.exe (GUI).
# Build: uv run --group build pyinstaller packaging/sunocheck.spec --distpath dist --workpath build
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

ROOT: Path = Path(SPECPATH).parent  # noqa: F821 - injected by PyInstaller
PROJECT: dict[str, str] = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
VERSION: str = PROJECT["version"]
# Windows file versions are four plain integers; a pre-release suffix in pyproject must fail the build here.
NUMBERS: tuple[int, int, int, int] = (*(int(part) for part in VERSION.split(".")), 0)

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=NUMBERS, prodvers=NUMBERS),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    "040904B0",
                    [
                        StringStruct("CompanyName", "SunoCheck"),
                        StringStruct("FileDescription", "SunoCheck - AI music detection"),
                        StringStruct("FileVersion", VERSION),
                        StringStruct("InternalName", "SunoCheck"),
                        StringStruct("LegalCopyright", "MIT License"),
                        StringStruct("OriginalFilename", "SunoCheck.exe"),
                        StringStruct("ProductName", "SunoCheck"),
                        StringStruct("ProductVersion", VERSION),
                    ],
                )
            ]
        ),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

analysis = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    # Package metadata carries the version shown in Help -> About.
    datas=collect_data_files("sunocheck") + copy_metadata("sunocheck"),
    # winrt extension modules import each other from C, which the import scanner cannot see.
    hiddenimports=collect_submodules("winrt"),
    excludes=["tkinter"],
)
pyz = PYZ(analysis.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    name="SunoCheck",
    icon=str(ROOT / "src" / "sunocheck" / "assets" / "sunocheck.png"),
    version=version_info,
    console=False,
    # UPX-packed executables trigger far more antivirus false positives.
    upx=False,
)
