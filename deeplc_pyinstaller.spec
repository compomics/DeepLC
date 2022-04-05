import os
import os
import sys

import importlib.metadata
from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, Tree

import gooey
from deeplc import __version__


# Package info
exe_name = "deeplc"
script_name = "deeplc/gui.py"
if sys.platform[:6] == "darwin":
	icon = './img/deeplc.icns'
else:
	icon = './img/deeplc.ico'
block_cipher = None
location = os.getcwd()
project = "deeplc"
bundle_name = "deeplc"
bundle_identifier = f"{bundle_name}.{__version__}"


# Collect hidden imports and datas for all requirements
requirements = {req.split()[0] for req in importlib.metadata.requires(project)}
requirements.update([project, "distributed", "sklearn", "gooey"])
hidden_imports = set()
datas = []
binaries = []
checked = set()
while requirements:
    requirement = requirements.pop()
    checked.add(requirement)
    try:
        module_version = importlib.metadata.version(requirement)
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError, ImportError):
        if requirement != "sklearn":
            continue
    try:
        datas_, binaries_, hidden_imports_ = collect_all(
            requirement, include_py_files=True
        )
    except ImportError:
        continue
    datas += datas_
    hidden_imports_ = set(hidden_imports_)
    if "" in hidden_imports_:
        hidden_imports_.remove("")
    if None in hidden_imports_:
        hidden_imports_.remove(None)
    requirements |= hidden_imports_ - checked
    hidden_imports |= hidden_imports_

hidden_imports = sorted([h for h in hidden_imports if "tests" not in h.split(".")])
hidden_imports = [h for h in hidden_imports if "__pycache__" not in h]
datas = [d for d in datas if ("__pycache__" not in d[0]) and (d[1] not in [".", "build", "dist", "Output"])]


# Additional Gooey files
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, "languages"), prefix="gooey/languages")
gooey_images = Tree(os.path.join(gooey_root, "images"), prefix="gooey/images")


# Build package
a = Analysis(
    [script_name],
    pathex=[location],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    windowed=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="./img/deeplc.ico"
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    gooey_languages,
    gooey_images,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=exe_name
)
