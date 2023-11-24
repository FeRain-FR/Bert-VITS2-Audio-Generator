# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata, collect_data_files
d=[]
d += copy_metadata('tqdm')
d += copy_metadata('regex')
d += copy_metadata('requests')
d += copy_metadata('packaging')
d += copy_metadata('filelock')
d += copy_metadata('numpy')
d += copy_metadata('tokenizers')
d += copy_metadata('huggingface-hub')
d += copy_metadata('safetensors')
d += copy_metadata('accelerate')
d += copy_metadata('pyyaml')
d += collect_data_files('gradio_client')
d += collect_data_files('gradio')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=d,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={
    'gradio': 'py'
    }
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
