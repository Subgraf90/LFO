# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['../MihilabUI/Main.py'],
    pathex=[
        '/Users/mgraf/Library/Mobile Documents/com~apple~CloudDocs/Documents/74_Python/slbook_env/lib/python3.11/site-packages',
    ],
    binaries=[],
    datas=[
        ('../MihilabUI/Module_Mihilab', 'Module_Mihilab'),
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.sip',
        'PyQt5.QtPrintSupport',
        'PyQt5.QtOpenGL',
        'qtpy.QtCore',
        'qtpy.QtGui',
        'qtpy.QtWidgets',
        'qtpy.QtOpenGL',
        'numpy',
        'scipy',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.figure',
        'matplotlib.ticker',
        'qtpy',
        'csv',
        'reportlab',
        'logging',
        're',
        'pickle',
        'random',
        'cProfile',
        'Module_Mihilab.Modules_Calculate.BeamSteering',
        'Module_Mihilab.Modules_Calculate.Functions',
        'Module_Mihilab.Modules_Calculate.ImpulseCalculator',
        'Module_Mihilab.Modules_Calculate.PolarCalculator',
        'Module_Mihilab.Modules_Calculate.SoundfieldCalculator',
        'Module_Mihilab.Modules_Calculate.WindowingCalculator'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    target_arch='universal2'
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Mihilab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.icns'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Mihilab'
)

app = BUNDLE(
    coll,
    name='Mihilab.app',
    icon='icon.icns',
    bundle_identifier='de.mihilab.app',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSRequiresAquaSystemAppearance': 'No',
        'LSMinimumSystemVersion': '10.13.0',
        'CFBundleExecutable': 'Mihilab',
        'LSBackgroundOnly': False,
        'NSAppleEventsUsageDescription': 'App requires access for automation.',
        'CFBundleIdentifier': 'de.mihilab.app',
        'CFBundlePackageType': 'APPL',
        'CFBundleInfoDictionaryVersion': '6.0',
    },
) 