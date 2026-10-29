#!/usr/bin/env python3
"""Check that an actual macOS-made Theos package contains arm64e + ML model."""
import pathlib,struct,subprocess,tempfile
p=next(pathlib.Path('packages').glob('*.deb'),None)
if p is None: raise SystemExit('Missing packages/*.deb')
with tempfile.TemporaryDirectory() as d:
  subprocess.run(['dpkg-deb','-x',str(p),d],check=True)
  prefix=pathlib.Path(d)/'var/jb/Library'
  dylib=prefix/'MobileSubstrate/DynamicLibraries/PixelGuard.dylib'
  model=prefix/'Application Support/PixelGuard/NudeNet.mlmodelc'
  assert dylib.is_file(),f'Missing {dylib}'
  assert model.is_dir() and any(p.is_file() for p in model.rglob('*')),f'Missing model files under {model}'
  disabled=prefix/'Application Support/PixelGuard/disabled'
  assert not disabled.exists(), 'The package should be enabled by default'
  data=dylib.read_bytes()[:16]
  magic,cpu,subtype,kind=struct.unpack('<4I',data)
  assert magic==0xfeedfacf and cpu==0x0100000c and kind==6
  assert subtype & 0xffffff == 2, f'Not arm64e (Mach-O subtype {subtype:#x})'
  print('PASS: arm64e Mach-O, rootless layout, compiled NudeNet model, enabled by default')
  print('Device test still required for injection, model inference and Photos stability')
