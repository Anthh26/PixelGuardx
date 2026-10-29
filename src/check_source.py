#!/usr/bin/env python3
"""Source-level guardrails for opt-in model-inclusive build. No device claims."""
from pathlib import Path
p=Path(__file__).resolve().parent
src=(p/'PixelGuard.xm').read_text()
assert '%hook UIImageView' in src
assert 'PGMaxOutstanding = 3' in src
assert 'PGIsEnabled()' in src
assert 'objc_getAssociatedObject' in src
assert 'PGPlaceholder()' in src
assert 'PGTokenKey' in src
assert 'VNRecognizedObjectObservation' in src
assert 'VNCoreMLRequest' in src
assert 'pg_decode_nms' in src
assert 'ROOT_PATH_NS' in src
assert 'dispatch_async(dispatch_get_main_queue()' in src
assert not (p/'layout/Library/Application Support/PixelGuard/disabled').exists()
assert 'nms=True' in (p/'export_coreml.py').read_text()
assert 'arm64e' in (p/'Makefile').read_text()
assert 'com.apple.mobileslideshow' in (p/'PixelGuard.plist').read_text()
assert 'Run workflow' in (p/'README.md').read_text()
print('PASS: enabled-by-default, Photos-only, NMS model contract, async UI update, kill switch, arm64e source checks')
