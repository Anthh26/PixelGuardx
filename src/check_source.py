#!/usr/bin/env python3
"""Source-level guardrails for opt-in model-inclusive build. No device claims."""
from pathlib import Path
p=Path(__file__).resolve().parent
src=(p/'PixelGuard.xm').read_text()
assert '%hook UIImageView' in src
assert 'width >= 96' in src
assert 'MAX(pixelWidth, pixelHeight) <= 2560.0' in src
assert '- (void)setAnimationImages:' in src
assert '- (void)startAnimating' in src
assert 'PGUprightImage' in src
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
assert 'Run' in (p/'README.md').read_text() and 'v0.8' in (p/'README.md').read_text()
print('PASS: PixelGuard v0.8, small photos, downsampling, UIKit animations, NMS contract, arm64e and kill switch')
