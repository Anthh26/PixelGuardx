# PixelGuard v0.8 — experimental on-device censorship for Photos

Target: iPhone 14 Pro, iOS 16.6, Dopamine rootless, arm64e. Injected only into Apple's Photos process.

## Changes from v0.7
- Scans smaller still-image candidates, down to 96×96 pixel CGImages, to try to include photo-grid thumbnails.
- Scans much larger source images (including typical 48 MP images) by drawing an upright, at-most-2560-pixel working image before Core ML detection and pixelation.
- Hooks UIKit `setAnimationImages:` and blocks `startAnimating` while a frame sequence is being processed, to reduce unredacted animation leakage. Supports at most 48 UIImage frames; other formats remain covered or unsupported.
- Keeps v0.7's processing-overload placeholder, deferred scan retry, per-view reuse token, and cached detections.

## Important remaining problems
- UIKit image interception cannot guarantee censorship in grid/zoom display paths using different Photos renderers, Core Animation, tiled layers, Metal, Video, Live Photos, or WebKit.
- No full video, GIF decoding outside UIImageView, Safari, or third-party apps.
- Enlarged/redrawn images may have lower visible resolution; photo grids may appear covered while scanning; large sets of photos can tax memory.
- This is NOT a guaranteed privacy or content-filtering barrier. Always test with ordinary images before relying on selective pixelation.

## Build and installation
Run the `Build PixelGuard AI and publish Sileo updates` GitHub Actions workflow in the public PixelGuardx repo. It downloads checksum-pinned NudeNet checkpoint, exports a compiled Core ML model, compiles the rootless arm64e tweak, validates the package, and deploys APT metadata to GitHub Pages.

Installed enabled automatically. Emergency disable: create `/var/jb/Library/Application Support/PixelGuard/disabled` and relaunch Photos (or uninstall in Sileo). Dopamine Safe Mode is not guaranteed to activate on Photos crashes.

NudeNet repo/license: https://github.com/notAI-tech/NudeNet (AGPL-3.0). Third-party model checkpoint and redistribution terms should be reviewed before distributing packages. Please use appropriately licensed, consenting adult test material.

## Local checks
```
python3 test_detector.py
python3 check_source.py
```
