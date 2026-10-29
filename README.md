# PixelGuardx — Sileo source

Experimental package repository for PixelGuard AI on Dopamine rootless (iOS 16.6; arm64e).

**The source URL, once GitHub Pages has deployed, is** `https://anthh26.github.io/PixelGuardx/`

The development repository is private. This public repository is solely for APT package metadata, releases, and corresponding source archives.

## Publishing

1. In Settings → Pages, choose **Source: GitHub Actions**.
2. Place a verified `com.anth.pixelguard_*.deb` package in `pool/`.
3. Include matching corresponding source files in `source/` for the same release, including the tweak's changes and the instructions to regenerate the Core ML model. Confirm upstream licensing and source-distribution requirements before public binary distribution.
4. Push the files to `main`. The `Publish PixelGuardx Sileo Repo` workflow checks and deploys the indexes.
5. Open `https://anthh26.github.io/PixelGuardx/`, then add that URL in Sileo → Sources → +.

On subsequent releases, push a higher version package with unchanged `Package: com.anth.pixelguard`. Sileo offers an update when sources refresh; installation still needs user approval.

**Warning:** The available v0.7 tweak is an experimental Photos-only implementation. It does not guarantee censorship while scrolling, on videos, in GIFs, or across all applications. The APT index is currently unsigned; do not distribute it as a trusted security filter.

NudeNet project: https://github.com/notAI-tech/NudeNet (AGPL-3.0). The v0.7 model comes from a separately hosted checkpoint and needs source/licensing diligence before redistribution.
