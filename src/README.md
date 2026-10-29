# PixelGuard 0.6 — enabled-by-default, experimental Photos-only jailbreak tweak

**Target:** iPhone 14 Pro (A16, modern arm64e), iOS 16.6, Dopamine rootless. Photos app (`com.apple.mobileslideshow`) only.

**Current status:** source code + automated macOS/Theos build instructions. **This ZIP does not contain an installable `.deb`, an iOS-built tweak, or trained NudeNet weights.** All host-side C tests run, but the Mac workflow has NOT been run, and NOTHING has been tested on an iPhone. Do not install any previously generated Linux prototype `.deb` as though it were ready.

## User request: installed enabled immediately

0.6 is **enabled by default**, unlike 0.5. The package does not install a `disabled` sentinel file. If its compiled `.deb` is eventually installed, it will attempt processing immediately on eligible Photos images. The model must be bundled; the build fails if it is missing. To turn the tweak OFF, create an empty file at:

`/var/jb/Library/Application Support/PixelGuard/disabled`

Then force-close and relaunch Photos. Delete the `disabled` file to turn it on again. Rootless jailbreaks can have relocatable prefixes; use the actual `/var/jb` link or appropriate rootless path in your file manager. If Photos continually crashes, disable tweak injection in Dopamine, or uninstall PixelGuard in Sileo/Zebra. **Safe Mode is not guaranteed to activate for Photos crashes.** Back up your iPhone first.

## What is different in 0.6?

- Replaces 0.5's incorrect assumption that the Core ML exporter always returns a single `[1,22,N]` raw YOLO tensor. The conversion now requests **Core ML NMS (`nms=True`)** and explicitly checks that the model outputs `confidence` and `coordinates` with 18/80 and 4 channels respectively. The tweak handles either `VNRecognizedObjectObservation` or paired feature arrays.
- Fixes image-view reuse: each asynchronous processing job carries a per-view token; an old job cannot replace a newer photo.
- Eligible images are briefly shown as a black placeholder before inference finishes to avoid flashing the original via the hooked `UIImageView` pathway. Ordinary photos may also briefly appear black. When detection fails, the original is displayed. Processing overload can still leave an image unredacted.
- Still uses exposed-only classes: breasts, buttocks, genitalia, anus. Bounding boxes are pixelated on device; all rendering uses Core Image coordinates (bottom-left origin).
- Disabled-on-install file removed. A manual kill switch remains available.
- The macOS packaging check now **fails** if the `disabled` file exists, if the rootless layout is wrong, if the model is absent, or if the dylib does not have modern arm64e Mach-O metadata.

## Build your first real `.deb` on macOS

1. Place these files at the root of a GitHub repository, including the hidden `.github/workflows/build-arm64e.yml` directory. Consider a **private** repo for experiments.
2. In GitHub, select **Actions → Build PixelGuard v0.6 enabled-by-default ARM64E package → Run workflow**.
3. The workflow installs Theos, downloads a SHA-256-pinned third-party NudeNet v3.4 checkpoint, validates its exposed class labels, exports Core ML with embedded NMS, compiles it to `.mlmodelc`, and builds a rootless arm64e tweak.
4. If all checks pass, download the artifact named `PixelGuard-v0.6-arm64e-enabled`. It should contain the `.deb`. A CI build passing does NOT guarantee it will inject into Photos or render correctly on the phone; test one step at a time.

**Caution:** `320n.pt` is a third-party PyTorch pickle model. Its SHA-256 value prevents a changed download, but **does not prove the checkpoint is benign**. The workflow executes it during model loading. Use only in a build environment you are comfortable trusting and inspect licences before redistribution. Unmodified model bytes are not in this ZIP.

## First test on the device

1. Have Sileo/Zebra, Filza or another rootless file manager, and Dopamine's tweak-injection controls accessible. Back up first.
2. On first install of a **successfully verified macOS-built** `.deb`, open Photos. It is ON automatically, so begin with ordinary pictures. Check thumbnail views and full-screen views separately; large images and rotated images may bypass the hook.
3. If there is a problem, create the empty `disabled` file above, force-close and reopen Photos. Alternatively uninstall via Sileo/Zebra or boot with tweak injection disabled.
4. Send screenshots of the redacted results and any crash logs (do not send intimate originals unless you want to). We will use those to identify the Photos display pathway and model output, and then adjust the hook.

## Limitations that matter

- This is an experimental image-processing tweak, **not guaranteed nudity protection**: app UI that does not use `UIImageView.setImage:`, skipped/rotated images, low-confidence predictions, or inference failures can show originals.
- A black placeholder is only applied to eligible UIImageViews. Unsupported views and requests at the concurrency cap are displayed normally. There may still be flashes elsewhere in Photos.
- No video, Live Photos frame processing, Safari, Reddit or Telegram support yet.
- Photos could crash without SpringBoard entering Safe Mode. System stability has not been evaluated.
- The model conversion and Theos build require macOS/Xcode. This environment cannot produce a validated modern arm64e iPhone package.

## Locally validated commands

```bash
python3 test_detector.py
python3 check_source.py
```

## References

- https://theos.dev/docs/rootless
- https://theos.dev/docs/arm64e-deployment
- https://github.com/opa334/Dopamine
- https://huggingface.co/vladmandic/nudenet/blob/main/nudenet-v34-320n.pt
- https://docs.ultralytics.com/integrations/coreml
