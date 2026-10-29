THEOS_PACKAGE_SCHEME = rootless
ARCHS = arm64e
TARGET = iphone:clang:latest:16.0
INSTALL_TARGET_PROCESSES = MobileSlideShow
TWEAK_NAME = PixelGuard
PixelGuard_FILES = PixelGuard.xm PGDetector.c
PixelGuard_CFLAGS = -O2 -fobjc-arc -Wno-deprecated-declarations
PixelGuard_FRAMEWORKS = Foundation UIKit Vision CoreML CoreImage
include $(THEOS)/makefiles/common.mk
include $(THEOS_MAKE_PATH)/tweak.mk
