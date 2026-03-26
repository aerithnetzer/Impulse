"""Tests for the image processing module.

Covers:
- Channel normalization (BGR, BGRA, grayscale)
- Denoising (Non-Local Means on grayscale)
- Deskewing (angle detection + rotation)
- Binarization (adaptive Gaussian threshold)
- Morphological cleanup
- End-to-end ``process_image`` pipeline
- Encoding helpers
"""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from impulse.processing.images import (
    binarize,
    denoise,
    detect_and_correct_skew,
    encode_to_image,
    is_rgb,
    morphological_cleanup,
    normalize_channels,
    process_image,
    to_array,
    to_grayscale,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_gray(h: int = 100, w: int = 100, value: int = 128) -> np.ndarray:
    """Create a uniform grayscale image."""
    return np.full((h, w), value, dtype=np.uint8)


def _make_bgr(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a 3-channel BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_bgra(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a 4-channel BGRA image."""
    return np.zeros((h, w, 4), dtype=np.uint8)


def _make_noisy_gray(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a grayscale image with random noise."""
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def _make_text_block_image(
    h: int = 400, w: int = 600, angle: float = 0.0
) -> np.ndarray:
    """Create a synthetic grayscale document with horizontal text lines.

    Returns a white background with dark horizontal lines (simulating text
    lines). If *angle* is non-zero the entire image is rotated by that
    many degrees to simulate skew.
    """
    img = np.full((h, w), 230, dtype=np.uint8)  # light gray background

    # Draw horizontal "text lines"
    for y in range(50, h - 50, 30):
        cv2.line(img, (40, y), (w - 40, y), 30, 2)

    if abs(angle) > 0.01:
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(
            img,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=230,
        )

    return img


# ── is_rgb ───────────────────────────────────────────────────────────────


class TestIsRgb(unittest.TestCase):
    def test_true_for_3_channels(self):
        self.assertTrue(is_rgb(_make_bgr()))

    def test_false_for_grayscale(self):
        self.assertFalse(is_rgb(_make_gray()))

    def test_false_for_rgba(self):
        self.assertFalse(is_rgb(_make_bgra()))


# ── normalize_channels ───────────────────────────────────────────────────


class TestNormalizeChannels(unittest.TestCase):
    def test_grayscale_passthrough(self):
        gray = _make_gray()
        result = normalize_channels(gray)
        self.assertEqual(len(result.shape), 2)
        np.testing.assert_array_equal(result, gray)

    def test_bgr_to_gray(self):
        bgr = _make_bgr()
        result = normalize_channels(bgr)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape, (100, 100))

    def test_bgra_to_gray(self):
        bgra = _make_bgra()
        result = normalize_channels(bgra)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape, (100, 100))


# ── to_grayscale ─────────────────────────────────────────────────────────


class TestToGrayscale(unittest.TestCase):
    def test_produces_2d(self):
        bgr = _make_bgr()
        bgr[:, :, 0] = 200  # Blue channel
        gray = to_grayscale(bgr)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))


# ── denoise ──────────────────────────────────────────────────────────────


class TestDenoise(unittest.TestCase):
    def test_output_shape_matches_input(self):
        noisy = _make_noisy_gray()
        result = denoise(noisy)
        self.assertEqual(result.shape, noisy.shape)

    def test_reduces_noise(self):
        """Denoising a noisy image should reduce high-frequency variation."""
        rng = np.random.RandomState(42)
        # Start with a smooth gradient, add noise
        base = np.tile(np.linspace(50, 200, 100, dtype=np.uint8), (100, 1))
        noisy = np.clip(
            base.astype(np.int16) + rng.randint(-30, 30, base.shape), 0, 255
        ).astype(np.uint8)

        denoised = denoise(noisy)

        # The Laplacian variance (measure of high-frequency content) should
        # be lower after denoising
        lap_noisy = cv2.Laplacian(noisy, cv2.CV_64F).var()
        lap_denoised = cv2.Laplacian(denoised, cv2.CV_64F).var()
        self.assertLess(lap_denoised, lap_noisy)

    def test_handles_color_input(self):
        """If accidentally given a 3-channel image, denoise should convert
        to grayscale internally (guard clause)."""
        bgr = _make_bgr()
        result = denoise(bgr)
        self.assertEqual(len(result.shape), 2)


# ── detect_and_correct_skew ──────────────────────────────────────────────


class TestDeskew(unittest.TestCase):
    def test_straight_image_unchanged(self):
        """An image with no skew should be returned with the same shape."""
        img = _make_text_block_image(angle=0.0)
        result = detect_and_correct_skew(img)
        # Shape should be identical (no rotation applied)
        self.assertEqual(result.shape, img.shape)

    def test_skewed_image_corrected(self):
        """An image rotated by 5 degrees should be detected and corrected.

        We verify that the detected correction brings the angle close to
        zero by running detection again on the output.
        """
        from deskew import determine_skew

        img = _make_text_block_image(angle=5.0)
        corrected = detect_and_correct_skew(img, threshold_deg=0.1)

        # The corrected image's skew should be much smaller
        residual_angle = determine_skew(corrected)
        if residual_angle is not None:
            self.assertLess(abs(residual_angle), 1.5)

    def test_slight_skew_below_threshold_ignored(self):
        """Skew below the threshold should not trigger rotation."""
        img = _make_text_block_image(angle=0.0)

        # Mock determine_skew to return a tiny angle
        import deskew as _deskew

        with patch.object(_deskew, "determine_skew", return_value=0.2):
            result = detect_and_correct_skew(img, threshold_deg=0.5)
            # Should be returned unchanged (same object)
            np.testing.assert_array_equal(result, img)

    def test_none_angle_returns_unchanged(self):
        """If determine_skew returns None (no lines detected), image is
        returned as-is."""
        img = _make_gray()
        import deskew as _deskew

        with patch.object(_deskew, "determine_skew", return_value=None):
            result = detect_and_correct_skew(img)
            np.testing.assert_array_equal(result, img)

    def test_rotation_fills_with_white(self):
        """After rotation the border pixels should be white (255)."""
        img = _make_text_block_image(angle=0.0)
        # Force a rotation
        import deskew as _deskew

        with patch.object(_deskew, "determine_skew", return_value=10.0):
            result = detect_and_correct_skew(img, threshold_deg=0.1)
            # The top-left corner should be white (border fill)
            self.assertEqual(result[0, 0], 255)


# ── binarize (adaptive) ─────────────────────────────────────────────────


class TestBinarize(unittest.TestCase):
    def test_output_is_binary(self):
        """Binarized image should contain only 0 and 255."""
        gray = _make_noisy_gray()
        result = binarize(gray)
        unique = set(np.unique(result))
        self.assertTrue(unique.issubset({0, 255}))

    def test_output_shape_matches(self):
        gray = _make_noisy_gray(200, 300)
        result = binarize(gray)
        self.assertEqual(result.shape, (200, 300))

    def test_handles_rgb_input(self):
        """If given a 3-channel image, binarize should convert internally."""
        bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = binarize(bgr)
        self.assertEqual(len(result.shape), 2)
        unique = set(np.unique(result))
        self.assertTrue(unique.issubset({0, 255}))

    def test_handles_uneven_illumination(self):
        """Adaptive thresholding should recover text from a gradient background.

        Create an image with dark text on a background that varies from
        light (left) to dark (right).  Global Otsu would fail on the dark
        side; adaptive threshold should binarize correctly across both.
        """
        h, w = 200, 400
        # Gradient background: 240 on left, 80 on right
        bg = np.tile(np.linspace(240, 80, w, dtype=np.uint8), (h, 1))
        # Add "text" (darker than local background by 60 intensity)
        img = bg.copy()
        for y in range(40, h - 40, 25):
            for x in range(20, w - 20, 5):
                local_bg = bg[y, x]
                ink = max(0, int(local_bg) - 60)
                img[y, x] = ink

        result = binarize(img)

        # The "ink" pixels on the RIGHT side (where bg is dark) should
        # still be classified as foreground (black = 0).
        # Sample a known text pixel on the right side.
        right_text_y, right_text_x = 40, 380
        ink_value = max(0, int(bg[right_text_y, right_text_x]) - 60)
        img[right_text_y, right_text_x] = ink_value
        result = binarize(img)

        # Count foreground (0) pixels in the right half
        right_half = result[:, w // 2 :]
        fg_right = np.sum(right_half == 0)
        # There should be a meaningful number of foreground pixels
        # (adaptive threshold catches them; global Otsu might not)
        self.assertGreater(fg_right, 0)


# ── morphological_cleanup ───────────────────────────────────────────────


class TestMorphologicalCleanup(unittest.TestCase):
    def test_removes_single_pixel_noise(self):
        """Isolated single white pixels in a black region should be removed."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # Sprinkle single-pixel "salt" noise
        img[10, 10] = 255
        img[50, 50] = 255
        img[80, 30] = 255

        result = morphological_cleanup(img)
        # Single pixels should be eroded away
        self.assertEqual(result[10, 10], 0)
        self.assertEqual(result[50, 50], 0)

    def test_preserves_large_structures(self):
        """A large white rectangle should survive opening."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 20:80] = 255

        result = morphological_cleanup(img)
        # Interior of the rectangle should still be white
        self.assertEqual(result[50, 50], 255)

    def test_output_shape_matches(self):
        img = np.zeros((120, 80), dtype=np.uint8)
        result = morphological_cleanup(img)
        self.assertEqual(result.shape, (120, 80))


# ── encode_to_image ──────────────────────────────────────────────────────


class TestEncodeToImage(unittest.TestCase):
    def test_encode_to_png(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        encoded = encode_to_image(arr, ".png")
        self.assertIsInstance(encoded, bytes)
        self.assertTrue(encoded.startswith(b"\x89PNG"))

    def test_encode_to_jp2(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        encoded = encode_to_image(arr, ".jp2")
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)

    def test_encode_to_jpg(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        encoded = encode_to_image(arr, ".jpg")
        self.assertIsInstance(encoded, bytes)
        # JPEG magic bytes
        self.assertTrue(encoded[:2] == b"\xff\xd8")


# ── to_array ─────────────────────────────────────────────────────────────


class TestToArray(unittest.TestCase):
    def test_invalid_bytes_raises(self):
        with self.assertRaises(ValueError):
            to_array(b"not an image")

    def test_roundtrip_png(self):
        """Encode a numpy array to PNG and decode it back."""
        original = _make_gray(50, 60, value=100)
        png_bytes = encode_to_image(original, ".png")
        decoded = to_array(png_bytes)
        self.assertEqual(decoded.shape, (50, 60))
        np.testing.assert_array_equal(decoded, original)


# ── process_image (end-to-end) ───────────────────────────────────────────


class TestProcessImage(unittest.TestCase):
    def test_produces_valid_jpg(self):
        """Full pipeline should produce valid JPEG bytes."""
        img = _make_text_block_image()
        png_bytes = encode_to_image(img, ".png")

        result_bytes = process_image(png_bytes, output_format=".jpg")

        self.assertIsInstance(result_bytes, bytes)
        self.assertTrue(result_bytes[:2] == b"\xff\xd8")

    def test_produces_valid_jp2(self):
        """Full pipeline should produce valid JPEG 2000 bytes."""
        img = _make_text_block_image()
        png_bytes = encode_to_image(img, ".png")

        result_bytes = process_image(png_bytes, output_format=".jp2")

        self.assertIsInstance(result_bytes, bytes)
        self.assertGreater(len(result_bytes), 0)

    def test_handles_color_input(self):
        """Pipeline should work with a 3-channel BGR image."""
        bgr = _make_bgr(200, 300)
        bgr[:, :, 1] = 128  # some green
        png_bytes = encode_to_image(bgr, ".png")

        result_bytes = process_image(png_bytes, output_format=".jpg")
        self.assertIsInstance(result_bytes, bytes)

    def test_handles_bgra_input(self):
        """Pipeline should work with a 4-channel BGRA image."""
        bgra = _make_bgra(200, 300)
        bgra[:, :, 0] = 100  # some blue
        bgra[:, :, 3] = 255  # opaque alpha

        # Encode as PNG (supports alpha)
        success, buffer = cv2.imencode(".png", bgra)
        self.assertTrue(success)
        png_bytes = buffer.tobytes()

        result_bytes = process_image(png_bytes, output_format=".jpg")
        self.assertIsInstance(result_bytes, bytes)


if __name__ == "__main__":
    unittest.main()
