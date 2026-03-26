"""Image processing functions: deskew, binarize, denoise, grayscale, encode.

Pipeline order (called by :func:`process_image`):
  1. Decode raw bytes to numpy array
  2. Normalize to single-channel grayscale (handles BGR, BGRA, gray)
  3. Denoise on grayscale (Non-Local Means — more effective before binarization)
  4. Deskew (Hough-based angle detection + rotation)
  5. Binarize (adaptive Gaussian threshold)
  6. Morphological cleanup (remove salt-and-pepper noise)
  7. Encode to output format
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from cv2.typing import MatLike


# ── Decode / encode helpers ─────────────────────────────────────────────


def to_array(content: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV array."""
    arr = np.frombuffer(content, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Failed to decode image bytes")
    return decoded


def encode_to_image(arr: np.ndarray, filetype: str = ".jp2") -> bytes:
    """Encode an OpenCV array to image bytes in the given format.

    Supports ``.jp2`` (JPEG 2000) via Pillow and other formats via OpenCV.
    """
    from PIL import Image

    if filetype == ".jp2":
        rgb = (
            cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            if len(arr.shape) == 2
            else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        )
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG2000")
        return buf.getvalue()

    success, buffer = cv2.imencode(filetype, arr)
    if not success:
        raise RuntimeError(f"cv2.imencode failed for {filetype}")
    return buffer.tobytes()


# ── Channel helpers ─────────────────────────────────────────────────────


def is_rgb(arr: np.ndarray) -> bool:
    """Return *True* if *arr* has 3 colour channels."""
    return len(arr.shape) == 3 and arr.shape[2] == 3


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    .. note::  ``cv2.imdecode`` returns BGR ordering, so we use
       ``COLOR_BGR2GRAY`` (not ``RGB2GRAY``).
    """
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


def normalize_channels(arr: np.ndarray) -> np.ndarray:
    """Normalize any input image to single-channel grayscale.

    Handles:
    - 2-D arrays (already grayscale) — returned as-is
    - 3-channel BGR (from ``cv2.imdecode``) — converted via BGR2GRAY
    - 4-channel BGRA — converted via BGRA2GRAY
    """
    if len(arr.shape) == 2:
        return arr
    channels = arr.shape[2]
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    if channels == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # Fallback: take the first channel
    return arr[:, :, 0]


# ── Denoising ───────────────────────────────────────────────────────────


def denoise(arr: np.ndarray, h: int = 10) -> np.ndarray:
    """Apply Non-Local Means Denoising to a *grayscale* image.

    Designed to run **before** binarization so that the adaptive-threshold
    algorithm receives a cleaner histogram.

    Parameters
    ----------
    arr : np.ndarray
        Single-channel grayscale image.
    h : int
        Filter strength. Higher values remove more noise but may blur fine
        details.  ``10`` is the OpenCV default and a reasonable starting
        point for scanned documents.
    """
    if len(arr.shape) == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(arr, None, h, 7, 21)


# ── Deskewing ───────────────────────────────────────────────────────────

# Minimum detected angle (in degrees) before we bother rotating.
# Below this threshold the interpolation artifacts outweigh the benefit.
_SKEW_THRESHOLD_DEG = 0.5


def detect_and_correct_skew(
    arr: np.ndarray,
    threshold_deg: float = _SKEW_THRESHOLD_DEG,
) -> np.ndarray:
    """Detect and correct document skew using Hough-based angle detection.

    Uses the ``deskew`` library (Canny + Hough Transform) for angle
    detection and ``cv2.warpAffine`` for rotation.

    Parameters
    ----------
    arr : np.ndarray
        Grayscale image (single-channel).
    threshold_deg : float
        Minimum absolute skew angle (in degrees) to trigger rotation.
        Below this threshold the image is returned unchanged.

    Returns
    -------
    np.ndarray
        The deskewed image (same dtype, may have a slightly larger canvas
        if rotation was applied).
    """
    from deskew import determine_skew

    angle = determine_skew(arr)

    if angle is None or abs(angle) < threshold_deg:
        return arr

    return _rotate_image(arr, angle)


def _rotate_image(arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *arr* by *angle* degrees with canvas expansion and white fill.

    Uses bilinear interpolation (``INTER_LINEAR``), which provides the
    best speed-to-quality trade-off for scanned documents.
    """
    h, w = arr.shape[:2]
    center = (w / 2, h / 2)

    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions so no content is clipped.
    cos = abs(rotation_mat[0, 0])
    sin = abs(rotation_mat[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix for the expanded canvas.
    rotation_mat[0, 2] += (new_w / 2) - center[0]
    rotation_mat[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        arr,
        rotation_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # white fill for grayscale
    )


# ── Binarization ────────────────────────────────────────────────────────


def binarize(
    arr: np.ndarray,
    block_size: int = 51,
    C: int = 10,
) -> np.ndarray:
    """Binarize using adaptive Gaussian thresholding.

    Unlike Otsu (a single global threshold), adaptive thresholding
    computes a local threshold for each pixel based on its neighborhood.
    This handles uneven illumination, staining, and gradual background
    changes that are common in scanned historical documents.

    Parameters
    ----------
    arr : np.ndarray
        Grayscale (single-channel) image.
    block_size : int
        Size of the pixel neighborhood used to compute the threshold.
        Must be odd.  ``51`` covers roughly a text-line height at 300 DPI.
    C : int
        Constant subtracted from the weighted mean.  Controls sensitivity:
        higher values classify more pixels as foreground (ink).
    """
    if len(arr.shape) == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        arr,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )


# ── Morphological cleanup ──────────────────────────────────────────────


def morphological_cleanup(arr: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Remove salt-and-pepper noise from a binary image via morphological opening.

    Opening (erosion → dilation) removes small isolated bright/dark blobs
    without affecting the shapes of larger structures like text characters.

    This is the correct post-binarization cleanup tool — much faster and
    more appropriate for binary images than Non-Local Means Denoising.

    Parameters
    ----------
    arr : np.ndarray
        Binary image (values 0 and 255).
    kernel_size : int
        Size of the square structuring element.  ``2`` removes single-
        pixel noise; ``3`` is more aggressive.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)


# ── Main pipeline ───────────────────────────────────────────────────────


def process_image(content: bytes, output_format: str = ".jp2") -> bytes:
    """Full image processing pipeline.

    Steps:
      1. Decode raw bytes → numpy array
      2. Normalize to grayscale (BGR / BGRA / gray)
      3. Denoise (Non-Local Means on continuous-tone grayscale)
      4. Deskew (Hough angle detection → rotation)
      5. Binarize (adaptive Gaussian threshold)
      6. Morphological cleanup (remove salt-and-pepper noise)
      7. Encode to *output_format*

    Called by the ECS image handler (``image_handler.py``).
    """
    raw_arr = to_array(content)
    gray = normalize_channels(raw_arr)
    denoised = denoise(gray)
    deskewed = detect_and_correct_skew(denoised)
    binary = binarize(deskewed)
    cleaned = morphological_cleanup(binary)
    return encode_to_image(cleaned, output_format)
