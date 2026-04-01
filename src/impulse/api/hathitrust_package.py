"""POST /jobs/{jobId}/hathitrust-package -- Build a HathiTrust ingest package.

Accepts a JSON body with a ``pages`` array (populated from the job's output
images on the frontend) together with scanner settings.  Builds a HathiTrust
YAML manifest from the page data, bundles it with the job's output images
(converting to JP2 if needed), and returns a presigned download URL for the
resulting ZIP.
"""

from __future__ import annotations

import io
import json
import zipfile

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection
from impulse.processing.mets import build_yaml_from_pages
from impulse.utils import generate_presigned_download_url

# File extensions already in JP2 format.
_JP2_EXTS = {"jp2", "j2k", "jpf", "jpx"}

# Image extensions we can convert to JP2.
_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "tif", "tiff", "jp2", "j2k"}

_DOWNLOAD_PREFIX = "downloads"


def create_hathitrust_package(job_id: str, body: dict, user_id: str) -> dict:
    """Build a HathiTrust ingest package for the given job.

    Expected JSON body::

        {
            "pages": [
                {"filename": "00000001.jp2", "s3_key": "results/.../00000001.jp2",
                 "label": "FRONT_COVER", "orderlabel": ""},
                {"filename": "00000002.jp2", "s3_key": "results/.../00000002.jp2",
                 "label": "", "orderlabel": "1"},
                ...
            ],
            "capture_date": "2024-06-15T00:00:00-06:00",
            "scanner_make": "Kirtas",         // optional
            "scanner_model": "APT 1200",      // optional
            "resolution": 400,                // optional
            "scanning_order": "left-to-right", // optional
            "reading_order": "left-to-right"   // optional
        }
    """
    # ── Validate job ownership ───────────────────────────────────────────
    jobs_col = get_collection("jobs")
    job = jobs_col.find_one(
        {"job_id": job_id, "user_id": user_id},
        {
            "_id": 0,
            "job_id": 1,
            "custom_id": 1,
            "status": 1,
            "output_s3_prefix": 1,
        },
    )

    if not job:
        return _response(404, {"error": "Job not found"})

    if job.get("status") != "COMPLETED":
        return _response(
            400,
            {"error": "Job must be completed before creating a HathiTrust package"},
        )

    # ── Validate pages array ─────────────────────────────────────────────
    pages = body.get("pages")
    if not pages or not isinstance(pages, list):
        return _response(
            400, {"error": "pages array is required and must not be empty"}
        )

    for i, page in enumerate(pages):
        if not isinstance(page, dict) or "filename" not in page or "s3_key" not in page:
            return _response(
                400,
                {
                    "error": f"Each page must have 'filename' and 's3_key' (invalid at index {i})"
                },
            )

    # ── Parse settings (with defaults) ───────────────────────────────────
    capture_date = body.get("capture_date", "unknown")
    scanner_make = body.get("scanner_make", "Kirtas")
    scanner_model = body.get("scanner_model", "APT 1200")
    resolution = int(body.get("resolution", 400))
    scanning_order = body.get("scanning_order", "left-to-right")
    reading_order = body.get("reading_order", "left-to-right")

    # ── Build YAML manifest from pages ───────────────────────────────────
    try:
        yaml_content = build_yaml_from_pages(
            pages,
            capture_date=capture_date,
            scanner_make=scanner_make,
            scanner_model=scanner_model,
            resolution=resolution,
            scanning_order=scanning_order,
            reading_order=reading_order,
        )
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    # ── Build ZIP ────────────────────────────────────────────────────────
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    included_images = 0
    warnings: list[str] = []

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write the YAML manifest
        yaml_filename = (job.get("custom_id") or job_id) + ".yml"
        yaml_filename = _sanitise_filename(yaml_filename)
        zf.writestr(yaml_filename, yaml_content)

        # Write images — each page carries its own s3_key, so no resolution
        # step is needed (the frontend already knows the exact keys).
        for page in pages:
            s3_key = page["s3_key"]
            filename = page["filename"]

            try:
                image_bytes = _download_s3(s3, s3_key)
            except Exception as exc:
                warnings.append(f"Failed to download {filename}: {exc}")
                continue

            # Ensure the image is JP2 format
            ext = _ext(filename)
            target_filename = _to_jp2_filename(filename)

            if ext not in _JP2_EXTS:
                try:
                    image_bytes = _convert_to_jp2(image_bytes)
                except Exception as exc:
                    warnings.append(f"Failed to convert {filename} to JP2: {exc}")
                    continue

            zf.writestr(target_filename, image_bytes)
            included_images += 1

    buf.seek(0)
    zip_bytes = buf.getvalue()

    # ── Upload ZIP to S3 ─────────────────────────────────────────────────
    zip_name = _sanitise_filename(job.get("custom_id") or job_id)
    zip_key = f"{_DOWNLOAD_PREFIX}/{job_id}/{zip_name}_hathitrust.zip"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=zip_key,
        Body=zip_bytes,
        ContentType="application/zip",
    )

    download_url = generate_presigned_download_url(zip_key, expires_in=3600)

    logger.info(
        f"Created HathiTrust package for job {job_id}: "
        f"{zip_key} ({len(zip_bytes)} bytes, {included_images} images)"
    )

    result: dict = {
        "download_url": download_url,
        "images_included": included_images,
        "total_pages": len(pages),
    }
    if warnings:
        result["warnings"] = warnings

    return _response(200, result)


# ── Private helpers ──────────────────────────────────────────────────────────


def _convert_to_jp2(image_bytes: bytes) -> bytes:
    """Convert raw image bytes to JP2 format using Pillow.

    Uses a lightweight Pillow-only path so the API Lambda does not need
    the heavier ``opencv-python`` package.  Output images from the
    pipeline are already processed (deskewed, denoised, etc.), so no
    additional image cleanup is performed here.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    # JP2 encoder supports RGB and L (grayscale); convert other modes.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG2000")
    return buf.getvalue()


def _to_jp2_filename(filename: str) -> str:
    """Replace the file extension with ``.jp2``."""
    if "." in filename:
        return filename.rsplit(".", 1)[0] + ".jp2"
    return filename + ".jp2"


def _download_s3(s3_client, key: str) -> bytes:
    """Download an S3 object and return its bytes."""
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def _ext(filename: str) -> str:
    """Return lowercase file extension (without the dot)."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _sanitise_filename(name: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }
