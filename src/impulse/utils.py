"""
Shared utility functions for the Impulse pipeline.

All S3 path parsing, content fetching, and common helpers live here.
No other module should duplicate these.
"""

from __future__ import annotations

import re
from io import BytesIO

import boto3

from impulse.config import S3_BUCKET

# ── Module-level clients (reused across invocations in Lambda / ECS) ────────

_s3_client = None


def _get_s3_client():
    """Return a shared S3 client.  Uses the default credential chain
    (IAM role in Lambda/Fargate, profile/env in local dev).

    Configured with SigV4 signing for presigned URL compatibility.
    """
    global _s3_client
    if _s3_client is None:
        from botocore.config import Config

        _s3_client = boto3.client(
            "s3",
            config=Config(signature_version="s3v4"),
        )
    return _s3_client


# ── S3 helpers ──────────────────────────────────────────────────────────────


def is_s3_path(path: str) -> bool:
    """Return *True* if *path* is an ``s3://`` or ``s3a://`` URI."""
    return bool(re.match(r"^s3a?://", path))


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse an S3 URI into *(bucket, key)*.

    Supports both ``s3://`` and ``s3a://`` schemes.
    """
    path = re.sub(r"^s3a?://", "", s3_path)
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def get_s3_content(s3_path: str) -> bytes:
    """Download an object from S3 and return its bytes."""
    bucket, key = parse_s3_path(s3_path)
    client = _get_s3_client()
    buf = BytesIO()
    client.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf.read()


def put_s3_content(s3_path: str, body: bytes, content_type: str | None = None) -> None:
    """Upload *body* bytes to *s3_path*.

    If *content_type* is not provided, it is inferred from the file extension.
    """
    bucket, key = parse_s3_path(s3_path)
    client = _get_s3_client()

    if content_type is None:
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
        _ct_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "tif": "image/tiff",
            "tiff": "image/tiff",
            "jp2": "image/jp2",
            "pdf": "application/pdf",
            "txt": "text/plain",
            "json": "application/json",
            "xml": "application/xml",
            "yaml": "text/yaml",
            "yml": "text/yaml",
        }
        content_type = _ct_map.get(ext, "application/octet-stream")

    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def generate_presigned_upload_url(
    key: str,
    bucket: str | None = None,
    expires_in: int = 3600,
    content_type: str = "application/octet-stream",
) -> str:
    """Generate a presigned PUT URL for direct browser uploads."""
    client = _get_s3_client()
    return client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket or S3_BUCKET,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )


def generate_presigned_download_url(
    key: str,
    bucket: str | None = None,
    expires_in: int = 3600,
) -> str:
    """Generate a presigned GET URL for downloading/viewing results."""
    client = _get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket or S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )


# ── File-type detection ─────────────────────────────────────────────────────


def detect_filetype(contents: bytes) -> str | None:
    """Determine file type from raw bytes using magic numbers.

    Returns a file extension string (e.g. ``"png"``, ``"pdf"``) or *None*.
    """
    if not contents or len(contents) < 4:
        return None

    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if contents.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if contents.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if contents.startswith(b"%PDF"):
        return "pdf"
    if contents.startswith(b"PK\x03\x04"):
        return "zip"
    if contents.startswith(b"\x1f\x8b"):
        return "gz"
    if contents.startswith(b"ID3"):
        return "mp3"
    if len(contents) > 8 and contents[4:8] == b"ftyp":
        return "mp4"
    if contents.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        return "jp2"

    # Plain text heuristic
    try:
        contents[:512].decode("utf-8")
        return "txt"
    except UnicodeDecodeError:
        pass

    return None


# ── PDF helpers ─────────────────────────────────────────────────────────────


def pdf_to_png_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """Convert each PDF page to raw PNG bytes.

    Returns a list of PNG byte strings, one per page.  Raises
    ``ValueError`` for empty or corrupt PDFs.
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

    images: list[bytes] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        images.append(pix.tobytes("png"))
    return images


def pdf_to_base64_images(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """Convert each PDF page to a base64-encoded PNG string."""
    import base64

    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        images.append(img_b64)
    return images
