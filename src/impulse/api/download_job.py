"""Job ZIP download -- async build with polling.

POST /jobs/{jobId}/download  -- kick off ZIP creation (async Lambda self-invoke)
GET  /jobs/{jobId}/download  -- poll for ZIP readiness / return presigned URL

Internal entry point ``build_zip`` is called by the async Lambda invocation.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection
from impulse.utils import generate_presigned_download_url

# File extensions considered images (go into transformed/ folder)
_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "jp2", "tif", "tiff"}

_DOWNLOAD_PREFIX = "downloads"


# ── Public API handlers ──────────────────────────────────────────────────────


def start_download(job_id: str, user_id: str) -> dict:
    """POST /jobs/{id}/download -- kick off async ZIP build.

    If a ZIP already exists for this job, returns its URL immediately.
    Otherwise, fires an async Lambda self-invocation and returns 202.
    """
    jobs_col = get_collection("jobs")
    job = jobs_col.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0, "job_id": 1, "custom_id": 1},
    )
    if not job:
        return _response(404, {"error": "Job not found"})

    # Check if a ZIP already exists
    existing = _find_existing_zip(job_id)
    if existing:
        url = generate_presigned_download_url(existing, expires_in=3600)
        return _response(200, {"status": "ready", "download_url": url})

    # Invoke ourselves asynchronously to build the ZIP
    lambda_client = boto3.client("lambda")
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "impulse-api")

    payload = {
        "_internal": "build_zip",
        "job_id": job_id,
        "user_id": user_id,
    }

    lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="Event",  # fire-and-forget
        Payload=json.dumps(payload).encode(),
    )

    logger.info(f"Kicked off async ZIP build for job {job_id}")
    return _response(202, {"status": "preparing"})


def check_download(job_id: str, user_id: str) -> dict:
    """GET /jobs/{id}/download -- check if the ZIP is ready.

    Returns ``{"status": "ready", "download_url": "..."}`` when done,
    or ``{"status": "preparing"}`` while in progress.
    """
    jobs_col = get_collection("jobs")
    job = jobs_col.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0, "job_id": 1},
    )
    if not job:
        return _response(404, {"error": "Job not found"})

    existing = _find_existing_zip(job_id)
    if existing:
        url = generate_presigned_download_url(existing, expires_in=3600)
        return _response(200, {"status": "ready", "download_url": url})

    return _response(200, {"status": "preparing"})


# ── Internal: async ZIP builder ──────────────────────────────────────────────


def build_zip(job_id: str, user_id: str) -> dict:
    """Build the ZIP archive and upload to S3.

    Called from the async Lambda self-invocation (not via API Gateway),
    so there is no 29-second timeout constraint.
    """
    logger.info(f"Building ZIP for job {job_id}")

    jobs_col = get_collection("jobs")
    job = jobs_col.find_one(
        {"job_id": job_id, "user_id": user_id},
        {
            "_id": 0,
            "job_id": 1,
            "custom_id": 1,
            "input_s3_prefix": 1,
            "output_s3_prefix": 1,
        },
    )

    if not job:
        logger.error(f"Job {job_id} not found during ZIP build")
        return {"error": "Job not found"}

    input_prefix = job.get("input_s3_prefix", f"uploads/{job_id}")
    output_prefix = job.get("output_s3_prefix", f"results/{job_id}")

    s3 = boto3.client("s3")

    input_files = _list_s3_objects(s3, input_prefix)
    output_files = _list_s3_objects(s3, output_prefix)

    if not input_files and not output_files:
        logger.warning(f"No files found for job {job_id}")
        return {"error": "No files"}

    # Load OCR results from MongoDB
    results_col = get_collection("results")
    ocr_by_key: dict[str, str] = {}
    for r in results_col.find(
        {"job_id": job_id},
        {"_id": 0, "document_key": 1, "extracted_text": 1},
    ):
        text = r.get("extracted_text", "")
        if text:
            ocr_by_key[r.get("document_key", "")] = text

    # Categorise output files
    transformed: list[dict] = []
    ocr_txt: list[dict] = []
    metadata_files: list[dict] = []

    for f in output_files:
        ext = _ext(f["filename"])
        if ext in _IMAGE_EXTS:
            transformed.append(f)
        elif ext == "txt":
            ocr_txt.append(f)
        else:
            metadata_files.append(f)

    ocr_basenames_on_s3 = {f["filename"].rsplit(".", 1)[0] for f in ocr_txt}

    # Build the ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in input_files:
            _add_s3_file(s3, zf, f["key"], f"source/{f['filename']}")

        for f in transformed:
            _add_s3_file(s3, zf, f["key"], f"transformed/{f['filename']}")

        for f in ocr_txt:
            _add_s3_file(s3, zf, f["key"], f"ocr/{f['filename']}")

        for doc_key, text in ocr_by_key.items():
            base = doc_key.split("/")[-1]
            base_no_ext = base.rsplit(".", 1)[0] if "." in base else base
            if base_no_ext not in ocr_basenames_on_s3:
                zf.writestr(f"ocr/{base_no_ext}.txt", text)

        for f in metadata_files:
            _add_s3_file(s3, zf, f["key"], f"metadata/{f['filename']}")

    buf.seek(0)

    zip_name = job.get("custom_id") or job_id
    zip_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in zip_name)
    zip_key = f"{_DOWNLOAD_PREFIX}/{job_id}/{zip_name}.zip"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=zip_key,
        Body=buf.getvalue(),
        ContentType="application/zip",
    )

    logger.info(f"Created ZIP for job {job_id}: {zip_key} ({buf.tell()} bytes)")
    return {"status": "ready", "zip_key": zip_key}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_existing_zip(job_id: str) -> str | None:
    """Return the S3 key of an existing ZIP for *job_id*, or None."""
    s3 = boto3.client("s3")
    prefix = f"{_DOWNLOAD_PREFIX}/{job_id}/"
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
    contents = resp.get("Contents", [])
    if contents and contents[0]["Key"].endswith(".zip"):
        return contents[0]["Key"]
    return None


def _ext(filename: str) -> str:
    """Return lowercase file extension (without the dot)."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _add_s3_file(
    s3_client,
    zf: zipfile.ZipFile,
    s3_key: str,
    zip_path: str,
) -> None:
    """Download an S3 object and write it into the ZIP archive."""
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        zf.writestr(zip_path, obj["Body"].read())
    except Exception as e:
        logger.warning(f"Skipping {s3_key}: {e}")


def _list_s3_objects(s3_client, prefix: str) -> list[dict]:
    """List all objects under an S3 prefix, returning filename/key/size."""
    objects: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename:
                continue
            objects.append(
                {
                    "key": key,
                    "filename": filename,
                    "size": obj["Size"],
                }
            )

    return sorted(objects, key=lambda x: x["filename"])


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
