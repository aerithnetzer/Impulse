"""Tests for PDF page expansion in the validate step.

Covers:
- ``pdf_to_png_images()`` utility function
- ``_resolve_task_type()`` helper
- ``_build_output_key()`` helper
- ``_expand_pdf()`` — PDF-to-page-image expansion logic
- ``_validate_job()`` — integration with the full validate step
"""

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF

from impulse.utils import pdf_to_png_images


# ── Helper to create minimal valid PDFs for testing ────────────────────────


def _make_pdf(page_count: int = 3) -> bytes:
    """Create a minimal in-memory PDF with *page_count* blank A5 pages."""
    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page(width=420, height=595)  # A5 dimensions
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# ── pdf_to_png_images() tests ──────────────────────────────────────────────


class TestPdfToPngImages(unittest.TestCase):
    def test_returns_correct_page_count(self):
        pdf_bytes = _make_pdf(5)
        images = pdf_to_png_images(pdf_bytes)
        self.assertEqual(len(images), 5)

    def test_each_page_is_valid_png(self):
        pdf_bytes = _make_pdf(2)
        images = pdf_to_png_images(pdf_bytes)
        for img in images:
            self.assertIsInstance(img, bytes)
            self.assertTrue(img.startswith(b"\x89PNG"), "Expected PNG header")

    def test_single_page_pdf(self):
        pdf_bytes = _make_pdf(1)
        images = pdf_to_png_images(pdf_bytes)
        self.assertEqual(len(images), 1)

    def test_empty_pdf_raises(self):
        """A PDF with zero pages should raise ValueError.

        PyMuPDF refuses to save a zero-page document, so we patch
        ``fitz.open`` at import time to return a zero-page doc.
        """
        import fitz as _fitz

        original_open = _fitz.open

        def _fake_open(**kwargs):
            mock_doc = MagicMock()
            mock_doc.page_count = 0
            return mock_doc

        with patch.object(_fitz, "open", side_effect=_fake_open):
            with self.assertRaises(ValueError, msg="PDF has no pages"):
                pdf_to_png_images(b"%PDF-1.4 fake")

    def test_corrupt_bytes_raises(self):
        with self.assertRaises(ValueError):
            pdf_to_png_images(b"this is not a pdf")

    def test_empty_bytes_raises(self):
        with self.assertRaises(ValueError):
            pdf_to_png_images(b"")


# ── _resolve_task_type() tests ─────────────────────────────────────────────


class TestResolveTaskType(unittest.TestCase):
    def setUp(self):
        from impulse.handlers.lambda_handler import _resolve_task_type

        self.resolve = _resolve_task_type

    def test_full_pipeline_png(self):
        self.assertEqual(
            self.resolve("png", "full_pipeline", "textract"),
            "image_transform",
        )

    def test_full_pipeline_jpg(self):
        self.assertEqual(
            self.resolve("jpg", "full_pipeline", "textract"),
            "image_transform",
        )

    def test_full_pipeline_pdf_textract(self):
        self.assertEqual(
            self.resolve("pdf", "full_pipeline", "textract"),
            "ocr_textract",
        )

    def test_full_pipeline_pdf_marker(self):
        self.assertEqual(
            self.resolve("pdf", "full_pipeline", "marker_pdf"),
            "document_extraction",
        )

    def test_full_pipeline_xml(self):
        self.assertEqual(
            self.resolve("xml", "full_pipeline", "textract"),
            "mets",
        )

    def test_full_pipeline_unknown_ext(self):
        self.assertEqual(
            self.resolve("docx", "full_pipeline", "textract"),
            "metadata",
        )

    def test_image_transform_passthrough(self):
        self.assertEqual(
            self.resolve("png", "image_transform", "textract"),
            "image_transform",
        )

    def test_document_extraction_bedrock(self):
        self.assertEqual(
            self.resolve("pdf", "document_extraction", "bedrock_claude"),
            "ocr_bedrock_claude",
        )


# ── _build_output_key() tests ──────────────────────────────────────────────


class TestBuildOutputKey(unittest.TestCase):
    def setUp(self):
        from impulse.handlers.lambda_handler import _build_output_key

        self.build = _build_output_key

    def test_image_transform_produces_jpg(self):
        result = self.build(
            "scan_pdf_page_0000000001.png", "image_transform", "results/j1"
        )
        self.assertEqual(result, "results/j1/scan_pdf_page_0000000001.jpg")

    def test_non_image_keeps_filename(self):
        result = self.build("scan.xml", "mets", "results/j1")
        self.assertEqual(result, "results/j1/scan.xml")


# ── _expand_pdf() tests ───────────────────────────────────────────────────


class TestExpandPdf(unittest.TestCase):
    """Test the PDF expansion helper that downloads a PDF from S3,
    converts each page to PNG, uploads page images, and returns
    processing descriptors.
    """

    def _import_expand(self):
        from impulse.handlers.lambda_handler import _expand_pdf

        return _expand_pdf

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    def test_basic_expansion(self, mock_get, mock_put):
        """A 3-page PDF should produce 3 image_transform descriptors."""
        mock_get.return_value = _make_pdf(3)
        expand = self._import_expand()

        descriptors = expand(
            key="uploads/j1/doc.pdf",
            input_prefix="uploads/j1",
            output_prefix="results/j1",
            job_task_type="image_transform",
            ocr_engine="textract",
            job_id="j1",
            custom_id="",
        )

        self.assertEqual(len(descriptors), 3)
        # Verify S3 uploads were made for each page
        self.assertEqual(mock_put.call_count, 3)

        for i, desc in enumerate(descriptors, start=1):
            self.assertEqual(desc["task_type"], "image_transform")
            self.assertEqual(desc["job_id"], "j1")
            expected_key = f"uploads/j1/doc_pdf_page_{i:010d}.png"
            self.assertEqual(desc["document_key"], expected_key)
            self.assertIn(".jpg", desc["output_key"])  # image_transform -> jpg

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    def test_full_pipeline_adds_ocr_tasks(self, mock_get, mock_put):
        """full_pipeline with textract should produce image_transform + OCR
        tasks for each page."""
        mock_get.return_value = _make_pdf(2)
        expand = self._import_expand()

        descriptors = expand(
            key="uploads/j1/doc.pdf",
            input_prefix="uploads/j1",
            output_prefix="results/j1",
            job_task_type="full_pipeline",
            ocr_engine="textract",
            job_id="j1",
            custom_id="",
        )

        # 2 pages x (image_transform + ocr_textract) = 4 tasks
        self.assertEqual(len(descriptors), 4)

        task_types = [d["task_type"] for d in descriptors]
        self.assertEqual(task_types.count("image_transform"), 2)
        self.assertEqual(task_types.count("ocr_textract"), 2)

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    def test_full_pipeline_marker_pdf_no_ocr_duplication(self, mock_get, mock_put):
        """full_pipeline with marker_pdf should NOT duplicate OCR tasks."""
        mock_get.return_value = _make_pdf(2)
        expand = self._import_expand()

        descriptors = expand(
            key="uploads/j1/doc.pdf",
            input_prefix="uploads/j1",
            output_prefix="results/j1",
            job_task_type="full_pipeline",
            ocr_engine="marker_pdf",
            job_id="j1",
            custom_id="",
        )

        # marker_pdf: png -> image_transform (no extra OCR task)
        self.assertEqual(len(descriptors), 2)
        for d in descriptors:
            self.assertEqual(d["task_type"], "image_transform")

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    def test_corrupt_pdf_returns_empty(self, mock_get, mock_put):
        """Corrupt PDF bytes should return an empty list (no crash)."""
        mock_get.return_value = b"not a pdf"
        expand = self._import_expand()

        descriptors = expand(
            key="uploads/j1/bad.pdf",
            input_prefix="uploads/j1",
            output_prefix="results/j1",
            job_task_type="full_pipeline",
            ocr_engine="textract",
            job_id="j1",
            custom_id="",
        )

        self.assertEqual(descriptors, [])
        mock_put.assert_not_called()

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    def test_custom_id_propagated(self, mock_get, mock_put):
        """Custom ID should be passed through to each descriptor."""
        mock_get.return_value = _make_pdf(1)
        expand = self._import_expand()

        descriptors = expand(
            key="uploads/j1/doc.pdf",
            input_prefix="uploads/j1",
            output_prefix="results/j1",
            job_task_type="image_transform",
            ocr_engine="textract",
            job_id="j1",
            custom_id="NUL-2026-0042",
        )

        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0]["accession_number"], "NUL-2026-0042")
        self.assertEqual(descriptors[0]["impulse_identifier"], "NUL-2026-0042")

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    def test_page_filename_uses_pdf_page_infix(self, mock_get, mock_put):
        """Page filenames should use ``_pdf_page_NNN.png`` to avoid
        collisions with user-uploaded files."""
        mock_get.return_value = _make_pdf(1)
        expand = self._import_expand()

        descriptors = expand(
            key="uploads/j1/scan.pdf",
            input_prefix="uploads/j1",
            output_prefix="results/j1",
            job_task_type="image_transform",
            ocr_engine="textract",
            job_id="j1",
            custom_id="",
        )

        self.assertEqual(
            descriptors[0]["document_key"],
            "uploads/j1/scan_pdf_page_0000000001.png",
        )


# ── _validate_job() integration test ──────────────────────────────────────


class TestValidateJobPdfExpansion(unittest.TestCase):
    """Integration-level test that the validate step correctly expands PDFs."""

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    @patch("impulse.handlers.lambda_handler.get_collection")
    @patch("impulse.handlers.lambda_handler.boto3")
    def test_validate_expands_pdf_alongside_images(
        self, mock_boto3, mock_get_coll, mock_get_s3, mock_put_s3
    ):
        """A job with 1 image + 1 two-page PDF should produce 3 image
        descriptors (1 original image + 2 page images)."""
        from impulse.handlers.lambda_handler import _validate_job

        # Mock MongoDB
        mock_jobs = MagicMock()
        mock_jobs.find_one.return_value = {
            "job_id": "j1",
            "task_type": "image_transform",
            "ocr_engine": "textract",
            "output_s3_prefix": "results/j1",
            "custom_id": "",
        }
        mock_get_coll.return_value = mock_jobs

        # Mock S3 listing: 1 image + 1 PDF
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "uploads/j1/photo.jpg"},
                    {"Key": "uploads/j1/document.pdf"},
                ]
            }
        ]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_s3

        # Mock PDF download
        mock_get_s3.return_value = _make_pdf(2)

        result = _validate_job({"job_id": "j1"})

        # 1 original image + 2 page images from the PDF = 3 tasks
        self.assertEqual(result["total"], 3)
        docs = result["documents"]

        # First should be the original image
        self.assertEqual(docs[0]["document_key"], "uploads/j1/photo.jpg")
        self.assertEqual(docs[0]["task_type"], "image_transform")

        # Next two should be expanded PDF pages
        self.assertEqual(
            docs[1]["document_key"], "uploads/j1/document_pdf_page_0000000001.png"
        )
        self.assertEqual(
            docs[2]["document_key"], "uploads/j1/document_pdf_page_0000000002.png"
        )

        # Verify page images were uploaded to S3
        self.assertEqual(mock_put_s3.call_count, 2)

        # Verify the job was updated with the correct total
        mock_jobs.update_one.assert_called_once()
        update_call = mock_jobs.update_one.call_args
        self.assertEqual(update_call[0][1]["$set"]["total_documents"], 3)

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    @patch("impulse.handlers.lambda_handler.get_collection")
    @patch("impulse.handlers.lambda_handler.boto3")
    def test_validate_full_pipeline_pdf_creates_image_and_ocr_tasks(
        self, mock_boto3, mock_get_coll, mock_get_s3, mock_put_s3
    ):
        """full_pipeline + textract on a 2-page PDF should create
        2 image_transform + 2 ocr_textract = 4 tasks."""
        from impulse.handlers.lambda_handler import _validate_job

        mock_jobs = MagicMock()
        mock_jobs.find_one.return_value = {
            "job_id": "j1",
            "task_type": "full_pipeline",
            "ocr_engine": "textract",
            "output_s3_prefix": "results/j1",
            "custom_id": "",
        }
        mock_get_coll.return_value = mock_jobs

        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "uploads/j1/scan.pdf"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_s3

        mock_get_s3.return_value = _make_pdf(2)

        result = _validate_job({"job_id": "j1"})

        self.assertEqual(result["total"], 4)
        task_types = [d["task_type"] for d in result["documents"]]
        self.assertEqual(task_types.count("image_transform"), 2)
        self.assertEqual(task_types.count("ocr_textract"), 2)

    @patch("impulse.handlers.lambda_handler.put_s3_content")
    @patch("impulse.handlers.lambda_handler.get_s3_content")
    @patch("impulse.handlers.lambda_handler.get_collection")
    @patch("impulse.handlers.lambda_handler.boto3")
    def test_validate_corrupt_pdf_skipped_gracefully(
        self, mock_boto3, mock_get_coll, mock_get_s3, mock_put_s3
    ):
        """A corrupt PDF should be skipped without crashing the job."""
        from impulse.handlers.lambda_handler import _validate_job

        mock_jobs = MagicMock()
        mock_jobs.find_one.return_value = {
            "job_id": "j1",
            "task_type": "image_transform",
            "ocr_engine": "textract",
            "output_s3_prefix": "results/j1",
            "custom_id": "",
        }
        mock_get_coll.return_value = mock_jobs

        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "uploads/j1/good.jpg"},
                    {"Key": "uploads/j1/bad.pdf"},
                ]
            }
        ]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3.client.return_value = mock_s3

        # Return corrupt bytes for the PDF
        mock_get_s3.return_value = b"not a pdf"

        result = _validate_job({"job_id": "j1"})

        # Only the good image should be in the result
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["documents"][0]["document_key"], "uploads/j1/good.jpg")


if __name__ == "__main__":
    unittest.main()
