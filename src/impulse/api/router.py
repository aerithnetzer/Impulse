"""API Gateway Lambda router.

This single Lambda handles all API Gateway routes, dispatching
based on the HTTP method and URL path.

Uses a {proxy+} catch-all integration, so routing is done by
matching the actual request path (not the API Gateway resource).
"""

from __future__ import annotations

import json
import re
import traceback

from loguru import logger


def handler(event: dict, context) -> dict:
    """API Gateway proxy integration entry point."""
    method = event.get("httpMethod", "")
    # With {proxy+}, use the actual path, stripped of the stage prefix
    raw_path = event.get("path", "")
    # Strip leading /prod/ or similar stage prefix
    path = "/" + raw_path.strip("/")

    logger.info(f"API request: {method} {path}")

    try:
        claims = event.get("requestContext", {}).get("authorizer", {}).get("claims", {})
        user_id = claims.get("sub", "anonymous")

        body = {}
        if event.get("body"):
            body = json.loads(event["body"])

        # ── Route matching ───────────────────────────────────────────

        # Jobs
        if path == "/jobs" and method == "POST":
            from impulse.api.create_job import create_job

            return create_job(body, user_id)

        elif path == "/jobs" and method == "GET":
            from impulse.api.list_jobs import list_jobs

            return list_jobs(user_id)

        elif (m := re.match(r"^/jobs/([^/]+)$", path)) and method == "GET":
            from impulse.api.get_job import get_job

            return get_job(m.group(1), user_id)

        elif (m := re.match(r"^/jobs/([^/]+)$", path)) and method == "DELETE":
            from impulse.api.delete_job import delete_job

            return delete_job(m.group(1), user_id)

        elif (m := re.match(r"^/jobs/([^/]+)/results$", path)) and method == "GET":
            from impulse.api.get_results import get_results

            query = event.get("queryStringParameters") or {}
            return get_results(
                m.group(1),
                user_id,
                page=int(query.get("page", "1")),
                page_size=int(query.get("page_size", "50")),
            )

        elif (m := re.match(r"^/jobs/([^/]+)/upload-url$", path)) and method == "POST":
            from impulse.api.presigned_upload import generate_upload_urls

            return generate_upload_urls(m.group(1), body, user_id)

        elif (m := re.match(r"^/jobs/([^/]+)/documents$", path)) and method == "GET":
            from impulse.api.list_documents import list_documents

            return list_documents(m.group(1), user_id)

        elif (m := re.match(r"^/jobs/([^/]+)/restart$", path)) and method == "POST":
            from impulse.api.restart_job import restart_job

            return restart_job(m.group(1), user_id)

        elif (
            m := re.match(r"^/jobs/([^/]+)/pipeline-status$", path)
        ) and method == "GET":
            from impulse.api.pipeline_status import get_pipeline_status

            return get_pipeline_status(m.group(1), user_id)

        elif (
            m := re.match(r"^/jobs/([^/]+)/environmental-impact$", path)
        ) and method == "GET":
            from impulse.api.environmental_impact import get_job_environmental_impact

            return get_job_environmental_impact(m.group(1), user_id)

        elif (
            m := re.match(r"^/jobs/([^/]+)/hathitrust-package$", path)
        ) and method == "POST":
            from impulse.api.hathitrust_package import create_hathitrust_package

            return create_hathitrust_package(m.group(1), body, user_id)

        # Collections
        elif path == "/collections" and method == "POST":
            from impulse.api.collections import create_collection

            return create_collection(body, user_id)

        elif path == "/collections" and method == "GET":
            from impulse.api.collections import list_collections

            return list_collections(user_id)

        elif (m := re.match(r"^/collections/([^/]+)$", path)) and method == "GET":
            from impulse.api.collections import get_collection_detail

            return get_collection_detail(m.group(1), user_id)

        elif (m := re.match(r"^/collections/([^/]+)$", path)) and method == "PUT":
            from impulse.api.collections import update_collection

            return update_collection(m.group(1), body, user_id)

        elif (m := re.match(r"^/collections/([^/]+)$", path)) and method == "DELETE":
            from impulse.api.collections import delete_collection

            return delete_collection(m.group(1), user_id)

        elif (
            m := re.match(r"^/collections/([^/]+)/documents$", path)
        ) and method == "POST":
            from impulse.api.collections import modify_collection_documents

            return modify_collection_documents(m.group(1), body, user_id)

        elif (
            m := re.match(r"^/collections/([^/]+)/download$", path)
        ) and method == "GET":
            from impulse.api.collections import download_collection

            return download_collection(m.group(1), user_id)

        elif (
            m := re.match(r"^/collections/([^/]+)/environmental-impact$", path)
        ) and method == "GET":
            from impulse.api.environmental_impact import (
                get_collection_environmental_impact,
            )

            return get_collection_environmental_impact(m.group(1), user_id)

        # Admin
        elif path == "/admin/users" and method == "GET":
            from impulse.api.admin import list_users

            return list_users(user_id)

        elif path == "/admin/users" and method == "POST":
            from impulse.api.admin import create_user

            return create_user(body, user_id)

        elif (m := re.match(r"^/admin/users/([^/]+)$", path)) and method == "DELETE":
            from impulse.api.admin import delete_user

            return delete_user(m.group(1), user_id)

        elif (
            m := re.match(r"^/admin/users/([^/]+)/disable$", path)
        ) and method == "POST":
            from impulse.api.admin import disable_user

            return disable_user(m.group(1), user_id)

        elif (
            m := re.match(r"^/admin/users/([^/]+)/enable$", path)
        ) and method == "POST":
            from impulse.api.admin import enable_user

            return enable_user(m.group(1), user_id)

        # Search
        elif path == "/search" and method == "GET":
            from impulse.api.search import search

            query = event.get("queryStringParameters") or {}
            return search(user_id, query)

        # Analyses
        elif path == "/analyses" and method == "POST":
            from impulse.api.analysis import create_analysis

            return create_analysis(body, user_id)

        elif path == "/analyses" and method == "GET":
            from impulse.api.analysis import list_analyses

            return list_analyses(user_id)

        elif (m := re.match(r"^/analyses/([^/]+)$", path)) and method == "GET":
            from impulse.api.analysis import get_analysis

            return get_analysis(m.group(1), user_id)

        elif (m := re.match(r"^/analyses/([^/]+)$", path)) and method == "DELETE":
            from impulse.api.analysis import delete_analysis

            return delete_analysis(m.group(1), user_id)

        elif (m := re.match(r"^/analyses/([^/]+)/sources$", path)) and method == "PUT":
            from impulse.api.analysis import update_sources

            return update_sources(m.group(1), body, user_id)

        elif (m := re.match(r"^/analyses/([^/]+)/run$", path)) and method == "POST":
            from impulse.api.analysis import run_analysis

            return run_analysis(m.group(1), user_id)

        # CORS preflight
        elif method == "OPTIONS":
            return _response(200, {})

        else:
            return _response(
                404, {"error": "Not found", "path": path, "method": method}
            )

    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict:
    """Build an API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }
