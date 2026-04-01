"""Convert METS XML to a HathiTrust-compliant YAML manifest.

METS (Metadata Encoding and Transmission Standard) is an XML schema from the
Library of Congress for encoding structural, descriptive, and administrative
metadata about digital library objects.

HathiTrust requires an ingest package consisting of JP2 images and a YAML
manifest describing scanner metadata, capture dates, and per-page labels.
"""

from __future__ import annotations

from lxml import etree as ET
from loguru import logger


# HathiTrust structural labels recognised in the METS logical structMap.
_LABEL_MAP: dict[str, str] = {
    "Title": "TITLE",
    "Contents": "TABLE_OF_CONTENTS",
    "Preface": "PREFACE",
    "Notes": "REFERENCES",
    "Bibliography": "REFERENCES",
    "Index": "INDEX",
}

# METS / XLink namespaces.
_NS = {
    "xmlns": "http://www.loc.gov/METS/",
    "xlink": "http://www.w3.org/1999/xlink",
}


def convert_mets_to_yaml(
    xml_bytes: bytes,
    *,
    scanner_make: str = "Kirtas",
    scanner_model: str = "APT 1200",
    resolution: int = 400,
    scanning_order: str = "left-to-right",
    reading_order: str = "left-to-right",
) -> tuple[str, list[str]]:
    """Convert a METS XML document to a HathiTrust ingest YAML string.

    Parameters
    ----------
    xml_bytes:
        Raw bytes of the METS XML file.
    scanner_make:
        Scanner manufacturer name.
    scanner_model:
        Scanner model identifier.
    resolution:
        Contone scanning resolution in DPI.
    scanning_order:
        Physical scanning direction (``"left-to-right"`` or ``"right-to-left"``).
    reading_order:
        Logical reading direction (``"left-to-right"`` or ``"right-to-left"``).

    Returns
    -------
    tuple[str, list[str]]
        A 2-tuple of:
        - The YAML manifest content as a string.
        - An ordered list of image filenames referenced by the METS
          (typically JP2 filenames).  The caller uses this to decide which
          images to include in the ingest package.

    Raises
    ------
    ValueError
        If the XML cannot be parsed or contains no usable page entries.
    """
    # ── Parse XML ────────────────────────────────────────────────────────
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        root = ET.fromstring(xml_bytes, parser)
    except ET.XMLSyntaxError as exc:
        raise ValueError(f"Invalid METS XML: {exc}") from exc

    # ── Helper: resolve file ID → filename ───────────────────────────────
    def _filename_for_file_id(file_id: str) -> str | None:
        nodes = root.xpath(
            f"//xmlns:file[@ID='{file_id}']/xmlns:FLocat",
            namespaces=_NS,
        )
        if not nodes:
            return None
        href = nodes[0].get("{http://www.w3.org/1999/xlink}href", "")
        return href[7:] if href.startswith("file://") else href

    # ── Extract capture date from metsHdr ────────────────────────────────
    mets_hdr = root.xpath("//xmlns:metsHdr", namespaces=_NS)
    if mets_hdr:
        raw_date = mets_hdr[0].get("CREATEDATE", "")
        # Append a timezone offset if the date doesn't already have one.
        capture_date = (
            raw_date
            if ("+" in raw_date or raw_date.endswith("Z"))
            else f"{raw_date}-06:00"
        )
    else:
        capture_date = "unknown"
        logger.warning("METS XML has no metsHdr element; capture_date set to 'unknown'")

    # ── Build YAML header ────────────────────────────────────────────────
    lines: list[str] = [
        f"capture_date: {capture_date}",
        f"scanner_make: {scanner_make}",
        f"scanner_model: {scanner_model}",
        'scanner_user: "Northwestern University Library: Repository & Digital Curation"',
        f"contone_resolution_dpi: {resolution}",
        f"image_compression_date: {capture_date}",
        "image_compression_agent: northwestern",
        'image_compression_tool: ["LIMB v4.5.0.0"]',
        f"scanning_order: {scanning_order}",
        f"reading_order: {reading_order}",
        "pagedata:",
    ]

    # ── Iterate logical structMap pages ──────────────────────────────────
    logical_pages = root.xpath(
        '//xmlns:structMap[@TYPE="logical"]//xmlns:div[@TYPE="page"]',
        namespaces=_NS,
    )

    if not logical_pages:
        raise ValueError(
            "METS XML contains no logical structMap page entries. "
            'Ensure the file has a <structMap TYPE="logical"> with page divs.'
        )

    referenced_filenames: list[str] = []

    for element in logical_pages:
        # Prefer JP2 file pointers; fall back to any fptr.
        fileptr = element.xpath(
            "./xmlns:fptr[starts-with(@FILEID, 'JP2')]", namespaces=_NS
        )
        if not fileptr:
            fileptr = element.xpath("./xmlns:fptr", namespaces=_NS)
        if not fileptr:
            continue

        file_id = fileptr[0].get("FILEID")
        page_filename = _filename_for_file_id(file_id)
        if not page_filename:
            logger.warning(f"Could not resolve filename for FILEID={file_id}")
            continue

        referenced_filenames.append(page_filename)

        parent = element.getparent()
        parent_label = parent.get("LABEL", "") if parent is not None else ""
        parent_type = parent.get("TYPE", "") if parent is not None else ""
        orderlabel = element.get("ORDERLABEL", "")

        line = _build_page_line(
            page_filename,
            element,
            parent,
            parent_label,
            parent_type,
            orderlabel,
            logical_pages,
        )

        if line:
            lines.append("    " + line)

    yaml_content = "\n".join(lines)
    logger.info(
        f"Generated HathiTrust YAML manifest with {len(referenced_filenames)} pages"
    )
    return yaml_content, referenced_filenames


# ── Private helpers ──────────────────────────────────────────────────────────


def _build_page_line(
    filename: str,
    element,
    parent,
    parent_label: str,
    parent_type: str,
    orderlabel: str,
    all_logical_pages: list,
) -> str | None:
    """Build a single YAML ``pagedata`` line for one page element.

    Returns ``None`` if no meaningful label or orderlabel can be derived.
    """
    # Only assign structural labels to the *first* page of a logical section.
    if parent is not None and element == parent[0]:
        # Front cover detection
        if (
            parent_label == "Cover"
            and parent_type == "cover"
            and parent == all_logical_pages[0].getparent()
        ):
            return f'{filename}: {{ label: "FRONT_COVER" }}'

        # Back cover: another "Cover/cover" div that is NOT the first one
        if parent_label == "Cover" and parent_type == "cover":
            return f'{filename}: {{ label: "BACK_COVER" }}'

        # Front matter with an orderlabel
        if parent_label == "Front Matter" and orderlabel:
            return f'{filename}: {{ orderlabel: "{orderlabel}" }}'

        # Chapter-like sections
        if parent_label.startswith("Chapter") or parent_label == "Appendix":
            return _fmt(filename, orderlabel, "CHAPTER_START")

        # Named structural sections (Title, Contents, Preface, etc.)
        if parent_label in _LABEL_MAP:
            return _fmt(filename, orderlabel, _LABEL_MAP[parent_label])

    # Fallback: page with an orderlabel but no special structural role.
    if orderlabel:
        return f'{filename}: {{ orderlabel: "{orderlabel}" }}'

    return None


def _fmt(filename: str, orderlabel: str, label: str) -> str:
    """Format a YAML page entry with optional orderlabel and label."""
    if orderlabel:
        return f'{filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
    return f'{filename}: {{ label: "{label}" }}'


# ── Form-based YAML builder (no METS) ───────────────────────────────────────


def build_yaml_from_pages(
    pages: list[dict],
    *,
    capture_date: str = "unknown",
    scanner_make: str = "Kirtas",
    scanner_model: str = "APT 1200",
    resolution: int = 400,
    scanning_order: str = "left-to-right",
    reading_order: str = "left-to-right",
) -> str:
    """Build a HathiTrust YAML manifest directly from a pages list.

    Each entry in *pages* is a dict with keys:

    - ``filename`` (str): the image filename (e.g. ``00000001.jp2``)
    - ``label`` (str, optional): structural label such as ``FRONT_COVER``
    - ``orderlabel`` (str, optional): page number string such as ``"1"``

    Parameters
    ----------
    pages:
        Ordered list of page dicts.
    capture_date:
        ISO-8601 capture date string (with timezone offset).
    scanner_make / scanner_model / resolution / scanning_order / reading_order:
        Scanner metadata fields for the YAML header.

    Returns
    -------
    str
        The complete YAML manifest content.

    Raises
    ------
    ValueError
        If *pages* is empty.
    """
    if not pages:
        raise ValueError("pages list must not be empty")

    lines: list[str] = [
        f"capture_date: {capture_date}",
        f"scanner_make: {scanner_make}",
        f"scanner_model: {scanner_model}",
        'scanner_user: "Northwestern University Library: Repository & Digital Curation"',
        f"contone_resolution_dpi: {resolution}",
        f"image_compression_date: {capture_date}",
        "image_compression_agent: northwestern",
        'image_compression_tool: ["LIMB v4.5.0.0"]',
        f"scanning_order: {scanning_order}",
        f"reading_order: {reading_order}",
        "pagedata:",
    ]

    for page in pages:
        filename = page["filename"]
        label = page.get("label", "").strip()
        orderlabel = page.get("orderlabel", "").strip()

        if label and orderlabel:
            lines.append(
                f'    {filename}: {{ orderlabel: "{orderlabel}", label: "{label}" }}'
            )
        elif label:
            lines.append(f'    {filename}: {{ label: "{label}" }}')
        elif orderlabel:
            lines.append(f'    {filename}: {{ orderlabel: "{orderlabel}" }}')
        else:
            lines.append(f"    {filename}: {{ }}")

    yaml_content = "\n".join(lines)
    logger.info(
        f"Built HathiTrust YAML manifest from form data with {len(pages)} pages"
    )
    return yaml_content
