"use client";

import "@/lib/amplify-config";
import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter, usePathname } from "next/navigation";
import { apiGet, apiPost, apiDelete } from "@/lib/api";
import { StatusBadge, ProgressBar, MetadataTags } from "@/components/status";
import Link from "next/link";

interface Job {
  job_id: string;
  status: string;
  task_type: string;
  ocr_engine: string;
  custom_id: string;
  metadata: Record<string, string>;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  input_s3_prefix: string;
  output_s3_prefix: string;
  step_functions_arn: string;
  created_at: string;
  updated_at: string;
  progress_percent: number;
}

interface DocumentFile {
  filename: string;
  input: {
    key: string;
    size: number;
    url: string;
  } | null;
  outputs: {
    key: string;
    filename: string;
    size: number;
    url: string;
  }[];
  ocr_text?: string;
  ocr_model?: string;
}

interface DocumentsResponse {
  documents: DocumentFile[];
  input_count: number;
  output_count: number;
}

interface Result {
  result_id: string;
  document_key: string;
  page_number: number;
  extraction_model: string;
  extracted_text: string;
  summary: string;
  metadata: Record<string, unknown>;
  ner_entities: unknown[];
}

interface ResultsResponse {
  results: Result[];
  page: number;
  total: number;
  total_pages: number;
}

interface PipelineStep {
  name: string;
  status: string;
  started_at: string;
  ended_at: string | null;
  input_preview: string;
  output_preview: string | null;
  error: string | null;
  cause: string | null;
}

interface FailedItem {
  document_key: string;
  task_type?: string;
  error: string;
  cause: string;
}

interface MapRunInfo {
  status: string;
  total: number;
  succeeded: number;
  failed: number;
  pending: number;
  running: number;
  aborted: number;
  failed_items?: FailedItem[];
}

interface PipelineStatus {
  execution_status: string;
  started_at: string;
  stopped_at: string;
  steps: PipelineStep[];
  error: string | null;
  cause: string | null;
  map_run: MapRunInfo | null;
}

type Tab = "documents" | "results" | "pipeline" | "environmental";

interface ImpactSummary {
  total_energy_kwh: number;
  total_carbon_g_co2e: number;
  total_water_ml: number;
  total_bedrock_input_tokens: number;
  total_bedrock_output_tokens: number;
  total_bedrock_invocations: number;
  total_textract_api_calls: number;
  total_processing_duration_ms: number;
  total_input_bytes: number;
  total_output_bytes: number;
  document_count: number;
  metrics_count: number;
}

interface DocumentImpact {
  document_key: string;
  task_type: string;
  energy_kwh: number;
  carbon_g_co2e: number;
  water_ml: number;
  processing_duration_ms: number;
  bedrock_input_tokens: number;
  bedrock_output_tokens: number;
  textract_api_calls: number;
  compute_type: string;
}

interface ImpactComparisons {
  km_driven: number;
  hours_led_bulb: number;
  smartphone_charges: number;
  google_searches: number;
}

interface EnvironmentalImpact {
  job_id: string;
  summary: ImpactSummary | null;
  per_document: DocumentImpact[];
  comparisons: ImpactComparisons | null;
}

export default function JobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const pathname = usePathname();
  // In static export, useParams returns the build-time placeholder "_".
  // usePathname() is reactive and always reflects the real current URL.
  const rawId = params.id as string;
  const jobId = rawId === "_"
    ? (pathname.split("/").filter(Boolean)[1] || "_")
    : rawId;

  const [job, setJob] = useState<Job | null>(null);
  const [documents, setDocuments] = useState<DocumentFile[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [resultsPage, setResultsPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<Tab>("documents");
  const [selectedDoc, setSelectedDoc] = useState<number>(0);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const [expandedFailure, setExpandedFailure] = useState<number | null>(null);
  const [showAddToCollection, setShowAddToCollection] = useState(false);
  const [addDocKey, setAddDocKey] = useState<{s3_key: string; filename: string; size: number; source_type: string} | null>(null);
  const [userCollections, setUserCollections] = useState<{collection_id: string; name: string}[]>([]);
  const [addingToCollection, setAddingToCollection] = useState("");
  const [envImpact, setEnvImpact] = useState<EnvironmentalImpact | null>(null);
  const [envLoading, setEnvLoading] = useState(false);

  // HathiTrust package
  interface HtPageEntry {
    filename: string;
    s3_key: string;
    label: string;
    orderlabel: string;
  }
  const [showHathiTrustModal, setShowHathiTrustModal] = useState(false);
  const [hathiTrustLoading, setHathiTrustLoading] = useState(false);
  const [htPages, setHtPages] = useState<HtPageEntry[]>([]);
  const [htCaptureDate, setHtCaptureDate] = useState("");
  const [htScannerMake, setHtScannerMake] = useState("Kirtas");
  const [htScannerModel, setHtScannerModel] = useState("APT 1200");
  const [htResolution, setHtResolution] = useState(400);
  const [htScanningOrder, setHtScanningOrder] = useState("left-to-right");
  const [htReadingOrder, setHtReadingOrder] = useState("left-to-right");

  const fetchJob = useCallback(async () => {
    try {
      const data = await apiGet<{ job: Job }>(`/jobs/${jobId}`);
      setJob(data.job);
    } catch {
      // handle error
    }
  }, [jobId]);

  const fetchDocuments = useCallback(async () => {
    try {
      const data = await apiGet<DocumentsResponse>(`/jobs/${jobId}/documents`);
      setDocuments(data.documents || []);
    } catch {
      // Not available yet
    }
  }, [jobId]);

  const fetchResults = useCallback(
    async (page: number) => {
      try {
        const data = await apiGet<ResultsResponse>(
          `/jobs/${jobId}/results?page=${page}&page_size=20`
        );
        setResults(data.results || []);
        setTotalPages(data.total_pages || 1);
      } catch {
        // handle error
      }
    },
    [jobId]
  );

  useEffect(() => {
    Promise.all([fetchJob(), fetchDocuments(), fetchResults(1)]).then(() =>
      setLoading(false)
    );
  }, [fetchJob, fetchDocuments, fetchResults]);

  // Auto-refresh while processing
  useEffect(() => {
    if (!job || job.status !== "PROCESSING") return;
    const interval = setInterval(() => {
      fetchJob();
      fetchDocuments();
      fetchResults(resultsPage);
    }, 5000);
    return () => clearInterval(interval);
  }, [job, resultsPage, fetchJob, fetchDocuments, fetchResults]);

  const handleRestart = async () => {
    if (!job || restarting) return;
    if (!confirm("Restart this job? Progress counters will be reset and all documents will be reprocessed.")) {
      return;
    }
    setRestarting(true);
    try {
      await apiPost(`/jobs/${jobId}/restart`, {});
      await fetchJob();
    } catch (err) {
      alert(`Failed to restart: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRestarting(false);
    }
  };

  const confirmationId = job?.custom_id || job?.job_id || "";

  const handleDelete = async () => {
    if (!job || deleting) return;
    if (deleteConfirmText !== confirmationId) return;
    setDeleting(true);
    try {
      await apiDelete(`/jobs/${jobId}`);
      router.replace("/dashboard");
    } catch (err) {
      alert(`Failed to delete: ${err instanceof Error ? err.message : String(err)}`);
      setDeleting(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const openAddToCollection = async (s3_key: string, filename: string, size: number, source_type: string) => {
    setAddDocKey({ s3_key, filename, size, source_type });
    try {
      const data = await apiGet<{ collections: {collection_id: string; name: string}[] }>("/collections");
      setUserCollections(data.collections || []);
    } catch { /* ignore */ }
    setShowAddToCollection(true);
  };

  const handleAddToCollection = async (collectionId: string) => {
    if (!addDocKey) return;
    setAddingToCollection(collectionId);
    try {
      await apiPost(`/collections/${collectionId}/documents`, {
        action: "add",
        documents: [{
          s3_key: addDocKey.s3_key,
          filename: addDocKey.filename,
          job_id: jobId,
          source_type: addDocKey.source_type,
          size: addDocKey.size,
        }],
      });
      setShowAddToCollection(false);
      setAddDocKey(null);
    } catch (err) {
      alert(`Failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAddingToCollection("");
    }
  };

  const HT_LABELS = [
    "", "FRONT_COVER", "TITLE", "TABLE_OF_CONTENTS", "PREFACE",
    "CHAPTER_START", "REFERENCES", "INDEX", "BACK_COVER",
  ];

  const openHathiTrustModal = () => {
    // Populate page table from job output images, sorted by filename
    const outputPages: HtPageEntry[] = [];
    for (const doc of documents) {
      for (const out of doc.outputs) {
        const ext = out.filename.split(".").pop()?.toLowerCase() || "";
        if (["jpg", "jpeg", "png", "gif", "webp", "tif", "tiff", "jp2", "j2k"].includes(ext)) {
          outputPages.push({
            filename: out.filename,
            s3_key: out.key,
            label: "",
            orderlabel: "",
          });
        }
      }
    }
    outputPages.sort((a, b) => a.filename.localeCompare(b.filename));
    setHtPages(outputPages);
    // Default capture date to job creation date (YYYY-MM-DD format for date input)
    if (job?.created_at) {
      const d = new Date(job.created_at);
      setHtCaptureDate(d.toISOString().slice(0, 10));
    }
    setShowHathiTrustModal(true);
  };

  const updateHtPage = (index: number, field: "label" | "orderlabel", value: string) => {
    setHtPages(prev => {
      const next = [...prev];
      next[index] = { ...next[index], [field]: value };
      return next;
    });
  };

  const htSetFirstAsFrontCover = () => {
    if (htPages.length === 0) return;
    setHtPages(prev => {
      const next = [...prev];
      next[0] = { ...next[0], label: "FRONT_COVER" };
      return next;
    });
  };

  const htSetLastAsBackCover = () => {
    if (htPages.length === 0) return;
    setHtPages(prev => {
      const next = [...prev];
      next[next.length - 1] = { ...next[next.length - 1], label: "BACK_COVER" };
      return next;
    });
  };

  const htAutoNumberPages = () => {
    setHtPages(prev =>
      prev.map((p, i) => ({ ...p, orderlabel: String(i + 1) }))
    );
  };

  const htClearAllLabels = () => {
    setHtPages(prev =>
      prev.map(p => ({ ...p, label: "", orderlabel: "" }))
    );
  };

  const handleCreateHathiTrustPackage = async () => {
    if (htPages.length === 0 || hathiTrustLoading) return;
    setHathiTrustLoading(true);
    try {
      // Format capture date with timezone offset
      const captureDateFormatted = htCaptureDate
        ? `${htCaptureDate}T00:00:00-06:00`
        : "unknown";

      const data = await apiPost<{
        download_url: string;
        images_included: number;
        total_pages: number;
        warnings?: string[];
      }>(`/jobs/${jobId}/hathitrust-package`, {
        pages: htPages.map(p => ({
          filename: p.filename,
          s3_key: p.s3_key,
          label: p.label,
          orderlabel: p.orderlabel,
        })),
        capture_date: captureDateFormatted,
        scanner_make: htScannerMake,
        scanner_model: htScannerModel,
        resolution: htResolution,
        scanning_order: htScanningOrder,
        reading_order: htReadingOrder,
      });

      if (data.warnings && data.warnings.length > 0) {
        alert(
          `Package created with ${data.images_included}/${data.total_pages} images.\n\nWarnings:\n${data.warnings.join("\n")}`
        );
      }

      // Trigger download
      window.open(data.download_url, "_blank");
      setShowHathiTrustModal(false);
    } catch (err) {
      alert(`Failed to create HathiTrust package: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setHathiTrustLoading(false);
    }
  };

  const fetchEnvImpact = useCallback(async () => {
    if (envLoading) return;
    setEnvLoading(true);
    try {
      const data = await apiGet<EnvironmentalImpact>(`/jobs/${jobId}/environmental-impact`);
      setEnvImpact(data);
    } catch {
      // Not available
    } finally {
      setEnvLoading(false);
    }
  }, [jobId]);

  // Fetch environmental impact when the tab is selected
  useEffect(() => {
    if (activeTab === "environmental" && !envImpact && !envLoading) {
      fetchEnvImpact();
    }
  }, [activeTab, envImpact, envLoading, fetchEnvImpact]);

  const isImage = (filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase() || "";
    return ["jpg", "jpeg", "png", "gif", "webp"].includes(ext);
  };

  if (loading) {
    return (
      <div style={{ padding: "4rem", textAlign: "center", color: "var(--muted)" }}>
        Loading...
      </div>
    );
  }

  if (!job) {
    return (
      <div style={{ padding: "4rem", textAlign: "center" }}>
        <p style={{ color: "var(--muted)", marginBottom: "1rem" }}>Job not found.</p>
        <Link href="/dashboard" style={{ color: "var(--primary)" }}>
          Back to Dashboard
        </Link>
      </div>
    );
  }

  const currentDoc = documents[selectedDoc];

  return (
    <div style={{ maxWidth: "80rem", margin: "0 auto", padding: "2rem" }}>
      {/* Navigation */}
      <div style={{ marginBottom: "1.5rem" }}>
        <Link
          href="/dashboard"
          style={{ color: "var(--muted)", textDecoration: "none", fontSize: "0.875rem" }}
        >
          &larr; Dashboard
        </Link>
      </div>

      {/* ── Job Header Card ─────────────────────────────────────── */}
      <div
        style={{
          border: "1px solid var(--border)",
          borderRadius: "0.75rem",
          padding: "1.5rem",
          marginBottom: "1.5rem",
          backgroundColor: "white",
        }}
      >
        {/* Title row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
          <div>
            {job.custom_id && (
              <h1 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                {job.custom_id}
              </h1>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontFamily: "monospace", fontSize: "0.8rem", color: "var(--muted)" }}>
                {job.job_id}
              </span>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <StatusBadge status={job.status} />
            {(job.status === "FAILED" || job.status === "COMPLETED") && (
              <button
                onClick={handleRestart}
                disabled={restarting}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.375rem",
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "white",
                  backgroundColor: restarting ? "#9ca3af" : "var(--primary)",
                  border: "none",
                  borderRadius: "0.375rem",
                  cursor: restarting ? "not-allowed" : "pointer",
                  transition: "background-color 0.15s",
                }}
              >
                {restarting ? "Restarting..." : "Restart Job"}
              </button>
            )}
            {job.status === "COMPLETED" && (
              <button
                onClick={openHathiTrustModal}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.375rem",
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "white",
                  backgroundColor: "#7c3aed",
                  border: "none",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                  transition: "background-color 0.15s",
                }}
              >
                Create HathiTrust Package
              </button>
            )}
            {job.status !== "PROCESSING" && (
              <button
                onClick={() => { setDeleteConfirmText(""); setShowDeleteModal(true); }}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.375rem",
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "var(--danger)",
                  backgroundColor: "transparent",
                  border: "1px solid var(--danger)",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                  transition: "background-color 0.15s",
                }}
              >
                Delete Job
              </button>
            )}
          </div>
        </div>

        {/* Metadata tags */}
        {job.metadata && Object.keys(job.metadata).length > 0 && (
          <MetadataTags metadata={job.metadata} style={{ marginBottom: "1rem" }} />
        )}

        {/* Info grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(10rem, 1fr))",
            gap: "1rem",
            marginBottom: "1rem",
            fontSize: "0.875rem",
          }}
        >
          <div>
            <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginBottom: "0.125rem" }}>
              Task Type
            </div>
            <div style={{ fontWeight: 500 }}>{job.task_type.replace(/_/g, " ")}</div>
          </div>
          <div>
            <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginBottom: "0.125rem" }}>
              OCR Engine
            </div>
            <div style={{ fontWeight: 500 }}>
              {job.ocr_engine === "textract" ? "AWS Textract" : job.ocr_engine === "bedrock_claude" ? "Claude Vision" : job.ocr_engine === "marker_pdf" ? "Marker PDF" : job.ocr_engine || "—"}
            </div>
          </div>
          <div>
            <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginBottom: "0.125rem" }}>
              Documents
            </div>
            <div style={{ fontWeight: 500 }}>{job.total_documents}</div>
          </div>
          <div>
            <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginBottom: "0.125rem" }}>
              Created
            </div>
            <div style={{ fontWeight: 500 }}>{new Date(job.created_at).toLocaleString()}</div>
          </div>
          <div>
            <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginBottom: "0.125rem" }}>
              Last Updated
            </div>
            <div style={{ fontWeight: 500 }}>{new Date(job.updated_at).toLocaleString()}</div>
          </div>
        </div>

        {/* Environmental Impact Summary (shown when data available) */}
        {envImpact?.summary && (
          <div
            style={{
              display: "flex",
              gap: "0.75rem",
              marginBottom: "1rem",
              flexWrap: "wrap",
            }}
          >
            {[
              { label: "CO\u2082e", value: envImpact.summary.total_carbon_g_co2e < 1 ? `${(envImpact.summary.total_carbon_g_co2e * 1000).toFixed(1)} mg` : `${envImpact.summary.total_carbon_g_co2e.toFixed(2)} g`, color: "#16a34a", bg: "#f0fdf4" },
              { label: "Energy", value: envImpact.summary.total_energy_kwh < 0.001 ? `${(envImpact.summary.total_energy_kwh * 1_000_000).toFixed(1)} mWh` : `${(envImpact.summary.total_energy_kwh * 1000).toFixed(2)} Wh`, color: "#d97706", bg: "#fffbeb" },
              { label: "Water", value: envImpact.summary.total_water_ml < 1 ? `${(envImpact.summary.total_water_ml * 1000).toFixed(1)} \u00b5L` : `${envImpact.summary.total_water_ml.toFixed(2)} mL`, color: "#2563eb", bg: "#eff6ff" },
            ].map((kpi) => (
              <button
                key={kpi.label}
                onClick={() => setActiveTab("environmental")}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.375rem",
                  padding: "0.25rem 0.625rem",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  color: kpi.color,
                  backgroundColor: kpi.bg,
                  border: `1px solid ${kpi.color}22`,
                  borderRadius: "9999px",
                  cursor: "pointer",
                  transition: "opacity 0.15s",
                }}
              >
                <span>{kpi.label}</span>
                <span style={{ fontWeight: 700 }}>{kpi.value}</span>
              </button>
            ))}
          </div>
        )}

        <ProgressBar
          processed={job.processed_documents}
          failed={job.failed_documents}
          total={job.total_documents}
        />
      </div>

      {/* ── Tab Bar ─────────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          gap: "0",
          borderBottom: "2px solid var(--border)",
          marginBottom: "1.5rem",
        }}
      >
        {([
          { key: "documents" as Tab, label: "Documents", count: documents.length },
          { key: "results" as Tab, label: "Results", count: results.length },
          { key: "environmental" as Tab, label: "Environmental Impact", count: envImpact?.per_document?.length ?? null },
        ]).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: "0.75rem 1.25rem",
              fontSize: "0.875rem",
              fontWeight: activeTab === tab.key ? 600 : 400,
              color: activeTab === tab.key ? "var(--primary)" : "var(--muted)",
              backgroundColor: "transparent",
              border: "none",
              borderBottom: activeTab === tab.key ? "2px solid var(--primary)" : "2px solid transparent",
              marginBottom: "-2px",
              cursor: "pointer",
              transition: "color 0.15s",
            }}
          >
            {tab.label}
            {tab.count !== null && (
            <span
              style={{
                marginLeft: "0.5rem",
                padding: "0.125rem 0.375rem",
                borderRadius: "9999px",
                fontSize: "0.7rem",
                backgroundColor: activeTab === tab.key ? "#eff6ff" : "#f3f4f6",
                color: activeTab === tab.key ? "var(--primary)" : "var(--muted)",
              }}
            >
              {tab.count}
            </span>
            )}
          </button>
        ))}
      </div>

      {/* ── Documents Tab ───────────────────────────────────────── */}
      {activeTab === "documents" && (
        <>
          {documents.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                padding: "3rem",
                border: "1px dashed var(--border)",
                borderRadius: "0.5rem",
                color: "var(--muted)",
              }}
            >
              {job.status === "PROCESSING"
                ? "Documents will appear here as they are processed..."
                : "No documents found for this job."}
            </div>
          ) : (
            <div style={{ display: "flex", gap: "1.5rem", minHeight: "24rem" }}>
              {/* File list sidebar */}
              <div
                style={{
                  flex: "0 0 16rem",
                  border: "1px solid var(--border)",
                  borderRadius: "0.5rem",
                  overflow: "hidden",
                  backgroundColor: "white",
                }}
              >
                <div
                  style={{
                    padding: "0.625rem 0.75rem",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "var(--muted)",
                    borderBottom: "1px solid var(--border)",
                    backgroundColor: "#fafafa",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  Files ({documents.length})
                </div>
                <div style={{ maxHeight: "32rem", overflowY: "auto" }}>
                  {documents.map((doc, i) => (
                    <button
                      key={doc.filename}
                      onClick={() => setSelectedDoc(i)}
                      style={{
                        display: "block",
                        width: "100%",
                        padding: "0.5rem 0.75rem",
                        textAlign: "left",
                        backgroundColor: selectedDoc === i ? "#eff6ff" : "transparent",
                        border: "none",
                        borderBottom: "1px solid var(--border)",
                        borderLeft: selectedDoc === i ? "3px solid var(--primary)" : "3px solid transparent",
                        cursor: "pointer",
                        transition: "background-color 0.1s",
                      }}
                    >
                      <div
                        style={{
                          fontFamily: "monospace",
                          fontSize: "0.75rem",
                          fontWeight: selectedDoc === i ? 600 : 400,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {doc.filename}
                      </div>
                      <div style={{ fontSize: "0.675rem", color: "var(--muted)", marginTop: "0.125rem" }}>
                        {doc.input ? formatSize(doc.input.size) : ""}
                        {doc.outputs.length > 0 && ` | ${doc.outputs.length} output${doc.outputs.length > 1 ? "s" : ""}`}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Image viewer */}
              <div style={{ flex: 1, minWidth: 0 }}>
                {currentDoc && (
                  <div>
                    <h3 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "1rem", fontFamily: "monospace" }}>
                      {currentDoc.filename}
                    </h3>

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          currentDoc.ocr_text
                            ? currentDoc.outputs.length > 0
                              ? "1fr 1fr 1fr"
                              : "1fr 1fr"
                            : currentDoc.outputs.length > 0
                              ? "1fr 1fr"
                              : "1fr",
                        gap: "1rem",
                      }}
                    >
                      {/* Input image */}
                      {currentDoc.input && (
                        <div>
                          <div
                            style={{
                              fontSize: "0.7rem",
                              fontWeight: 600,
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                              color: "var(--muted)",
                              marginBottom: "0.5rem",
                            }}
                          >
                            Input
                          </div>
                          <div
                            style={{
                              border: "1px solid var(--border)",
                              borderRadius: "0.5rem",
                              overflow: "hidden",
                              backgroundColor: "#f9fafb",
                            }}
                          >
                            {isImage(currentDoc.filename) ? (
                              <>
                                <img
                                  src={currentDoc.input.url}
                                  alt={`Input: ${currentDoc.filename}`}
                                  crossOrigin="anonymous"
                                  style={{
                                    width: "100%",
                                    height: "auto",
                                    display: "block",
                                    cursor: "zoom-in",
                                  }}
                                  onClick={() => setLightboxUrl(currentDoc.input!.url)}
                                />
                                <div
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    padding: "0.375rem 0.75rem",
                                    borderTop: "1px solid var(--border)",
                                    fontSize: "0.7rem",
                                    color: "var(--muted)",
                                    backgroundColor: "white",
                                  }}
                                >
                                  <span style={{ fontFamily: "monospace" }}>
                                    {currentDoc.filename} ({formatSize(currentDoc.input.size)})
                                  </span>
                                  <a
                                    href={currentDoc.input.url}
                                    download={currentDoc.filename}
                                    target="_blank"
                                    rel="noreferrer"
                                    style={{
                                      color: "var(--primary)",
                                      textDecoration: "none",
                                      fontWeight: 500,
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    Download
                                  </a>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      openAddToCollection(currentDoc.input!.key, currentDoc.filename, currentDoc.input!.size, "input");
                                    }}
                                    style={{
                                      color: "var(--primary)",
                                      backgroundColor: "transparent",
                                      border: "none",
                                      cursor: "pointer",
                                      fontWeight: 500,
                                      fontSize: "0.7rem",
                                      padding: 0,
                                      marginLeft: "0.5rem",
                                    }}
                                    title="Add to collection"
                                  >
                                    + Collection
                                  </button>
                                </div>
                              </>
                            ) : (
                              <div
                                style={{
                                  padding: "2rem",
                                  textAlign: "center",
                                  color: "var(--muted)",
                                  fontSize: "0.8rem",
                                }}
                              >
                                <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem", opacity: 0.3 }}>
                                  &#128196;
                                </div>
                                <div>{currentDoc.filename}</div>
                                <div style={{ marginTop: "0.25rem" }}>{formatSize(currentDoc.input.size)}</div>
                                <a
                                  href={currentDoc.input.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  style={{
                                    display: "inline-block",
                                    marginTop: "0.5rem",
                                    color: "var(--primary)",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  Download
                                </a>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Output images */}
                      {currentDoc.outputs.map((out, oi) => (
                        <div key={out.key}>
                          <div
                            style={{
                              fontSize: "0.7rem",
                              fontWeight: 600,
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                              color: "var(--success)",
                              marginBottom: "0.5rem",
                            }}
                          >
                            Output{currentDoc.outputs.length > 1 ? ` ${oi + 1}` : ""}
                          </div>
                          <div
                            style={{
                              border: "1px solid var(--border)",
                              borderRadius: "0.5rem",
                              overflow: "hidden",
                              backgroundColor: "#f9fafb",
                            }}
                          >
                            {isImage(out.filename) ? (
                              <>
                                <img
                                  src={out.url}
                                  alt={`Output: ${out.filename}`}
                                  crossOrigin="anonymous"
                                  style={{
                                    width: "100%",
                                    height: "auto",
                                    display: "block",
                                    cursor: "zoom-in",
                                  }}
                                  onClick={() => setLightboxUrl(out.url)}
                                />
                                <div
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    padding: "0.375rem 0.75rem",
                                    borderTop: "1px solid var(--border)",
                                    fontSize: "0.7rem",
                                    color: "var(--muted)",
                                    backgroundColor: "white",
                                  }}
                                >
                                  <span style={{ fontFamily: "monospace" }}>
                                    {out.filename} ({formatSize(out.size)})
                                  </span>
                                  <a
                                    href={out.url}
                                    download={out.filename}
                                    target="_blank"
                                    rel="noreferrer"
                                    style={{
                                      color: "var(--primary)",
                                      textDecoration: "none",
                                      fontWeight: 500,
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    Download
                                  </a>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      openAddToCollection(out.key, out.filename, out.size, "output");
                                    }}
                                    style={{
                                      color: "var(--primary)",
                                      backgroundColor: "transparent",
                                      border: "none",
                                      cursor: "pointer",
                                      fontWeight: 500,
                                      fontSize: "0.7rem",
                                      padding: 0,
                                      marginLeft: "0.5rem",
                                    }}
                                    title="Add to collection"
                                  >
                                    + Collection
                                  </button>
                                </div>
                              </>
                            ) : (
                              <div
                                style={{
                                  padding: "2rem",
                                  textAlign: "center",
                                  color: "var(--muted)",
                                  fontSize: "0.8rem",
                                }}
                              >
                                <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem", opacity: 0.3 }}>
                                  &#128196;
                                </div>
                                <div style={{ fontFamily: "monospace" }}>{out.filename}</div>
                                <div style={{ marginTop: "0.25rem" }}>{formatSize(out.size)}</div>
                                <a
                                  href={out.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  style={{
                                    display: "inline-block",
                                    marginTop: "0.5rem",
                                    color: "var(--primary)",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  Download
                                </a>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}

                      {/* No outputs yet */}
                      {currentDoc.outputs.length === 0 && !currentDoc.ocr_text && currentDoc.input && (
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            border: "1px dashed var(--border)",
                            borderRadius: "0.5rem",
                            padding: "2rem",
                            color: "var(--muted)",
                            fontSize: "0.8rem",
                            textAlign: "center",
                          }}
                        >
                          {job.status === "PROCESSING"
                            ? "Output will appear here once processing completes..."
                            : "No output generated for this file."}
                        </div>
                      )}

                      {/* OCR Text */}
                      {currentDoc.ocr_text && (
                        <div>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              marginBottom: "0.5rem",
                            }}
                          >
                            <div
                              style={{
                                fontSize: "0.7rem",
                                fontWeight: 600,
                                textTransform: "uppercase",
                                letterSpacing: "0.05em",
                                color: "#7c3aed",
                              }}
                            >
                              OCR Text
                            </div>
                            {currentDoc.ocr_model && (
                              <span
                                style={{
                                  fontSize: "0.65rem",
                                  fontWeight: 500,
                                  padding: "0.125rem 0.375rem",
                                  borderRadius: "0.25rem",
                                  backgroundColor: "#f3f0ff",
                                  color: "#7c3aed",
                                }}
                              >
                                {currentDoc.ocr_model === "textract"
                                  ? "AWS Textract"
                                  : currentDoc.ocr_model === "bedrock_claude"
                                    ? "Claude Vision"
                                    : currentDoc.ocr_model}
                              </span>
                            )}
                          </div>
                          <div
                            style={{
                              border: "1px solid var(--border)",
                              borderRadius: "0.5rem",
                              overflow: "hidden",
                              backgroundColor: "white",
                              display: "flex",
                              flexDirection: "column",
                              maxHeight: "32rem",
                            }}
                          >
                            <pre
                              style={{
                                flex: 1,
                                margin: 0,
                                padding: "0.75rem",
                                fontSize: "0.75rem",
                                lineHeight: 1.6,
                                fontFamily: "'SF Mono', 'Consolas', 'Liberation Mono', monospace",
                                whiteSpace: "pre-wrap",
                                wordBreak: "break-word",
                                overflowY: "auto",
                                color: "var(--foreground)",
                              }}
                            >
                              {currentDoc.ocr_text}
                            </pre>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                padding: "0.375rem 0.75rem",
                                borderTop: "1px solid var(--border)",
                                fontSize: "0.7rem",
                                color: "var(--muted)",
                                backgroundColor: "#fafafa",
                              }}
                            >
                              <span>{currentDoc.ocr_text.length.toLocaleString()} characters</span>
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(currentDoc.ocr_text || "");
                                }}
                                style={{
                                  color: "var(--primary)",
                                  backgroundColor: "transparent",
                                  border: "none",
                                  cursor: "pointer",
                                  fontWeight: 500,
                                  fontSize: "0.7rem",
                                  padding: 0,
                                }}
                              >
                                Copy text
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Results Tab ─────────────────────────────────────────── */}
      {activeTab === "results" && (
        <>
          {results.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                padding: "3rem",
                border: "1px dashed var(--border)",
                borderRadius: "0.5rem",
                color: "var(--muted)",
              }}
            >
              {job.status === "PROCESSING"
                ? "Results will appear here as documents are processed..."
                : "No results yet."}
            </div>
          ) : (
            <>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                {results.map((r, i) => (
                  <div
                    key={r.result_id || i}
                    style={{
                      border: "1px solid var(--border)",
                      borderRadius: "0.5rem",
                      padding: "1rem",
                      backgroundColor: "white",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: "0.5rem",
                        fontSize: "0.8rem",
                      }}
                    >
                      <span style={{ fontFamily: "monospace", fontWeight: 600 }}>
                        Page {r.page_number}
                      </span>
                      <span style={{ color: "var(--muted)" }}>{r.extraction_model}</span>
                    </div>
                    {r.extracted_text && (
                      <p
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--foreground)",
                          maxHeight: "6rem",
                          overflow: "hidden",
                          whiteSpace: "pre-wrap",
                          lineHeight: 1.5,
                        }}
                      >
                        {r.extracted_text.slice(0, 500)}
                        {r.extracted_text.length > 500 && "..."}
                      </p>
                    )}
                    {r.summary && (
                      <div style={{ marginTop: "0.5rem" }}>
                        <div
                          style={{
                            fontSize: "0.7rem",
                            fontWeight: 600,
                            color: "var(--muted)",
                            textTransform: "uppercase",
                            letterSpacing: "0.05em",
                            marginBottom: "0.25rem",
                          }}
                        >
                          Summary
                        </div>
                        <p style={{ fontSize: "0.8rem", lineHeight: 1.5 }}>{r.summary}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    gap: "0.5rem",
                    marginTop: "1.5rem",
                  }}
                >
                  <button
                    disabled={resultsPage <= 1}
                    onClick={() => {
                      setResultsPage((p) => p - 1);
                      fetchResults(resultsPage - 1);
                    }}
                    style={{
                      padding: "0.5rem 1rem",
                      border: "1px solid var(--border)",
                      borderRadius: "0.375rem",
                      cursor: resultsPage <= 1 ? "not-allowed" : "pointer",
                      fontSize: "0.8rem",
                      backgroundColor: "white",
                      opacity: resultsPage <= 1 ? 0.5 : 1,
                    }}
                  >
                    Previous
                  </button>
                  <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>
                    Page {resultsPage} of {totalPages}
                  </span>
                  <button
                    disabled={resultsPage >= totalPages}
                    onClick={() => {
                      setResultsPage((p) => p + 1);
                      fetchResults(resultsPage + 1);
                    }}
                    style={{
                      padding: "0.5rem 1rem",
                      border: "1px solid var(--border)",
                      borderRadius: "0.375rem",
                      cursor: resultsPage >= totalPages ? "not-allowed" : "pointer",
                      fontSize: "0.8rem",
                      backgroundColor: "white",
                      opacity: resultsPage >= totalPages ? 0.5 : 1,
                    }}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── Environmental Impact Tab ─────────────────────────────── */}
      {activeTab === "environmental" && (
        <>
          {envLoading ? (
            <div style={{ textAlign: "center", padding: "3rem", color: "var(--muted)" }}>
              Loading environmental impact data...
            </div>
          ) : !envImpact?.summary ? (
            <div
              style={{
                textAlign: "center",
                padding: "3rem",
                border: "1px dashed var(--border)",
                borderRadius: "0.5rem",
                color: "var(--muted)",
              }}
            >
              <p style={{ marginBottom: "0.5rem", fontWeight: 500 }}>No environmental impact data available yet.</p>
              <p style={{ fontSize: "0.8rem" }}>
                Impact metrics are captured during processing. Data will appear here once documents have been processed.
              </p>
            </div>
          ) : (
            <>
              {/* Summary Cards */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(14rem, 1fr))",
                  gap: "1rem",
                  marginBottom: "1.5rem",
                }}
              >
                {/* Carbon Card */}
                <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", padding: "1.25rem", backgroundColor: "white" }}>
                  <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "#16a34a", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
                    Carbon Emissions
                  </div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--foreground)", marginBottom: "0.25rem" }}>
                    {envImpact.summary.total_carbon_g_co2e < 1
                      ? `${(envImpact.summary.total_carbon_g_co2e * 1000).toFixed(1)} mg`
                      : `${envImpact.summary.total_carbon_g_co2e.toFixed(2)} g`}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>CO&#8322; equivalent</div>
                </div>

                {/* Energy Card */}
                <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", padding: "1.25rem", backgroundColor: "white" }}>
                  <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "#d97706", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
                    Energy Consumption
                  </div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--foreground)", marginBottom: "0.25rem" }}>
                    {envImpact.summary.total_energy_kwh < 0.001
                      ? `${(envImpact.summary.total_energy_kwh * 1_000_000).toFixed(1)} mWh`
                      : `${(envImpact.summary.total_energy_kwh * 1000).toFixed(2)} Wh`}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Total energy used</div>
                </div>

                {/* Water Card */}
                <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", padding: "1.25rem", backgroundColor: "white" }}>
                  <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "#2563eb", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
                    Water Usage
                  </div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--foreground)", marginBottom: "0.25rem" }}>
                    {envImpact.summary.total_water_ml < 1
                      ? `${(envImpact.summary.total_water_ml * 1000).toFixed(1)} \u00b5L`
                      : `${envImpact.summary.total_water_ml.toFixed(2)} mL`}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Datacenter cooling</div>
                </div>

                {/* AI Inference Card */}
                <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", padding: "1.25rem", backgroundColor: "white" }}>
                  <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "#7c3aed", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
                    AI Inference
                  </div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--foreground)", marginBottom: "0.25rem" }}>
                    {((envImpact.summary.total_bedrock_input_tokens + envImpact.summary.total_bedrock_output_tokens) / 1000).toFixed(1)}k
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>
                    {envImpact.summary.total_bedrock_invocations} Bedrock call{envImpact.summary.total_bedrock_invocations !== 1 ? "s" : ""}
                    {envImpact.summary.total_textract_api_calls > 0 && ` + ${envImpact.summary.total_textract_api_calls} Textract call${envImpact.summary.total_textract_api_calls !== 1 ? "s" : ""}`}
                  </div>
                </div>
              </div>

              {/* Real-world Comparisons */}
              {envImpact.comparisons && (
                <div
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: "0.75rem",
                    padding: "1.25rem",
                    marginBottom: "1.5rem",
                    backgroundColor: "white",
                  }}
                >
                  <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.75rem" }}>
                    Equivalent To
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(10rem, 1fr))",
                      gap: "1rem",
                    }}
                  >
                    {[
                      { value: envImpact.comparisons.km_driven, unit: "km driven", description: "by car" },
                      { value: envImpact.comparisons.hours_led_bulb, unit: "hours", description: "of LED bulb" },
                      { value: envImpact.comparisons.smartphone_charges, unit: "charges", description: "of a smartphone" },
                      { value: envImpact.comparisons.google_searches, unit: "searches", description: "on Google" },
                    ].map((comp) => (
                      <div key={comp.unit} style={{ textAlign: "center" }}>
                        <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--foreground)" }}>
                          {comp.value < 0.01 ? "<0.01" : comp.value.toFixed(2)}
                        </div>
                        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--muted)" }}>{comp.unit}</div>
                        <div style={{ fontSize: "0.675rem", color: "var(--muted)" }}>{comp.description}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Impact by Task Type */}
              {(() => {
                const taskGroups: Record<string, { carbon: number; energy: number; count: number }> = {};
                for (const doc of envImpact.per_document) {
                  const label = doc.task_type.replace(/_/g, " ");
                  if (!taskGroups[label]) taskGroups[label] = { carbon: 0, energy: 0, count: 0 };
                  taskGroups[label].carbon += doc.carbon_g_co2e;
                  taskGroups[label].energy += doc.energy_kwh;
                  taskGroups[label].count += 1;
                }
                const maxCarbon = Math.max(...Object.values(taskGroups).map((g) => g.carbon), 0.001);

                return (
                  <div
                    style={{
                      border: "1px solid var(--border)",
                      borderRadius: "0.75rem",
                      padding: "1.25rem",
                      marginBottom: "1.5rem",
                      backgroundColor: "white",
                    }}
                  >
                    <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.75rem" }}>
                      Carbon by Task Type
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                      {Object.entries(taskGroups)
                        .sort(([, a], [, b]) => b.carbon - a.carbon)
                        .map(([label, group]) => (
                          <div key={label}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.25rem" }}>
                              <span style={{ fontWeight: 500, textTransform: "capitalize" }}>{label}</span>
                              <span style={{ color: "var(--muted)" }}>
                                {group.carbon < 1 ? `${(group.carbon * 1000).toFixed(1)} mg` : `${group.carbon.toFixed(2)} g`} CO&#8322;e
                                ({group.count} task{group.count !== 1 ? "s" : ""})
                              </span>
                            </div>
                            <div
                              style={{
                                height: "0.5rem",
                                borderRadius: "9999px",
                                backgroundColor: "#f3f4f6",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  height: "100%",
                                  borderRadius: "9999px",
                                  backgroundColor: "#16a34a",
                                  width: `${Math.max((group.carbon / maxCarbon) * 100, 2)}%`,
                                  transition: "width 0.3s ease",
                                }}
                              />
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                );
              })()}

              {/* Per-Document Breakdown Table */}
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "0.75rem",
                  overflow: "hidden",
                  backgroundColor: "white",
                }}
              >
                <div
                  style={{
                    padding: "0.75rem 1rem",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "var(--muted)",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    borderBottom: "1px solid var(--border)",
                    backgroundColor: "#fafafa",
                  }}
                >
                  Per-Document Breakdown ({envImpact.per_document.length} entries)
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid var(--border)", backgroundColor: "#fafafa" }}>
                        <th style={{ textAlign: "left", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Document</th>
                        <th style={{ textAlign: "left", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Task</th>
                        <th style={{ textAlign: "right", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Carbon</th>
                        <th style={{ textAlign: "right", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Energy</th>
                        <th style={{ textAlign: "right", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Water</th>
                        <th style={{ textAlign: "right", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Tokens</th>
                        <th style={{ textAlign: "right", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {envImpact.per_document.map((doc, i) => (
                        <tr key={`${doc.document_key}-${doc.task_type}-${i}`} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td style={{ padding: "0.5rem 0.75rem", fontFamily: "monospace", fontSize: "0.7rem", maxWidth: "14rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {doc.document_key.split("/").pop()}
                          </td>
                          <td style={{ padding: "0.5rem 0.75rem", textTransform: "capitalize" }}>
                            {doc.task_type.replace(/_/g, " ")}
                          </td>
                          <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", fontFamily: "monospace" }}>
                            {doc.carbon_g_co2e < 0.001 ? `${(doc.carbon_g_co2e * 1_000_000).toFixed(0)} \u00b5g` : doc.carbon_g_co2e < 1 ? `${(doc.carbon_g_co2e * 1000).toFixed(1)} mg` : `${doc.carbon_g_co2e.toFixed(2)} g`}
                          </td>
                          <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", fontFamily: "monospace" }}>
                            {doc.energy_kwh < 0.000001 ? `${(doc.energy_kwh * 1_000_000_000).toFixed(0)} nWh` : doc.energy_kwh < 0.001 ? `${(doc.energy_kwh * 1_000_000).toFixed(1)} mWh` : `${(doc.energy_kwh * 1000).toFixed(2)} Wh`}
                          </td>
                          <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", fontFamily: "monospace" }}>
                            {doc.water_ml < 0.001 ? `${(doc.water_ml * 1000).toFixed(1)} \u00b5L` : `${doc.water_ml.toFixed(2)} mL`}
                          </td>
                          <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", fontFamily: "monospace" }}>
                            {(doc.bedrock_input_tokens + doc.bedrock_output_tokens) > 0
                              ? `${((doc.bedrock_input_tokens + doc.bedrock_output_tokens) / 1000).toFixed(1)}k`
                              : doc.textract_api_calls > 0
                                ? `${doc.textract_api_calls} call${doc.textract_api_calls !== 1 ? "s" : ""}`
                                : "\u2014"}
                          </td>
                          <td style={{ padding: "0.5rem 0.75rem", textAlign: "right", fontFamily: "monospace" }}>
                            {doc.processing_duration_ms < 1000
                              ? `${doc.processing_duration_ms} ms`
                              : `${(doc.processing_duration_ms / 1000).toFixed(1)} s`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Methodology Note */}
              <div
                style={{
                  marginTop: "1.5rem",
                  padding: "1rem",
                  fontSize: "0.75rem",
                  color: "var(--muted)",
                  lineHeight: 1.5,
                  borderRadius: "0.5rem",
                  backgroundColor: "#fafafa",
                  border: "1px solid var(--border)",
                }}
              >
                <strong>Methodology:</strong> Environmental impact estimates are based on measured processing duration, AI token usage,
                and file transfer sizes. Energy calculations use published power profiles for AWS Lambda and Fargate compute.
                Carbon intensity uses regional grid emission factors. Water usage is derived from AWS datacenter Water Usage
                Effectiveness (WUE). These are estimates and may differ from actual resource consumption.
              </div>
            </>
          )}
        </>
      )}

      {/* ── Add to Collection Modal ──────────────────────────────── */}
      {showAddToCollection && addDocKey && (
        <div
          onClick={() => setShowAddToCollection(false)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 50,
            padding: "2rem",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              padding: "1.5rem",
              maxWidth: "24rem",
              width: "100%",
              boxShadow: "0 25px 50px rgba(0,0,0,0.25)",
            }}
          >
            <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.25rem" }}>
              Add to Collection
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "1rem" }}>
              <span style={{ fontFamily: "monospace" }}>{addDocKey.filename}</span>
            </p>

            {userCollections.length === 0 ? (
              <div style={{ textAlign: "center", padding: "1rem", color: "var(--muted)", fontSize: "0.85rem" }}>
                No collections yet.{" "}
                <a href="/collections" style={{ color: "var(--primary)" }}>
                  Create one first
                </a>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "16rem", overflowY: "auto" }}>
                {userCollections.map((col) => (
                  <button
                    key={col.collection_id}
                    onClick={() => handleAddToCollection(col.collection_id)}
                    disabled={addingToCollection === col.collection_id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "0.625rem 0.75rem",
                      border: "1px solid var(--border)",
                      borderRadius: "0.375rem",
                      backgroundColor: "white",
                      cursor: addingToCollection === col.collection_id ? "not-allowed" : "pointer",
                      textAlign: "left",
                      fontSize: "0.85rem",
                      fontWeight: 500,
                      transition: "border-color 0.1s",
                    }}
                  >
                    <span>{col.name}</span>
                    <span style={{ fontSize: "0.75rem", color: "var(--primary)", fontWeight: 600 }}>
                      {addingToCollection === col.collection_id ? "Adding..." : "Add"}
                    </span>
                  </button>
                ))}
              </div>
            )}

            <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={() => setShowAddToCollection(false)}
                style={{
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.8rem",
                  backgroundColor: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Lightbox ────────────────────────────────────────────── */}
      {lightboxUrl && (
        <div
          onClick={() => setLightboxUrl(null)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.85)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 50,
            cursor: "zoom-out",
            padding: "2rem",
          }}
        >
          <img
            src={lightboxUrl}
            alt="Full size preview"
            style={{
              maxWidth: "90vw",
              maxHeight: "90vh",
              objectFit: "contain",
              borderRadius: "0.5rem",
              boxShadow: "0 25px 50px rgba(0, 0, 0, 0.5)",
            }}
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={() => setLightboxUrl(null)}
            style={{
              position: "absolute",
              top: "1rem",
              right: "1.5rem",
              color: "white",
              backgroundColor: "rgba(255, 255, 255, 0.1)",
              border: "none",
              borderRadius: "50%",
              width: "2.5rem",
              height: "2.5rem",
              fontSize: "1.25rem",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            &times;
          </button>
        </div>
      )}

      {/* ── Delete Confirmation Modal ───────────────────────────── */}
      {showDeleteModal && (
        <div
          onClick={() => setShowDeleteModal(false)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 50,
            padding: "2rem",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              padding: "1.5rem",
              maxWidth: "28rem",
              width: "100%",
              boxShadow: "0 25px 50px rgba(0, 0, 0, 0.25)",
            }}
          >
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "0.5rem", color: "var(--danger)" }}>
              Delete Job Permanently
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--muted)", lineHeight: 1.5, marginBottom: "1rem" }}>
              This will permanently delete the job record, all uploaded documents,
              and all processing results. This action cannot be undone.
            </p>
            <p style={{ fontSize: "0.85rem", marginBottom: "0.5rem" }}>
              To confirm, type{" "}
              <strong style={{ fontFamily: "monospace", backgroundColor: "#f3f4f6", padding: "0.125rem 0.375rem", borderRadius: "0.25rem" }}>
                {confirmationId}
              </strong>
            </p>
            <input
              type="text"
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              placeholder="Type the ID to confirm"
              autoFocus
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                border: "1px solid var(--border)",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontFamily: "monospace",
                boxSizing: "border-box",
                marginBottom: "1rem",
                outline: "none",
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && deleteConfirmText === confirmationId) {
                  handleDelete();
                }
              }}
            />
            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
              <button
                onClick={() => setShowDeleteModal(false)}
                disabled={deleting}
                style={{
                  padding: "0.5rem 1rem",
                  fontSize: "0.8rem",
                  backgroundColor: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleteConfirmText !== confirmationId || deleting}
                style={{
                  padding: "0.5rem 1rem",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "white",
                  backgroundColor: deleteConfirmText === confirmationId && !deleting
                    ? "var(--danger)"
                    : "#9ca3af",
                  border: "none",
                  borderRadius: "0.375rem",
                  cursor: deleteConfirmText === confirmationId && !deleting
                    ? "pointer"
                    : "not-allowed",
                  transition: "background-color 0.15s",
                }}
              >
                {deleting ? "Deleting..." : "Delete Permanently"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── HathiTrust Package Modal ────────────────────────────── */}
      {showHathiTrustModal && (
        <div
          onClick={() => { if (!hathiTrustLoading) setShowHathiTrustModal(false); }}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 50,
            padding: "2rem",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: "white",
              borderRadius: "0.75rem",
              padding: "1.5rem",
              maxWidth: "56rem",
              width: "100%",
              maxHeight: "90vh",
              display: "flex",
              flexDirection: "column",
              boxShadow: "0 25px 50px rgba(0, 0, 0, 0.25)",
            }}
          >
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "0.5rem", color: "#7c3aed" }}>
              Create HathiTrust Package
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--muted)", lineHeight: 1.5, marginBottom: "1rem" }}>
              Configure scanner settings and assign structural labels to each page.
              The package will contain a YAML manifest and JP2 images.
            </p>

            {/* Scanner settings grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Scanner Make
                </label>
                <input
                  type="text"
                  value={htScannerMake}
                  onChange={(e) => setHtScannerMake(e.target.value)}
                  disabled={hathiTrustLoading}
                  style={{
                    width: "100%",
                    padding: "0.375rem 0.5rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Scanner Model
                </label>
                <input
                  type="text"
                  value={htScannerModel}
                  onChange={(e) => setHtScannerModel(e.target.value)}
                  disabled={hathiTrustLoading}
                  style={{
                    width: "100%",
                    padding: "0.375rem 0.5rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Resolution (DPI)
                </label>
                <input
                  type="number"
                  value={htResolution}
                  onChange={(e) => setHtResolution(parseInt(e.target.value) || 400)}
                  disabled={hathiTrustLoading}
                  min={72}
                  max={1200}
                  style={{
                    width: "100%",
                    padding: "0.375rem 0.5rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Capture Date
                </label>
                <input
                  type="date"
                  value={htCaptureDate}
                  onChange={(e) => setHtCaptureDate(e.target.value)}
                  disabled={hathiTrustLoading}
                  style={{
                    width: "100%",
                    padding: "0.375rem 0.5rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Scanning Order
                </label>
                <select
                  value={htScanningOrder}
                  onChange={(e) => setHtScanningOrder(e.target.value)}
                  disabled={hathiTrustLoading}
                  style={{
                    width: "100%",
                    padding: "0.375rem 0.5rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                    backgroundColor: "white",
                  }}
                >
                  <option value="left-to-right">Left to Right</option>
                  <option value="right-to-left">Right to Left</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.25rem" }}>
                  Reading Order
                </label>
                <select
                  value={htReadingOrder}
                  onChange={(e) => setHtReadingOrder(e.target.value)}
                  disabled={hathiTrustLoading}
                  style={{
                    width: "100%",
                    padding: "0.375rem 0.5rem",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                    backgroundColor: "white",
                  }}
                >
                  <option value="left-to-right">Left to Right</option>
                  <option value="right-to-left">Right to Left</option>
                </select>
              </div>
            </div>

            {/* Quick-fill toolbar */}
            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
              <button
                onClick={htSetFirstAsFrontCover}
                disabled={hathiTrustLoading || htPages.length === 0}
                style={{
                  padding: "0.25rem 0.625rem",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  color: "#7c3aed",
                  backgroundColor: "#f3f0ff",
                  border: "1px solid #7c3aed22",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Set first as Front Cover
              </button>
              <button
                onClick={htSetLastAsBackCover}
                disabled={hathiTrustLoading || htPages.length === 0}
                style={{
                  padding: "0.25rem 0.625rem",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  color: "#7c3aed",
                  backgroundColor: "#f3f0ff",
                  border: "1px solid #7c3aed22",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Set last as Back Cover
              </button>
              <button
                onClick={htAutoNumberPages}
                disabled={hathiTrustLoading || htPages.length === 0}
                style={{
                  padding: "0.25rem 0.625rem",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  color: "#7c3aed",
                  backgroundColor: "#f3f0ff",
                  border: "1px solid #7c3aed22",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Auto-number pages
              </button>
              <button
                onClick={htClearAllLabels}
                disabled={hathiTrustLoading || htPages.length === 0}
                style={{
                  padding: "0.25rem 0.625rem",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  color: "#dc2626",
                  backgroundColor: "#fef2f2",
                  border: "1px solid #dc262622",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Clear all labels
              </button>
            </div>

            {/* Page table */}
            <div style={{ flex: 1, overflow: "auto", border: "1px solid var(--border)", borderRadius: "0.5rem", marginBottom: "1rem" }}>
              {htPages.length === 0 ? (
                <div style={{ textAlign: "center", padding: "2rem", color: "var(--muted)", fontSize: "0.85rem" }}>
                  No output images found for this job. Process the job first to generate output images.
                </div>
              ) : (
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)", backgroundColor: "#fafafa", position: "sticky", top: 0, zIndex: 1 }}>
                      <th style={{ textAlign: "left", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase", width: "3rem" }}>#</th>
                      <th style={{ textAlign: "left", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase" }}>Filename</th>
                      <th style={{ textAlign: "left", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase", width: "12rem" }}>Label</th>
                      <th style={{ textAlign: "left", padding: "0.5rem 0.75rem", fontWeight: 600, color: "var(--muted)", fontSize: "0.7rem", textTransform: "uppercase", width: "6rem" }}>Page #</th>
                    </tr>
                  </thead>
                  <tbody>
                    {htPages.map((page, i) => (
                      <tr key={page.s3_key} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "0.375rem 0.75rem", color: "var(--muted)", fontSize: "0.7rem" }}>{i + 1}</td>
                        <td style={{ padding: "0.375rem 0.75rem", fontFamily: "monospace", fontSize: "0.75rem" }}>{page.filename}</td>
                        <td style={{ padding: "0.375rem 0.75rem" }}>
                          <select
                            value={page.label}
                            onChange={(e) => updateHtPage(i, "label", e.target.value)}
                            disabled={hathiTrustLoading}
                            style={{
                              width: "100%",
                              padding: "0.25rem 0.375rem",
                              border: "1px solid var(--border)",
                              borderRadius: "0.25rem",
                              fontSize: "0.75rem",
                              boxSizing: "border-box",
                              backgroundColor: "white",
                            }}
                          >
                            {HT_LABELS.map((l) => (
                              <option key={l} value={l}>{l || "(none)"}</option>
                            ))}
                          </select>
                        </td>
                        <td style={{ padding: "0.375rem 0.75rem" }}>
                          <input
                            type="text"
                            value={page.orderlabel}
                            onChange={(e) => updateHtPage(i, "orderlabel", e.target.value)}
                            disabled={hathiTrustLoading}
                            placeholder=""
                            style={{
                              width: "100%",
                              padding: "0.25rem 0.375rem",
                              border: "1px solid var(--border)",
                              borderRadius: "0.25rem",
                              fontSize: "0.75rem",
                              boxSizing: "border-box",
                            }}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>
                {htPages.length} page{htPages.length !== 1 ? "s" : ""}
              </span>
              <div style={{ display: "flex", gap: "0.75rem" }}>
                <button
                  onClick={() => setShowHathiTrustModal(false)}
                  disabled={hathiTrustLoading}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    backgroundColor: "transparent",
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    cursor: hathiTrustLoading ? "not-allowed" : "pointer",
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateHathiTrustPackage}
                  disabled={htPages.length === 0 || hathiTrustLoading}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    fontWeight: 600,
                    color: "white",
                    backgroundColor: htPages.length > 0 && !hathiTrustLoading ? "#7c3aed" : "#9ca3af",
                    border: "none",
                    borderRadius: "0.375rem",
                    cursor: htPages.length > 0 && !hathiTrustLoading ? "pointer" : "not-allowed",
                    transition: "background-color 0.15s",
                  }}
                >
                  {hathiTrustLoading ? "Creating Package..." : "Create Package"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
