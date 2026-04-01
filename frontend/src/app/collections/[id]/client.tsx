"use client";

import "@/lib/amplify-config";
import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter, usePathname } from "next/navigation";
import { apiGet, apiPost, apiPut, apiDelete } from "@/lib/api";
import Link from "next/link";

interface CollectionDoc {
  s3_key: string;
  filename: string;
  job_id: string;
  source_type: string;
  size: number;
  url?: string;
}

interface CollectionDetail {
  collection_id: string;
  user_id: string;
  name: string;
  description: string;
  documents: CollectionDoc[];
  document_count: number;
  created_at: string;
  updated_at: string;
}

interface CollectionImpactSummary {
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

interface CollectionDocImpact {
  document_key: string;
  task_type: string;
  carbon_g_co2e: number;
  energy_kwh: number;
  water_ml: number;
}

interface CollectionImpactComparisons {
  km_driven: number;
  hours_led_bulb: number;
  smartphone_charges: number;
  google_searches: number;
}

interface CollectionImpact {
  collection_id: string;
  summary: CollectionImpactSummary | null;
  per_document: CollectionDocImpact[];
  comparisons: CollectionImpactComparisons | null;
}

export default function CollectionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const pathname = usePathname();
  // In static export, useParams returns the build-time placeholder "_".
  // usePathname() is reactive and always reflects the real current URL.
  const rawId = params.id as string;
  const collectionId = rawId === "_"
    ? (pathname.split("/").filter(Boolean)[1] || "_")
    : rawId;

  const [collection, setCollection] = useState<CollectionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [saving, setSaving] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [envImpact, setEnvImpact] = useState<CollectionImpact | null>(null);
  const [envCollapsed, setEnvCollapsed] = useState(true);

  const fetchCollection = useCallback(async () => {
    try {
      const data = await apiGet<{ collection: CollectionDetail }>(
        `/collections/${collectionId}`
      );
      setCollection(data.collection);
    } catch {
      // handle
    }
  }, [collectionId]);

  const fetchEnvImpact = useCallback(async () => {
    try {
      const data = await apiGet<CollectionImpact>(
        `/collections/${collectionId}/environmental-impact`
      );
      setEnvImpact(data);
    } catch {
      // Not available
    }
  }, [collectionId]);

  useEffect(() => {
    Promise.all([fetchCollection(), fetchEnvImpact()]).then(() => setLoading(false));
  }, [fetchCollection, fetchEnvImpact]);

  const isImage = (filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase() || "";
    return ["jpg", "jpeg", "png", "gif", "webp"].includes(ext);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleSave = async () => {
    if (!editName.trim()) return;
    setSaving(true);
    try {
      await apiPut(`/collections/${collectionId}`, {
        name: editName.trim(),
        description: editDesc.trim(),
      });
      await fetchCollection();
      setEditing(false);
    } catch (err) {
      alert(`Failed to save: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSaving(false);
    }
  };

  const handleRemoveDoc = async (s3Key: string) => {
    try {
      await apiPost(`/collections/${collectionId}/documents`, {
        action: "remove",
        documents: [{ s3_key: s3Key }],
      });
      await fetchCollection();
    } catch (err) {
      alert(`Failed to remove: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const data = await apiGet<{ download_url: string }>(
        `/collections/${collectionId}/download`
      );
      window.open(data.download_url, "_blank");
    } catch (err) {
      alert(`Download failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDownloading(false);
    }
  };

  const handleDelete = async () => {
    if (!collection || deleteConfirm !== collection.name) return;
    setDeleting(true);
    try {
      await apiDelete(`/collections/${collectionId}`);
      router.replace("/collections");
    } catch (err) {
      alert(`Failed to delete: ${err instanceof Error ? err.message : String(err)}`);
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: "4rem", textAlign: "center", color: "var(--muted)" }}>
        Loading...
      </div>
    );
  }

  if (!collection) {
    return (
      <div style={{ padding: "4rem", textAlign: "center" }}>
        <p style={{ color: "var(--muted)", marginBottom: "1rem" }}>Collection not found.</p>
        <Link href="/collections" style={{ color: "var(--primary)" }}>
          Back to Collections
        </Link>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: "72rem", margin: "0 auto", padding: "2rem" }}>
      {/* Nav */}
      <div style={{ marginBottom: "1.5rem" }}>
        <Link
          href="/collections"
          style={{ color: "var(--muted)", textDecoration: "none", fontSize: "0.875rem" }}
        >
          &larr; Collections
        </Link>
      </div>

      {/* Header card */}
      <div
        style={{
          border: "1px solid var(--border)",
          borderRadius: "0.75rem",
          padding: "1.5rem",
          marginBottom: "1.5rem",
          backgroundColor: "white",
        }}
      >
        {editing ? (
          <div>
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                border: "1px solid var(--border)",
                borderRadius: "0.375rem",
                fontSize: "1.1rem",
                fontWeight: 600,
                marginBottom: "0.5rem",
                boxSizing: "border-box",
                outline: "none",
              }}
            />
            <textarea
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              placeholder="Description (optional)"
              rows={2}
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                border: "1px solid var(--border)",
                borderRadius: "0.375rem",
                fontSize: "0.85rem",
                marginBottom: "0.75rem",
                boxSizing: "border-box",
                outline: "none",
                resize: "vertical",
              }}
            />
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                onClick={handleSave}
                disabled={!editName.trim() || saving}
                style={{
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "white",
                  backgroundColor: !editName.trim() || saving ? "#9ca3af" : "var(--primary)",
                  border: "none",
                  borderRadius: "0.375rem",
                  cursor: !editName.trim() || saving ? "not-allowed" : "pointer",
                }}
              >
                {saving ? "Saving..." : "Save"}
              </button>
              <button
                onClick={() => setEditing(false)}
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
        ) : (
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <h1 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                {collection.name}
              </h1>
              {collection.description && (
                <p style={{ fontSize: "0.85rem", color: "var(--muted)", marginBottom: "0.5rem" }}>
                  {collection.description}
                </p>
              )}
              <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>
                {collection.document_count} document{collection.document_count !== 1 ? "s" : ""}
                {" | "}Created {new Date(collection.created_at).toLocaleDateString()}
              </div>
            </div>
            <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0 }}>
              <button
                onClick={() => {
                  setEditName(collection.name);
                  setEditDesc(collection.description);
                  setEditing(true);
                }}
                style={{
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.75rem",
                  fontWeight: 500,
                  color: "var(--muted)",
                  backgroundColor: "transparent",
                  border: "1px solid var(--border)",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Edit
              </button>
              {collection.document_count > 0 && (
                <button
                  onClick={handleDownload}
                  disabled={downloading}
                  style={{
                    padding: "0.375rem 0.75rem",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "white",
                    backgroundColor: downloading ? "#9ca3af" : "var(--primary)",
                    border: "none",
                    borderRadius: "0.375rem",
                    cursor: downloading ? "not-allowed" : "pointer",
                  }}
                >
                  {downloading ? "Preparing..." : "Download All"}
                </button>
              )}
              <button
                onClick={() => { setDeleteConfirm(""); setShowDeleteModal(true); }}
                style={{
                  padding: "0.375rem 0.75rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  color: "var(--danger)",
                  backgroundColor: "transparent",
                  border: "1px solid var(--danger)",
                  borderRadius: "0.375rem",
                  cursor: "pointer",
                }}
              >
                Delete
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Environmental Impact Summary */}
      {envImpact?.summary && (
        <div
          style={{
            border: "1px solid var(--border)",
            borderRadius: "0.75rem",
            marginBottom: "1.5rem",
            backgroundColor: "white",
            overflow: "hidden",
          }}
        >
          <button
            onClick={() => setEnvCollapsed(!envCollapsed)}
            style={{
              width: "100%",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "0.875rem 1.25rem",
              backgroundColor: "transparent",
              border: "none",
              cursor: "pointer",
              textAlign: "left",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--foreground)" }}>
                Environmental Impact
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem", padding: "0.125rem 0.5rem", fontSize: "0.65rem", fontWeight: 600, color: "#16a34a", backgroundColor: "#f0fdf4", borderRadius: "9999px", border: "1px solid #16a34a22" }}>
                {envImpact.summary.total_carbon_g_co2e < 1
                  ? `${(envImpact.summary.total_carbon_g_co2e * 1000).toFixed(1)} mg CO\u2082e`
                  : `${envImpact.summary.total_carbon_g_co2e.toFixed(2)} g CO\u2082e`}
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem", padding: "0.125rem 0.5rem", fontSize: "0.65rem", fontWeight: 600, color: "#d97706", backgroundColor: "#fffbeb", borderRadius: "9999px", border: "1px solid #d9770622" }}>
                {envImpact.summary.total_energy_kwh < 0.001
                  ? `${(envImpact.summary.total_energy_kwh * 1_000_000).toFixed(1)} mWh`
                  : `${(envImpact.summary.total_energy_kwh * 1000).toFixed(2)} Wh`}
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem", padding: "0.125rem 0.5rem", fontSize: "0.65rem", fontWeight: 600, color: "#2563eb", backgroundColor: "#eff6ff", borderRadius: "9999px", border: "1px solid #2563eb22" }}>
                {envImpact.summary.total_water_ml < 1
                  ? `${(envImpact.summary.total_water_ml * 1000).toFixed(1)} \u00b5L H\u2082O`
                  : `${envImpact.summary.total_water_ml.toFixed(2)} mL H\u2082O`}
              </span>
            </div>
            <span style={{ fontSize: "0.75rem", color: "var(--muted)", transition: "transform 0.2s", transform: envCollapsed ? "rotate(0deg)" : "rotate(180deg)" }}>
              &#9660;
            </span>
          </button>
          {!envCollapsed && envImpact.comparisons && (
            <div style={{ padding: "0 1.25rem 1rem", borderTop: "1px solid var(--border)" }}>
              <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", margin: "0.75rem 0 0.5rem" }}>
                Equivalent To
              </div>
              <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
                {[
                  { value: envImpact.comparisons.km_driven, label: "km driven by car" },
                  { value: envImpact.comparisons.hours_led_bulb, label: "hrs of LED bulb" },
                  { value: envImpact.comparisons.smartphone_charges, label: "smartphone charges" },
                  { value: envImpact.comparisons.google_searches, label: "Google searches" },
                ].map((c) => (
                  <div key={c.label} style={{ fontSize: "0.8rem" }}>
                    <span style={{ fontWeight: 700 }}>{c.value < 0.01 ? "<0.01" : c.value.toFixed(2)}</span>
                    {" "}
                    <span style={{ color: "var(--muted)" }}>{c.label}</span>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: "0.75rem", fontSize: "0.675rem", color: "var(--muted)", lineHeight: 1.4 }}>
                Estimates based on measured processing duration, AI token usage, and AWS regional emission factors.
              </div>
            </div>
          )}
        </div>
      )}

      {/* Documents grid */}
      {collection.documents.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "3rem",
            border: "1px dashed var(--border)",
            borderRadius: "0.75rem",
            color: "var(--muted)",
          }}
        >
          <p style={{ marginBottom: "0.5rem" }}>No documents in this collection yet.</p>
          <p style={{ fontSize: "0.8rem" }}>
            Add documents from a job&apos;s Documents tab.
          </p>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(14rem, 1fr))",
            gap: "1rem",
          }}
        >
          {collection.documents.map((doc) => (
            <div
              key={doc.s3_key}
              style={{
                border: "1px solid var(--border)",
                borderRadius: "0.5rem",
                overflow: "hidden",
                backgroundColor: "white",
              }}
            >
              {/* Thumbnail or icon */}
              {isImage(doc.filename) && doc.url ? (
                <img
                  src={doc.url}
                  alt={doc.filename}
                  crossOrigin="anonymous"
                  style={{
                    width: "100%",
                    height: "10rem",
                    objectFit: "cover",
                    display: "block",
                    cursor: "zoom-in",
                  }}
                  onClick={() => setLightboxUrl(doc.url || null)}
                />
              ) : (
                <div
                  style={{
                    height: "10rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: "#f9fafb",
                    fontSize: "2rem",
                    opacity: 0.3,
                  }}
                >
                  &#128196;
                </div>
              )}

              {/* Info bar */}
              <div style={{ padding: "0.5rem 0.75rem" }}>
                <div
                  style={{
                    fontFamily: "monospace",
                    fontSize: "0.75rem",
                    fontWeight: 500,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    marginBottom: "0.25rem",
                  }}
                  title={doc.filename}
                >
                  {doc.filename}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.65rem", color: "var(--muted)" }}>
                    {formatSize(doc.size)}
                    {" | "}
                    <span
                      style={{
                        padding: "0.0625rem 0.25rem",
                        backgroundColor: doc.source_type === "output" ? "#dcfce7" : "#f3f4f6",
                        borderRadius: "0.25rem",
                        color: doc.source_type === "output" ? "#166534" : "var(--muted)",
                      }}
                    >
                      {doc.source_type}
                    </span>
                  </span>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    {/* Environmental impact badge */}
                    {(() => {
                      if (!envImpact?.per_document) return null;
                      const docMetrics = envImpact.per_document.filter((m) => m.document_key === doc.s3_key);
                      if (docMetrics.length === 0) return null;
                      const totalCarbon = docMetrics.reduce((sum, m) => sum + m.carbon_g_co2e, 0);
                      return (
                        <span
                          style={{
                            fontSize: "0.6rem",
                            fontWeight: 600,
                            color: "#16a34a",
                            backgroundColor: "#f0fdf4",
                            padding: "0.0625rem 0.3rem",
                            borderRadius: "0.25rem",
                            border: "1px solid #16a34a22",
                          }}
                          title={`Estimated carbon footprint: ${totalCarbon < 1 ? `${(totalCarbon * 1000).toFixed(1)} mg` : `${totalCarbon.toFixed(2)} g`} CO\u2082e`}
                        >
                          {totalCarbon < 0.001 ? `${(totalCarbon * 1_000_000).toFixed(0)} \u00b5g` : totalCarbon < 1 ? `${(totalCarbon * 1000).toFixed(1)} mg` : `${totalCarbon.toFixed(2)} g`} CO&#8322;e
                        </span>
                      );
                    })()}
                    {doc.url && (
                      <a
                        href={doc.url}
                        download={doc.filename}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          fontSize: "0.65rem",
                          color: "var(--primary)",
                          textDecoration: "none",
                          fontWeight: 500,
                        }}
                      >
                        DL
                      </a>
                    )}
                    <button
                      onClick={() => handleRemoveDoc(doc.s3_key)}
                      style={{
                        fontSize: "0.75rem",
                        color: "var(--muted)",
                        backgroundColor: "transparent",
                        border: "none",
                        cursor: "pointer",
                        padding: 0,
                        lineHeight: 1,
                      }}
                      title="Remove from collection"
                    >
                      &times;
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Delete modal */}
      {showDeleteModal && (
        <div
          onClick={() => setShowDeleteModal(false)}
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
              maxWidth: "28rem",
              width: "100%",
              boxShadow: "0 25px 50px rgba(0,0,0,0.25)",
            }}
          >
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "0.5rem", color: "var(--danger)" }}>
              Delete Collection
            </h3>
            <p style={{ fontSize: "0.85rem", color: "var(--muted)", lineHeight: 1.5, marginBottom: "1rem" }}>
              This will permanently delete the collection. The original files in S3 will not be affected.
            </p>
            <p style={{ fontSize: "0.85rem", marginBottom: "0.5rem" }}>
              Type{" "}
              <strong style={{ fontFamily: "monospace", backgroundColor: "#f3f4f6", padding: "0.125rem 0.375rem", borderRadius: "0.25rem" }}>
                {collection.name}
              </strong>
              {" "}to confirm.
            </p>
            <input
              type="text"
              value={deleteConfirm}
              onChange={(e) => setDeleteConfirm(e.target.value)}
              autoFocus
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                border: "1px solid var(--border)",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                boxSizing: "border-box",
                marginBottom: "1rem",
                outline: "none",
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && deleteConfirm === collection.name) handleDelete();
              }}
            />
            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
              <button
                onClick={() => setShowDeleteModal(false)}
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
                disabled={deleteConfirm !== collection.name || deleting}
                style={{
                  padding: "0.5rem 1rem",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "white",
                  backgroundColor: deleteConfirm === collection.name && !deleting ? "var(--danger)" : "#9ca3af",
                  border: "none",
                  borderRadius: "0.375rem",
                  cursor: deleteConfirm === collection.name && !deleting ? "pointer" : "not-allowed",
                }}
              >
                {deleting ? "Deleting..." : "Delete Permanently"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox */}
      {lightboxUrl && (
        <div
          onClick={() => setLightboxUrl(null)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.85)",
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
            alt="Full size"
            crossOrigin="anonymous"
            style={{ maxWidth: "90vw", maxHeight: "90vh", objectFit: "contain", borderRadius: "0.5rem" }}
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={() => setLightboxUrl(null)}
            style={{
              position: "absolute",
              top: "1rem",
              right: "1.5rem",
              color: "white",
              backgroundColor: "rgba(255,255,255,0.1)",
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
    </div>
  );
}
