"use client";

import "@/lib/amplify-config";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter, usePathname } from "next/navigation";
import { apiGet, apiPost, apiDelete } from "@/lib/api";
import Link from "next/link";

interface Entity { text: string; type: string; count: number; documents: string[]; }
interface Edge { source: string; target: string; weight: number; documents: string[]; }
interface Location { name: string; lat: number; lon: number; display_name: string; count: number; documents: string[]; }
interface WordFreq { word: string; count: number; }
interface DocCoord { doc_key: string; filename: string; x: number; y: number; }
interface TimelineEvent { job_id: string; custom_id?: string; status: string; task_type?: string; ocr_engine?: string; total_documents?: number; processed_documents?: number; created_at: string; updated_at?: string; }
interface SummaryStats { total_docs: number; total_chars: number; total_words: number; unique_entities: number; entity_type_counts: Record<string, number>; ocr_engine_counts: Record<string, number>; top_entities: { text: string; type: string; count: number }[]; source_count: number; }
interface Source { type: string; id: string; name: string; }

interface AnalysisDetail {
  analysis_id: string; user_id: string; name: string; description: string; status: string;
  sources: Source[]; entities: Entity[]; entity_edges: Edge[]; locations: Location[];
  word_frequencies: WordFreq[]; doc_coordinates: DocCoord[]; timeline_events: TimelineEvent[];
  summary_stats: SummaryStats; created_at: string; updated_at: string;
}

type Tab = "summary" | "network" | "map" | "words" | "timeline" | "clusters";

const TYPE_COLORS: Record<string, string> = { PER: "#3b82f6", LOC: "#22c55e", ORG: "#f59e0b", MISC: "#8b5cf6" };
const TYPE_LABELS: Record<string, string> = { PER: "Person", LOC: "Location", ORG: "Organization", MISC: "Other" };

export default function AnalysisWorkspacePage() {
  const params = useParams();
  const router = useRouter();
  const pathname = usePathname();
  // In static export, useParams returns the build-time placeholder "_".
  // usePathname() is reactive and always reflects the real current URL.
  const rawId = params.id as string;
  const analysisId = rawId === "_"
    ? (pathname.split("/").filter(Boolean)[1] || "_")
    : rawId;

  const [analysis, setAnalysis] = useState<AnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("summary");
  const [deleting, setDeleting] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInitRef = useRef(false);
  const networkRef = useRef<HTMLCanvasElement>(null);

  const fetchAnalysis = useCallback(async () => {
    try {
      const data = await apiGet<{ analysis: AnalysisDetail }>(`/analyses/${analysisId}`);
      setAnalysis(data.analysis);
    } catch { /* handle */ }
  }, [analysisId]);

  useEffect(() => { fetchAnalysis().then(() => setLoading(false)); }, [fetchAnalysis]);

  // Poll while running
  useEffect(() => {
    if (!analysis || analysis.status !== "RUNNING") return;
    const iv = setInterval(fetchAnalysis, 3000);
    return () => clearInterval(iv);
  }, [analysis, fetchAnalysis]);

  const handleRun = async () => {
    setRunning(true);
    try {
      await apiPost(`/analyses/${analysisId}/run`, {});
      await fetchAnalysis();
    } catch (err) {
      alert(`Failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally { setRunning(false); }
  };

  const handleDelete = async () => {
    if (!confirm("Delete this analysis permanently?")) return;
    setDeleting(true);
    try { await apiDelete(`/analyses/${analysisId}`); router.replace("/analyses"); }
    catch { setDeleting(false); }
  };

  // ── Leaflet Map ──────────────────────────────────────────────
  useEffect(() => {
    if (activeTab !== "map" || !analysis?.locations?.length || !mapRef.current || mapInitRef.current) return;
    mapInitRef.current = true;

    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
    document.head.appendChild(link);

    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
    script.onload = () => {
      const L = (window as any).L;
      if (!L || !mapRef.current) return;
      const map = L.map(mapRef.current).setView([30, 0], 2);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap",
      }).addTo(map);
      for (const loc of analysis.locations) {
        const r = Math.min(20, Math.max(6, loc.count * 3));
        L.circleMarker([loc.lat, loc.lon], { radius: r, color: "#3b82f6", fillColor: "#3b82f6", fillOpacity: 0.5, weight: 1 })
          .addTo(map)
          .bindPopup(`<b>${loc.name}</b><br>${loc.count} mention${loc.count > 1 ? "s" : ""}<br><small>${loc.display_name}</small>`);
      }
      if (analysis.locations.length > 0) {
        const bounds = analysis.locations.map((l) => [l.lat, l.lon]);
        map.fitBounds(bounds, { padding: [30, 30] });
      }
    };
    document.head.appendChild(script);
  }, [activeTab, analysis]);

  // ── Network Graph (Canvas) ───────────────────────────────────
  useEffect(() => {
    if (activeTab !== "network" || !analysis?.entities?.length || !networkRef.current) return;
    const canvas = networkRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width = canvas.parentElement?.clientWidth || 800;
    const H = canvas.height = 500;

    // Build nodes and edges
    const topEntities = analysis.entities.slice(0, 50);
    const nameSet = new Set(topEntities.map(e => e.text));
    const edges = analysis.entity_edges.filter(e => nameSet.has(e.source) && nameSet.has(e.target)).slice(0, 150);

    interface Node { x: number; y: number; vx: number; vy: number; text: string; type: string; count: number; r: number; }
    const nodes: Node[] = topEntities.map((e) => ({
      x: W / 2 + (Math.random() - 0.5) * W * 0.6,
      y: H / 2 + (Math.random() - 0.5) * H * 0.6,
      vx: 0, vy: 0,
      text: e.text, type: e.type, count: e.count,
      r: Math.min(20, Math.max(4, Math.sqrt(e.count) * 3)),
    }));
    const nodeIdx: Record<string, number> = {};
    nodes.forEach((n, i) => { nodeIdx[n.text] = i; });

    // Force simulation
    let frame = 0;
    const maxFrames = 200;
    const animate = () => {
      if (frame++ > maxFrames) { draw(); return; }

      // Repulsion
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          let dx = nodes[j].x - nodes[i].x;
          let dy = nodes[j].y - nodes[i].y;
          const d2 = dx * dx + dy * dy + 1;
          const f = 800 / d2;
          nodes[i].vx -= dx * f; nodes[i].vy -= dy * f;
          nodes[j].vx += dx * f; nodes[j].vy += dy * f;
        }
      }

      // Attraction (edges)
      for (const e of edges) {
        const si = nodeIdx[e.source], ti = nodeIdx[e.target];
        if (si === undefined || ti === undefined) continue;
        let dx = nodes[ti].x - nodes[si].x;
        let dy = nodes[ti].y - nodes[si].y;
        const d = Math.sqrt(dx * dx + dy * dy) + 1;
        const f = (d - 80) * 0.01 * e.weight;
        nodes[si].vx += dx / d * f; nodes[si].vy += dy / d * f;
        nodes[ti].vx -= dx / d * f; nodes[ti].vy -= dy / d * f;
      }

      // Center gravity
      for (const n of nodes) {
        n.vx += (W / 2 - n.x) * 0.005;
        n.vy += (H / 2 - n.y) * 0.005;
        n.vx *= 0.85; n.vy *= 0.85;
        n.x += n.vx; n.y += n.vy;
        n.x = Math.max(n.r, Math.min(W - n.r, n.x));
        n.y = Math.max(n.r, Math.min(H - n.r, n.y));
      }

      draw();
      requestAnimationFrame(animate);
    };

    const draw = () => {
      ctx.clearRect(0, 0, W, H);

      // Edges
      ctx.strokeStyle = "rgba(0,0,0,0.08)";
      for (const e of edges) {
        const si = nodeIdx[e.source], ti = nodeIdx[e.target];
        if (si === undefined || ti === undefined) continue;
        ctx.lineWidth = Math.min(4, e.weight);
        ctx.beginPath();
        ctx.moveTo(nodes[si].x, nodes[si].y);
        ctx.lineTo(nodes[ti].x, nodes[ti].y);
        ctx.stroke();
      }

      // Nodes
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = TYPE_COLORS[n.type] || "#8b5cf6";
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Labels for larger nodes
      ctx.fillStyle = "#111";
      ctx.font = "10px -apple-system, sans-serif";
      ctx.textAlign = "center";
      for (const n of nodes) {
        if (n.r >= 6) {
          ctx.fillText(n.text, n.x, n.y - n.r - 4);
        }
      }
    };

    animate();
  }, [activeTab, analysis]);

  if (loading) return <div style={{ padding: "4rem", textAlign: "center", color: "var(--muted)" }}>Loading...</div>;
  if (!analysis) return <div style={{ padding: "4rem", textAlign: "center" }}><p style={{ color: "var(--muted)" }}>Analysis not found.</p><Link href="/analyses" style={{ color: "var(--primary)" }}>Back</Link></div>;

  const stats = analysis.summary_stats || {} as SummaryStats;
  const tabs: { key: Tab; label: string }[] = [
    { key: "summary", label: "Summary" },
    { key: "network", label: "Network" },
    { key: "map", label: "Map" },
    { key: "words", label: "Words" },
    { key: "timeline", label: "Timeline" },
    { key: "clusters", label: "Clusters" },
  ];

  return (
    <div style={{ maxWidth: "80rem", margin: "0 auto", padding: "2rem" }}>
      <div style={{ marginBottom: "1.5rem" }}>
        <Link href="/analyses" style={{ color: "var(--muted)", textDecoration: "none", fontSize: "0.875rem" }}>&larr; Analyses</Link>
      </div>

      {/* Header */}
      <div style={{ border: "1px solid var(--border)", borderRadius: "0.75rem", padding: "1.5rem", marginBottom: "1.5rem", backgroundColor: "white" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.25rem" }}>{analysis.name}</h1>
            {analysis.description && <p style={{ fontSize: "0.85rem", color: "var(--muted)", marginBottom: "0.5rem" }}>{analysis.description}</p>}
            <div style={{ display: "flex", gap: "0.375rem", flexWrap: "wrap" }}>
              {analysis.sources?.map((s, i) => (
                <span key={i} style={{ fontSize: "0.65rem", padding: "0.0625rem 0.375rem", backgroundColor: s.type === "job" ? "#eff6ff" : "#f0fdf4", borderRadius: "0.25rem", color: s.type === "job" ? "#1e40af" : "#166534" }}>
                  {s.type}: {s.name}
                </span>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0 }}>
            <button onClick={handleRun} disabled={running || analysis.status === "RUNNING"} style={{ padding: "0.375rem 0.75rem", fontSize: "0.75rem", fontWeight: 600, color: "white", backgroundColor: running || analysis.status === "RUNNING" ? "#9ca3af" : "var(--primary)", border: "none", borderRadius: "0.375rem", cursor: running || analysis.status === "RUNNING" ? "not-allowed" : "pointer" }}>
              {analysis.status === "RUNNING" ? "Running..." : running ? "Starting..." : analysis.status === "COMPLETED" ? "Re-run Analysis" : "Run Analysis"}
            </button>
            <button onClick={handleDelete} disabled={deleting} style={{ padding: "0.375rem 0.75rem", fontSize: "0.75rem", fontWeight: 600, color: "var(--danger)", backgroundColor: "transparent", border: "1px solid var(--danger)", borderRadius: "0.375rem", cursor: "pointer" }}>Delete</button>
          </div>
        </div>
      </div>

      {analysis.status === "IDLE" && (
        <div style={{ textAlign: "center", padding: "3rem", border: "1px dashed var(--border)", borderRadius: "0.75rem", color: "var(--muted)" }}>
          <p style={{ marginBottom: "0.75rem" }}>Click &quot;Run Analysis&quot; to compute entity extraction, geographic mapping, word frequencies, and document clustering.</p>
        </div>
      )}

      {analysis.status === "RUNNING" && (
        <div style={{ textAlign: "center", padding: "3rem", border: "1px solid #dbeafe", borderRadius: "0.75rem", backgroundColor: "#eff6ff", color: "#1e40af" }}>
          <p style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Analysis is running...</p>
          <p style={{ fontSize: "0.8rem" }}>This may take a few minutes depending on the number of documents.</p>
        </div>
      )}

      {analysis.status === "COMPLETED" && (
        <>
          {/* Tab bar */}
          <div style={{ display: "flex", gap: 0, borderBottom: "2px solid var(--border)", marginBottom: "1.5rem" }}>
            {tabs.map((t) => (
              <button key={t.key} onClick={() => { setActiveTab(t.key); if (t.key === "map") mapInitRef.current = false; }}
                style={{ padding: "0.75rem 1.25rem", fontSize: "0.875rem", fontWeight: activeTab === t.key ? 600 : 400, color: activeTab === t.key ? "var(--primary)" : "var(--muted)", backgroundColor: "transparent", border: "none", borderBottom: activeTab === t.key ? "2px solid var(--primary)" : "2px solid transparent", marginBottom: "-2px", cursor: "pointer" }}>
                {t.label}
              </button>
            ))}
          </div>

          {/* ── Summary Tab ──────────────────────────────────── */}
          {activeTab === "summary" && (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(10rem, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
                {[
                  { label: "Documents", value: stats.total_docs || 0 },
                  { label: "Words", value: (stats.total_words || 0).toLocaleString() },
                  { label: "Characters", value: (stats.total_chars || 0).toLocaleString() },
                  { label: "Unique Entities", value: stats.unique_entities || 0 },
                  { label: "Sources", value: stats.source_count || 0 },
                ].map((s) => (
                  <div key={s.label} style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", padding: "1rem", backgroundColor: "white", textAlign: "center" }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700 }}>{s.value}</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>{s.label}</div>
                  </div>
                ))}
              </div>

              {/* Entity type breakdown */}
              {stats.entity_type_counts && Object.keys(stats.entity_type_counts).length > 0 && (
                <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", padding: "1rem", backgroundColor: "white", marginBottom: "1rem" }}>
                  <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.75rem" }}>Entity Types</h3>
                  <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                    {Object.entries(stats.entity_type_counts).map(([type, count]) => (
                      <div key={type} style={{ display: "flex", alignItems: "center", gap: "0.375rem" }}>
                        <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: TYPE_COLORS[type] || "#8b5cf6" }} />
                        <span style={{ fontSize: "0.8rem" }}>{TYPE_LABELS[type] || type}: <strong>{count}</strong></span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Top entities table */}
              {stats.top_entities?.length > 0 && (
                <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", overflow: "hidden", backgroundColor: "white" }}>
                  <div style={{ padding: "0.75rem 1rem", fontSize: "0.8rem", fontWeight: 600, borderBottom: "1px solid var(--border)", backgroundColor: "#fafafa" }}>Top Entities</div>
                  {stats.top_entities.map((e, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.5rem 1rem", borderBottom: i < stats.top_entities.length - 1 ? "1px solid var(--border)" : "none", fontSize: "0.85rem" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: TYPE_COLORS[e.type] || "#8b5cf6" }} />
                        <span style={{ fontWeight: 500 }}>{e.text}</span>
                        <span style={{ fontSize: "0.65rem", color: "var(--muted)", padding: "0.0625rem 0.25rem", backgroundColor: "#f3f4f6", borderRadius: "0.25rem" }}>{TYPE_LABELS[e.type] || e.type}</span>
                      </div>
                      <span style={{ fontWeight: 600, color: "var(--muted)" }}>{e.count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── Network Tab ──────────────────────────────────── */}
          {activeTab === "network" && (
            <div>
              {(!analysis.entities || analysis.entities.length === 0) ? (
                <div style={{ textAlign: "center", padding: "3rem", color: "var(--muted)", border: "1px dashed var(--border)", borderRadius: "0.5rem" }}>No entities found.</div>
              ) : (
                <>
                  <div style={{ display: "flex", gap: "1rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
                    {Object.entries(TYPE_COLORS).map(([type, color]) => (
                      <div key={type} style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontSize: "0.75rem" }}>
                        <div style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: color }} />
                        {TYPE_LABELS[type] || type}
                      </div>
                    ))}
                    <span style={{ fontSize: "0.7rem", color: "var(--muted)" }}>({analysis.entities.length} entities, {analysis.entity_edges.length} connections)</span>
                  </div>
                  <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", overflow: "hidden", backgroundColor: "white" }}>
                    <canvas ref={networkRef} style={{ width: "100%", display: "block" }} />
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── Map Tab ──────────────────────────────────────── */}
          {activeTab === "map" && (
            <div>
              {(!analysis.locations || analysis.locations.length === 0) ? (
                <div style={{ textAlign: "center", padding: "3rem", color: "var(--muted)", border: "1px dashed var(--border)", borderRadius: "0.5rem" }}>No locations found.</div>
              ) : (
                <>
                  <div style={{ fontSize: "0.75rem", color: "var(--muted)", marginBottom: "0.5rem" }}>{analysis.locations.length} locations mapped</div>
                  <div ref={mapRef} style={{ height: "28rem", borderRadius: "0.5rem", border: "1px solid var(--border)" }} />
                  <div style={{ marginTop: "1rem", border: "1px solid var(--border)", borderRadius: "0.5rem", overflow: "hidden", backgroundColor: "white" }}>
                    <div style={{ padding: "0.625rem 0.75rem", fontSize: "0.75rem", fontWeight: 600, borderBottom: "1px solid var(--border)", backgroundColor: "#fafafa" }}>Locations</div>
                    {analysis.locations.map((loc, i) => (
                      <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "0.375rem 0.75rem", borderBottom: i < analysis.locations.length - 1 ? "1px solid var(--border)" : "none", fontSize: "0.8rem" }}>
                        <span style={{ fontWeight: 500 }}>{loc.name}</span>
                        <span style={{ color: "var(--muted)" }}>{loc.count} mention{loc.count > 1 ? "s" : ""}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── Words Tab ────────────────────────────────────── */}
          {activeTab === "words" && (
            <div>
              {(!analysis.word_frequencies || analysis.word_frequencies.length === 0) ? (
                <div style={{ textAlign: "center", padding: "3rem", color: "var(--muted)", border: "1px dashed var(--border)", borderRadius: "0.5rem" }}>No word data.</div>
              ) : (
                <>
                  {/* Bar chart */}
                  <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", padding: "1rem", backgroundColor: "white", marginBottom: "1rem" }}>
                    <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.75rem" }}>Top 30 Words</h3>
                    {analysis.word_frequencies.slice(0, 30).map((w, i) => {
                      const maxCount = analysis.word_frequencies[0]?.count || 1;
                      const pct = (w.count / maxCount) * 100;
                      return (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.25rem" }}>
                          <span style={{ fontFamily: "monospace", fontSize: "0.75rem", width: "7rem", textAlign: "right", flexShrink: 0 }}>{w.word}</span>
                          <div style={{ flex: 1, height: "1.25rem", backgroundColor: "#f3f4f6", borderRadius: "0.25rem", overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${pct}%`, backgroundColor: "var(--primary)", borderRadius: "0.25rem", transition: "width 0.3s" }} />
                          </div>
                          <span style={{ fontSize: "0.7rem", color: "var(--muted)", width: "3rem", textAlign: "right", flexShrink: 0 }}>{w.count}</span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Word cloud (pure CSS) */}
                  <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", padding: "1.5rem", backgroundColor: "white" }}>
                    <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.75rem" }}>Word Cloud</h3>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem", justifyContent: "center" }}>
                      {analysis.word_frequencies.slice(0, 60).map((w, i) => {
                        const maxCount = analysis.word_frequencies[0]?.count || 1;
                        const size = 0.6 + (w.count / maxCount) * 1.8;
                        const opacity = 0.4 + (w.count / maxCount) * 0.6;
                        return (
                          <span key={i} style={{ fontSize: `${size}rem`, fontWeight: size > 1.2 ? 700 : 400, color: `rgba(37, 99, 235, ${opacity})`, lineHeight: 1.2, padding: "0.125rem 0.25rem" }}>
                            {w.word}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── Timeline Tab ─────────────────────────────────── */}
          {activeTab === "timeline" && (
            <div>
              {(!analysis.timeline_events || analysis.timeline_events.length === 0) ? (
                <div style={{ textAlign: "center", padding: "3rem", color: "var(--muted)", border: "1px dashed var(--border)", borderRadius: "0.5rem" }}>No timeline data.</div>
              ) : (
                <div style={{ position: "relative", paddingLeft: "2rem" }}>
                  {/* Vertical line */}
                  <div style={{ position: "absolute", left: "0.75rem", top: 0, bottom: 0, width: 2, backgroundColor: "var(--border)" }} />
                  {analysis.timeline_events.map((ev, i) => {
                    const color = ev.status === "COMPLETED" ? "#22c55e" : ev.status === "FAILED" ? "#ef4444" : ev.status === "PROCESSING" ? "#3b82f6" : "#d97706";
                    return (
                      <div key={i} style={{ position: "relative", marginBottom: "1.5rem" }}>
                        <div style={{ position: "absolute", left: "-1.6rem", top: "0.25rem", width: 12, height: 12, borderRadius: "50%", backgroundColor: color, border: "2px solid white", boxShadow: "0 0 0 2px var(--border)" }} />
                        <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", padding: "0.75rem 1rem", backgroundColor: "white" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
                            <span style={{ fontWeight: 600, fontSize: "0.85rem" }}>{ev.custom_id || ev.job_id?.slice(0, 8)}</span>
                            <span style={{ fontSize: "0.65rem", fontWeight: 600, padding: "0.0625rem 0.375rem", borderRadius: "9999px", backgroundColor: color + "20", color }}>{ev.status}</span>
                          </div>
                          <div style={{ display: "flex", gap: "0.75rem", fontSize: "0.75rem", color: "var(--muted)" }}>
                            {ev.task_type && <span>{ev.task_type.replace(/_/g, " ")}</span>}
                            {ev.ocr_engine && <span>{ev.ocr_engine}</span>}
                            {ev.total_documents !== undefined && <span>{ev.processed_documents || 0}/{ev.total_documents} docs</span>}
                          </div>
                          <div style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.25rem" }}>
                            {new Date(ev.created_at).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ── Clusters Tab ─────────────────────────────────── */}
          {activeTab === "clusters" && (
            <div>
              {(!analysis.doc_coordinates || analysis.doc_coordinates.length < 2) ? (
                <div style={{ textAlign: "center", padding: "3rem", color: "var(--muted)", border: "1px dashed var(--border)", borderRadius: "0.5rem" }}>Need at least 2 documents for clustering.</div>
              ) : (() => {
                const coords = analysis.doc_coordinates;
                const xs = coords.map(c => c.x);
                const ys = coords.map(c => c.y);
                const xMin = Math.min(...xs), xMax = Math.max(...xs);
                const yMin = Math.min(...ys), yMax = Math.max(...ys);
                const xRange = xMax - xMin || 1;
                const yRange = yMax - yMin || 1;
                const W = 700, H = 450, pad = 40;
                const colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

                return (
                  <div style={{ border: "1px solid var(--border)", borderRadius: "0.5rem", padding: "1rem", backgroundColor: "white" }}>
                    <h3 style={{ fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.75rem" }}>Document Similarity ({coords.length} documents)</h3>
                    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxHeight: "28rem" }}>
                      {/* Grid */}
                      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" />
                      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--border)" />
                      {/* Points */}
                      {coords.map((c, i) => {
                        const cx = pad + ((c.x - xMin) / xRange) * (W - 2 * pad);
                        const cy = H - pad - ((c.y - yMin) / yRange) * (H - 2 * pad);
                        const color = colors[i % colors.length];
                        return (
                          <g key={i}>
                            <circle cx={cx} cy={cy} r={6} fill={color} opacity={0.7} stroke="white" strokeWidth={1.5} />
                            <title>{c.filename}</title>
                            {coords.length <= 20 && (
                              <text x={cx} y={cy - 10} textAnchor="middle" fontSize={9} fill="#666">{c.filename}</text>
                            )}
                          </g>
                        );
                      })}
                    </svg>
                    <p style={{ fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.5rem" }}>
                      Documents closer together have more similar content. Based on TF-IDF text similarity.
                    </p>
                  </div>
                );
              })()}
            </div>
          )}
        </>
      )}
    </div>
  );
}
