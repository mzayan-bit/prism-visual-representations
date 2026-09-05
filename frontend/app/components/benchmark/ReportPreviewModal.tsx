"use client";

import React, { useState } from "react";
import { BenchmarkDatasetPayload } from "../../benchmarkData";

interface ReportPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataset: BenchmarkDatasetPayload;
  format: "markdown" | "json" | "csv";
}

export const ReportPreviewModal: React.FC<ReportPreviewModalProps> = ({
  isOpen,
  onClose,
  dataset,
  format,
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  // Generate preview content based on format
  let content = "";
  if (format === "json") {
    content = JSON.stringify(dataset, null, 2);
  } else if (format === "csv") {
    const table = dataset.benchmark_tables?.[0];
    if (table && table.rows) {
      const colKeys = Object.keys(table.rows[0] || {}).filter(
        (k) => k !== table.row_factor
      );
      const header = [table.row_factor, ...colKeys].join(",");
      const rows = table.rows.map((r) => {
        const rVal = String(r[table.row_factor] ?? "");
        const vals = colKeys.map((c) => {
          const val = r[c];
          if (typeof val === "object" && val !== null) {
            const vObj = val as Record<string, unknown>;
            return String(vObj.value ?? vObj.display ?? "");
          }
          return String(val ?? "");
        });
        return [rVal, ...vals].join(",");
      });
      content = [header, ...rows].join("\n");
    } else {
      content = "factor,val1,val2\n";
    }
  } else {
    // Markdown
    const lines = [];
    lines.push(`# PRISM Benchmark Synthesis: ${dataset.campaign.title}`);
    lines.push("");
    lines.push(`**Campaign ID:** \`${dataset.campaign.campaign_id}\``);
    lines.push(`**Fingerprint:** \`${dataset.campaign.fingerprint}\``);
    lines.push(`**Completion:** **${(dataset.coverage_summary.completion_fraction * 100).toFixed(1)}%**`);
    lines.push("");
    lines.push("## Executive Summary");
    lines.push("");
    lines.push(`> ${dataset.report_summary.executive_summary}`);
    lines.push("");
    lines.push("## Experimental Methodology");
    lines.push("");
    lines.push(dataset.report_summary.methodology_summary);
    lines.push("");
    lines.push("## Grounded Scientific Findings");
    lines.push("");
    dataset.findings.forEach((f) => {
      lines.push(`### [${f.finding_id}] ${f.evidence_strength.toUpperCase()}`);
      lines.push(`> ${f.statement}`);
      if (f.caveats?.length) {
        lines.push(`*Caveats:* ${f.caveats.join("; ")}`);
      }
      lines.push("");
    });
    content = lines.join("\n");
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([content], {
      type:
        format === "json"
          ? "application/json"
          : format === "csv"
          ? "text/csv"
          : "text/markdown",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `prism_benchmark_report.${format === "markdown" ? "md" : format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fadeIn">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">📜</span>
            <h3 className="text-sm font-bold text-white font-mono">
              Report Preview ({format.toUpperCase()})
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-mono transition-all"
            >
              {copied ? "✓ Copied!" : "📋 Copy"}
            </button>
            <button
              onClick={handleDownload}
              className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-mono font-bold transition-all"
            >
              ⬇ Download
            </button>
            <button
              onClick={onClose}
              className="p-1 text-slate-400 hover:text-white rounded-lg text-xs font-mono ml-2"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="p-4 overflow-y-auto flex-1 font-mono text-xs text-slate-300 bg-slate-950">
          <pre className="whitespace-pre-wrap leading-relaxed">{content}</pre>
        </div>
      </div>
    </div>
  );
};
