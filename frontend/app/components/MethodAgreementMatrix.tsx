"use client";

import React, { useState } from "react";
import { AttributionComparisonReport } from "../types";

interface MethodAgreementMatrixProps {
  report?: AttributionComparisonReport;
  selectedArch: string;
}

export const MethodAgreementMatrix: React.FC<MethodAgreementMatrixProps> = ({
  report,
  selectedArch,
}) => {
  const [metricMode, setMetricMode] = useState<"cosine" | "top_10">("cosine");

  if (!report) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-500 font-mono text-xs">
        No comparison report available for this configuration.
      </div>
    );
  }

  const methods = Object.keys(report.results);
  const matrix = metricMode === "cosine" ? report.cosine_similarity_matrix : report.top_10_overlap_matrix;

  // Format method labels for display
  const formatMethodLabel = (m: string) => {
    switch (m) {
      case "input_gradient":
        return "Input Grad";
      case "gradient_x_input":
        return "Grad × Input";
      case "occlusion_sensitivity":
        return "Occlusion";
      case "grad_cam":
        return "Grad-CAM";
      case "vit_attention":
        return "ViT Attention";
      default:
        return m;
    }
  };

  const getCellColor = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "bg-slate-950/60 text-slate-600";
    if (val >= 0.85) return "bg-emerald-950/80 text-emerald-300 font-bold border border-emerald-700/60";
    if (val >= 0.60) return "bg-cyan-950/80 text-cyan-300 font-semibold border border-cyan-800/60";
    if (val >= 0.35) return "bg-slate-800/80 text-slate-300 border border-slate-700/60";
    if (val >= 0.15) return "bg-amber-950/50 text-amber-300/80 border border-amber-900/40";
    return "bg-rose-950/40 text-rose-300/70 border border-rose-900/40";
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl mb-8">
      {/* Header with Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-slate-800/80">
        <div>
          <h3 className="text-sm font-black text-white font-mono flex items-center gap-2">
            <span>📐</span> CROSS-METHOD AGREEMENT MATRIX
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Quantifying spatial alignment and divergence between attribution techniques on {selectedArch.toUpperCase()}.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono">
          <button
            onClick={() => setMetricMode("cosine")}
            className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
              metricMode === "cosine"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Cosine Similarity
          </button>
          <button
            onClick={() => setMetricMode("top_10")}
            className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
              metricMode === "top_10"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Top 10% Jaccard Overlap
          </button>
        </div>
      </div>

      {/* Grid Matrix Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr>
              <th className="p-2.5 text-left text-slate-500 font-bold border-b border-slate-800">Method</th>
              {methods.map((m) => (
                <th key={m} className="p-2.5 text-center text-slate-300 font-bold border-b border-slate-800">
                  {formatMethodLabel(m)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {methods.map((rowMethod) => (
              <tr key={rowMethod} className="border-b border-slate-800/40 hover:bg-slate-800/20">
                <td className="p-2.5 font-bold text-slate-300 bg-slate-950/40 rounded-l">
                  {formatMethodLabel(rowMethod)}
                </td>
                {methods.map((colMethod) => {
                  const val = matrix[rowMethod]?.[colMethod];
                  const isDiag = rowMethod === colMethod;

                  return (
                    <td key={colMethod} className="p-2 text-center">
                      <div
                        className={`py-2 px-2.5 rounded-lg text-center transition-all ${
                          isDiag
                            ? "bg-slate-950 text-slate-500 font-normal border border-slate-800"
                            : getCellColor(val)
                        }`}
                      >
                        {val !== null && val !== undefined ? val.toFixed(3) : "N/A"}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary KPI Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-6 pt-4 border-t border-slate-800/80 text-xs font-mono">
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between">
          <span className="text-slate-400">Mean Pairwise Agreement:</span>
          <span className="text-cyan-400 font-bold text-sm">
            {report.mean_cross_method_agreement.toFixed(3)}
          </span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between">
          <span className="text-slate-400">Evaluated Method Pairs:</span>
          <span className="text-emerald-400 font-bold text-sm">
            {report.pairwise_agreements.length} pairs
          </span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between">
          <span className="text-slate-400">Disagreement Status:</span>
          <span className={`font-bold text-xs ${report.mean_cross_method_agreement < 0.30 ? "text-amber-400" : "text-emerald-400"}`}>
            {report.mean_cross_method_agreement < 0.30 ? "Moderate Divergence" : "Consistent Alignment"}
          </span>
        </div>
      </div>
    </div>
  );
};
