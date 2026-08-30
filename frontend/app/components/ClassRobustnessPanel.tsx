"use client";

import React from "react";
import { ClassCentroidDriftSummary } from "../types";

interface ClassRobustnessPanelProps {
  classDrifts: Record<string, ClassCentroidDriftSummary>;
  classNames?: string[];
}

export default function ClassRobustnessPanel({
  classDrifts,
  classNames = ["Class 0", "Class 1", "Class 2"],
}: ClassRobustnessPanelProps) {
  const driftList = Object.values(classDrifts);

  if (driftList.length === 0) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm">
        No class centroid drift data available.
      </div>
    );
  }

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>🏷️</span> Class Manifold & Centroid Displacement
          </h2>
          <p className="text-xs text-slate-400">
            Per-class centroid drift (Δμ), compactness degradation, and margin collapse
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-950/40">
              <th className="py-2.5 px-3">Class</th>
              <th className="py-2.5 px-3">Centroid Displacement (Δμ)</th>
              <th className="py-2.5 px-3">Cosine Similarity</th>
              <th className="py-2.5 px-3">Clean Compactness</th>
              <th className="py-2.5 px-3">Corrupted Compactness</th>
              <th className="py-2.5 px-3">Separation Δ</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-200 font-medium">
            {driftList.map((drift) => {
              const labelIdx = parseInt(drift.class_label, 10) || 0;
              const name = classNames[labelIdx] || `Class ${drift.class_label}`;

              return (
                <tr
                  key={drift.class_label}
                  className="hover:bg-slate-800/40 transition-colors"
                >
                  <td className="py-2.5 px-3 font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    {name}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-cyan-300 font-bold">
                    {drift.centroid_displacement.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-indigo-300">
                    {drift.cosine_similarity.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    {drift.clean_intra_compactness.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    {drift.corrupted_intra_compactness.toFixed(4)}{" "}
                    <span
                      className={`text-[10px] ml-1 ${
                        drift.compactness_delta > 0
                          ? "text-rose-400"
                          : "text-emerald-400"
                      }`}
                    >
                      ({drift.compactness_delta > 0 ? "+" : ""}
                      {drift.compactness_delta.toFixed(3)})
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                        drift.competing_separation_delta < 0
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      }`}
                    >
                      {drift.competing_separation_delta > 0 ? "+" : ""}
                      {drift.competing_separation_delta.toFixed(3)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
