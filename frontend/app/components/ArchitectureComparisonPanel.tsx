"use client";

import React from "react";
import { CrossArchitectureGeometryReport } from "../types";

interface ArchitectureComparisonPanelProps {
  comparison: CrossArchitectureGeometryReport | null;
}

export const ArchitectureComparisonPanel: React.FC<
  ArchitectureComparisonPanelProps
> = ({ comparison }) => {
  if (!comparison) {
    return (
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-8 text-center text-zinc-500 font-mono text-xs">
        No cross-architecture comparison available.
      </div>
    );
  }

  const { architectures, coordinate_space_note } = comparison;
  const archEntries = Object.entries(architectures);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-5 space-y-3">
        <div>
          <h2 className="text-sm font-bold font-mono text-zinc-100 flex items-center gap-2">
            <span>Cross-Architecture Geometry Benchmark</span>
            <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/40">
              Matched Data Budget &bull; Final Representation Layer
            </span>
          </h2>
          <p className="text-xs text-zinc-400">
            Comparing learned manifold separation, neighborhood structure, and inductive bias across model families
          </p>
        </div>

        {/* Comparison Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {archEntries.map(([archKey, summary]) => {
            const isViT = archKey.includes("vit") || summary.architecture.includes("vit");
            const isResNet = archKey.includes("resnet") || summary.architecture.includes("resnet");
            const accentColor = isViT
              ? "border-purple-500/50 bg-purple-950/20"
              : isResNet
              ? "border-emerald-500/50 bg-emerald-950/20"
              : "border-cyan-500/50 bg-cyan-950/20";

            return (
              <div
                key={archKey}
                className={`p-4 rounded-xl border ${accentColor} space-y-3 font-mono`}
              >
                <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
                  <div>
                    <div className="text-xs font-bold text-zinc-100">
                      {summary.architecture.toUpperCase()}
                    </div>
                    <div className="text-[10px] text-zinc-500">
                      Layer: {summary.layer_name}
                    </div>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] bg-zinc-800 text-zinc-300 font-bold">
                    {summary.feature_dim}D
                  </span>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between text-zinc-400">
                    <span>Intra Compactness (d̄)</span>
                    <span className="text-cyan-300 font-bold">
                      {summary.intra_class_compactness.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>Inter Separation (Δ)</span>
                    <span className="text-indigo-300 font-bold">
                      {summary.inter_class_separation.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>Sep / Comp Ratio</span>
                    <span className="text-purple-300 font-bold">
                      {summary.separation_to_compactness_ratio.toFixed(2)}x
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>k-NN Consistency</span>
                    <span className="text-emerald-300 font-bold">
                      {(summary.neighbor_label_consistency * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400">
                    <span>PCA 2D Variance</span>
                    <span className="text-amber-300 font-bold">
                      {(summary.pca_first_two_variance_ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Methodological Note */}
        <div className="p-3 bg-zinc-950/60 border border-zinc-800/80 rounded-lg text-xs font-mono text-zinc-400">
          <span className="text-cyan-400 font-bold">&#9432; Invariant Metrics Note: </span>
          {coordinate_space_note}
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-4 space-y-3">
        <h3 className="text-xs font-mono font-bold text-zinc-200">
          Unified Cross-Architecture Benchmark Matrix
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-zinc-800 text-[10px] text-zinc-400 uppercase">
                <th className="pb-2">Architecture</th>
                <th className="pb-2">Family</th>
                <th className="pb-2">Layer</th>
                <th className="pb-2">Dim (D)</th>
                <th className="pb-2">Mean Norm</th>
                <th className="pb-2">Intra Compactness (d̄)</th>
                <th className="pb-2">Inter Separation (Δ)</th>
                <th className="pb-2">Sep / Comp Ratio</th>
                <th className="pb-2">k-NN Consistency</th>
                <th className="pb-2">PCA 2D Var</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {archEntries.map(([key, s]) => (
                <tr key={key} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="py-2.5 font-bold text-zinc-100 uppercase">{key}</td>
                  <td className="py-2.5 text-zinc-400">{s.model_family}</td>
                  <td className="py-2.5 text-cyan-300">{s.layer_name}</td>
                  <td className="py-2.5 text-zinc-400">{s.feature_dim}</td>
                  <td className="py-2.5 text-zinc-300">{s.mean_vector_norm.toFixed(2)}</td>
                  <td className="py-2.5 text-cyan-200">{s.intra_class_compactness.toFixed(3)}</td>
                  <td className="py-2.5 text-indigo-200">{s.inter_class_separation.toFixed(3)}</td>
                  <td className="py-2.5 text-purple-300 font-bold">
                    {s.separation_to_compactness_ratio.toFixed(2)}x
                  </td>
                  <td className="py-2.5 text-emerald-300">
                    {(s.neighbor_label_consistency * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 text-amber-300">
                    {(s.pca_first_two_variance_ratio * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
