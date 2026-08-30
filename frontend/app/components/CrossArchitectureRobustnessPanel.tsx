"use client";

import React from "react";
import { CrossArchitectureRobustnessReport } from "../types";

interface CrossArchitectureRobustnessPanelProps {
  comparison: CrossArchitectureRobustnessReport | null;
}

export default function CrossArchitectureRobustnessPanel({
  comparison,
}: CrossArchitectureRobustnessPanelProps) {
  if (!comparison) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm">
        No cross-architecture robustness comparison data available.
      </div>
    );
  }

  const archs = Object.values(comparison.architectures);

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>⚖️</span> Cross-Architecture Robustness Benchmark
          </h2>
          <p className="text-xs text-slate-400">
            Comparative resilience under identical corruptions across CNN, ResNet, and Vision Transformer
          </p>
        </div>
      </div>

      {/* Architecture Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {archs.map((arch) => {
          const cleanAccPct = (arch.clean_accuracy * 100).toFixed(1);
          const meanCorrPct = (arch.mean_corrupted_accuracy * 100).toFixed(1);
          const dropPct = (arch.mean_accuracy_drop * 100).toFixed(1);

          return (
            <div
              key={arch.architecture}
              className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                    {arch.architecture}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    {arch.model_family}
                  </span>
                </div>
                <div className="text-2xl font-bold text-white mb-1">
                  {meanCorrPct}%
                </div>
                <div className="text-xs text-slate-400 mb-3 flex items-center gap-2">
                  <span>Clean: {cleanAccPct}%</span>
                  <span className="text-rose-400 font-semibold">
                    (↓ {dropPct}%)
                  </span>
                </div>
              </div>

              <div className="space-y-2 pt-3 border-t border-slate-800/80 text-xs text-slate-300">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Mean Representation Drift:</span>
                  <span className="font-mono text-cyan-300 font-bold">
                    {arch.mean_representation_drift.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Neighbor Overlap:</span>
                  <span className="font-mono text-emerald-300 font-bold">
                    {(arch.mean_neighbor_overlap * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Centroid Displacement:</span>
                  <span className="font-mono text-indigo-300 font-bold">
                    {arch.mean_centroid_displacement.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparative Table */}
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-950/40">
              <th className="py-2.5 px-3">Architecture</th>
              <th className="py-2.5 px-3">Family</th>
              <th className="py-2.5 px-3">Clean Acc</th>
              <th className="py-2.5 px-3">Mean Corrupted Acc</th>
              <th className="py-2.5 px-3">Mean Drop</th>
              <th className="py-2.5 px-3">Mean Drift (L2)</th>
              <th className="py-2.5 px-3">Neighbor Retention</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-200 font-medium">
            {archs.map((arch) => (
              <tr key={arch.architecture} className="hover:bg-slate-800/40">
                <td className="py-2.5 px-3 font-bold text-white uppercase">
                  {arch.architecture}
                </td>
                <td className="py-2.5 px-3 text-slate-400">{arch.model_family}</td>
                <td className="py-2.5 px-3 font-mono text-emerald-400">
                  {(arch.clean_accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono font-bold text-white">
                  {(arch.mean_corrupted_accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono text-rose-400">
                  -{(arch.mean_accuracy_drop * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono text-cyan-300">
                  {arch.mean_representation_drift.toFixed(3)}
                </td>
                <td className="py-2.5 px-3 font-mono text-emerald-300">
                  {(arch.mean_neighbor_overlap * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Scientific Methodology Note */}
      <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs text-slate-400 flex items-start gap-2">
        <span className="text-cyan-400 font-bold">ℹ️ Note:</span>
        <span>{comparison.coordinate_space_note}</span>
      </div>
    </div>
  );
}
