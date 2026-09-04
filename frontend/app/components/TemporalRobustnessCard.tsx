"use client";

import React, { useState } from "react";
import { TemporalRobustnessBenchmarkPayload } from "../types";

interface TemporalRobustnessCardProps {
  benchmarks: TemporalRobustnessBenchmarkPayload[];
}

export const TemporalRobustnessCard: React.FC<TemporalRobustnessCardProps> = ({
  benchmarks,
}) => {
  const [selectedCorruption, setSelectedCorruption] = useState<string>(
    benchmarks[0]?.corruption_type || "frame_drop"
  );

  const activeBenchmark =
    benchmarks.find((b) => b.corruption_type === selectedCorruption) ||
    benchmarks[0];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">🛡️</span>
          <div>
            <h2 className="text-sm font-bold text-slate-100">
              Temporal Robustness & Perturbation Stress Testing
            </h2>
            <p className="text-xs text-slate-400">
              Downstream stability under frame drops, stutter, shuffling, and subsampling
            </p>
          </div>
        </div>

        {/* Corruption Selectors */}
        <div className="flex flex-wrap gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          {benchmarks.map((b) => (
            <button
              key={b.corruption_type}
              onClick={() => setSelectedCorruption(b.corruption_type)}
              className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
                selectedCorruption === b.corruption_type
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {b.label.split(" (")[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Selected Benchmark Detail */}
      {activeBenchmark && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-[10px] font-mono text-slate-400 uppercase">Clean Video Acc</div>
            <div className="text-lg font-bold font-mono text-emerald-400 mt-1">
              {(activeBenchmark.clean_accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-[10px] font-mono text-slate-400 uppercase">Perturbed Acc</div>
            <div className="text-lg font-bold font-mono text-amber-400 mt-1">
              {(activeBenchmark.perturbed_accuracy * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-[10px] font-mono text-slate-400 uppercase">Accuracy Delta (ΔAcc)</div>
            <div
              className={`text-lg font-bold font-mono mt-1 ${
                activeBenchmark.accuracy_delta < -0.1
                  ? "text-rose-400"
                  : activeBenchmark.accuracy_delta < 0
                  ? "text-amber-400"
                  : "text-emerald-400"
              }`}
            >
              {(activeBenchmark.accuracy_delta * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
            <div className="text-[10px] font-mono text-slate-400 uppercase">Sequence Drift (Δz)</div>
            <div className="text-lg font-bold font-mono text-cyan-400 mt-1">
              {activeBenchmark.representation_drift.toFixed(3)}
            </div>
          </div>
        </div>
      )}

      {/* Comparative Matrix Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 font-mono text-[10px] text-slate-400 uppercase bg-slate-950/60">
              <th className="py-2.5 px-3">Perturbation Type</th>
              <th className="py-2.5 px-3">Clean Acc</th>
              <th className="py-2.5 px-3">Perturbed Acc</th>
              <th className="py-2.5 px-3">Δ Acc</th>
              <th className="py-2.5 px-3">Sequence Drift</th>
              <th className="py-2.5 px-3">Mechanism</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {benchmarks.map((b) => (
              <tr
                key={b.corruption_type}
                onClick={() => setSelectedCorruption(b.corruption_type)}
                className={`cursor-pointer transition-colors ${
                  selectedCorruption === b.corruption_type
                    ? "bg-amber-500/10 text-amber-200"
                    : "hover:bg-slate-800/40 text-slate-300"
                }`}
              >
                <td className="py-2.5 px-3 font-bold">{b.label}</td>
                <td className="py-2.5 px-3 text-emerald-400">
                  {(b.clean_accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 text-amber-300">
                  {(b.perturbed_accuracy * 100).toFixed(1)}%
                </td>
                <td
                  className={`py-2.5 px-3 font-bold ${
                    b.accuracy_delta < -0.15
                      ? "text-rose-400"
                      : b.accuracy_delta < 0
                      ? "text-amber-400"
                      : "text-emerald-400"
                  }`}
                >
                  {(b.accuracy_delta * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 text-cyan-300">
                  {b.representation_drift.toFixed(3)}
                </td>
                <td className="py-2.5 px-3 text-slate-400 font-sans text-[11px]">
                  {b.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
