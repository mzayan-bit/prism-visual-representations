"use client";

import React from "react";
import { AttentionDriftSummary } from "../types";

interface ViTAttentionDriftPanelProps {
  attentionDrift: AttentionDriftSummary | null;
  isViT: boolean;
}

export default function ViTAttentionDriftPanel({
  attentionDrift,
  isViT,
}: ViTAttentionDriftPanelProps) {
  if (!isViT) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm">
        <div className="flex items-center gap-2 text-amber-400 font-bold mb-1">
          <span>ℹ️</span> Attention Drift Analysis Inapplicable
        </div>
        <p className="text-xs text-slate-400">
          Multi-head self-attention entropy drift is specific to Vision Transformers (ViT).
          Select the <span className="text-white font-semibold">ViT</span> model architecture to view attention pattern shifts.
        </p>
      </div>
    );
  }

  if (!attentionDrift) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm">
        No attention drift data captured for current configuration.
      </div>
    );
  }

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>👁️</span> Vision Transformer Self-Attention Drift
          </h2>
          <p className="text-xs text-slate-400">
            Multi-head attention entropy dispersion and diagonal mass collapse under corruption
          </p>
        </div>
      </div>

      {/* Overview Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="text-xs text-slate-400">Clean Mean Entropy</div>
          <div className="text-lg font-bold text-white mt-1">
            {attentionDrift.clean_overall_mean_entropy.toFixed(3)}{" "}
            <span className="text-xs font-normal text-slate-400">nats</span>
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="text-xs text-slate-400">Corrupted Mean Entropy</div>
          <div className="text-lg font-bold text-cyan-300 mt-1">
            {attentionDrift.corrupted_overall_mean_entropy.toFixed(3)}{" "}
            <span className="text-xs font-normal text-slate-400">nats</span>
          </div>
          <div className="text-[11px] font-semibold text-cyan-400 mt-0.5">
            Δ {attentionDrift.overall_entropy_delta > 0 ? "+" : ""}
            {attentionDrift.overall_entropy_delta.toFixed(3)} (
            {attentionDrift.overall_entropy_delta > 0 ? "diffuse" : "concentrated"})
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="text-xs text-slate-400">Clean Diagonal Mass</div>
          <div className="text-lg font-bold text-white mt-1">
            {(attentionDrift.clean_overall_diagonal_mass * 100).toFixed(1)}%
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
          <div className="text-xs text-slate-400">Corrupted Diagonal Mass</div>
          <div className="text-lg font-bold text-indigo-300 mt-1">
            {(attentionDrift.corrupted_overall_diagonal_mass * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] font-semibold text-indigo-400 mt-0.5">
            Δ {(attentionDrift.overall_diagonal_mass_delta * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Layer-by-Layer Attention Drift Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-950/40">
              <th className="py-2.5 px-3">Transformer Layer</th>
              <th className="py-2.5 px-3">Clean Entropy</th>
              <th className="py-2.5 px-3">Corrupted Entropy</th>
              <th className="py-2.5 px-3">Entropy Shift (ΔH)</th>
              <th className="py-2.5 px-3">Clean Diag Mass</th>
              <th className="py-2.5 px-3">Corrupted Diag Mass</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-200 font-medium">
            {attentionDrift.layer_drifts.map((layer) => (
              <tr key={layer.layer_name} className="hover:bg-slate-800/40">
                <td className="py-2.5 px-3 font-mono font-bold text-white">
                  {layer.layer_name}
                </td>
                <td className="py-2.5 px-3 font-mono">
                  {layer.clean_entropy.toFixed(3)}
                </td>
                <td className="py-2.5 px-3 font-mono text-cyan-300 font-bold">
                  {layer.corrupted_entropy.toFixed(3)}
                </td>
                <td className="py-2.5 px-3 font-mono">
                  <span
                    className={`px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                      layer.entropy_delta > 0
                        ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                        : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    }`}
                  >
                    {layer.entropy_delta > 0 ? "+" : ""}
                    {layer.entropy_delta.toFixed(3)} nats
                  </span>
                </td>
                <td className="py-2.5 px-3 font-mono">
                  {(layer.clean_diagonal_mass * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono text-indigo-300">
                  {(layer.corrupted_diagonal_mass * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
