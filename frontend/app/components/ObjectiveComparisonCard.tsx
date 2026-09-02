"use client";

import React, { useState } from "react";
import { ThreeWayComparisonEntryPayload } from "../types";

interface ObjectiveComparisonCardProps {
  comparisons: ThreeWayComparisonEntryPayload[];
}

export function ObjectiveComparisonCard({
  comparisons,
}: ObjectiveComparisonCardProps) {
  const [selectedArch, setSelectedArch] = useState<string>(
    comparisons[0]?.architecture || "Vision Transformer (ViT)"
  );

  const entry =
    comparisons.find((c) => c.architecture === selectedArch) || comparisons[0];

  if (!entry) return null;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            3-Way Paradigm Comparison
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Supervised (Phase 13) vs SimCLR (Phase 18) vs Masked Reconstruction (Phase 19).
          </p>
        </div>

        {/* Architecture Switcher */}
        <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 self-start">
          {comparisons.map((c) => (
            <button
              key={c.architecture}
              onClick={() => setSelectedArch(c.architecture)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                selectedArch === c.architecture
                  ? "bg-slate-700 text-white font-semibold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {c.architecture.split(" ")[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Comparison Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        {/* Supervised Card */}
        <div className="bg-slate-950 p-4 rounded-lg border border-emerald-900/40 relative overflow-hidden">
          <div className="absolute top-0 right-0 px-2 py-0.5 bg-emerald-950/80 text-emerald-400 border-l border-b border-emerald-800/60 rounded-bl text-[10px] font-bold">
            SUPERVISED
          </div>
          <span className="text-xs text-slate-400 font-medium">Downstream Probe Acc</span>
          <div className="text-2xl font-black text-emerald-400 font-mono mt-1 mb-3">
            {(entry.supervised_accuracy * 100).toFixed(1)}%
          </div>
          <div className="space-y-1.5 text-[11px] border-t border-slate-800/80 pt-2 text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">Latent Std (&sigma;):</span>
              <span className="font-mono">{entry.supervised_latent_std.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Separation:</span>
              <span className="font-mono">{entry.supervised_separation.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Compactness:</span>
              <span className="font-mono">{entry.supervised_compactness.toFixed(3)}</span>
            </div>
          </div>
        </div>

        {/* SimCLR Contrastive Card */}
        <div className="bg-slate-950 p-4 rounded-lg border border-indigo-900/40 relative overflow-hidden">
          <div className="absolute top-0 right-0 px-2 py-0.5 bg-indigo-950/80 text-indigo-400 border-l border-b border-indigo-800/60 rounded-bl text-[10px] font-bold">
            SIMCLR (CONTRASTIVE)
          </div>
          <span className="text-xs text-slate-400 font-medium">Downstream Probe Acc</span>
          <div className="text-2xl font-black text-indigo-400 font-mono mt-1 mb-3">
            {(entry.simclr_accuracy * 100).toFixed(1)}%
          </div>
          <div className="space-y-1.5 text-[11px] border-t border-slate-800/80 pt-2 text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">Latent Std (&sigma;):</span>
              <span className="font-mono">{entry.simclr_latent_std.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Separation:</span>
              <span className="font-mono">{entry.simclr_separation.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Compactness:</span>
              <span className="font-mono">{entry.simclr_compactness.toFixed(3)}</span>
            </div>
          </div>
        </div>

        {/* Reconstruction MIM Card */}
        <div className="bg-slate-950 p-4 rounded-lg border border-violet-900/40 relative overflow-hidden">
          <div className="absolute top-0 right-0 px-2 py-0.5 bg-violet-950/80 text-violet-400 border-l border-b border-violet-800/60 rounded-bl text-[10px] font-bold">
            RECONSTRUCTION (MIM)
          </div>
          <span className="text-xs text-slate-400 font-medium">Downstream Probe Acc</span>
          <div className="text-2xl font-black text-violet-400 font-mono mt-1 mb-3">
            {(entry.reconstruction_accuracy * 100).toFixed(1)}%
          </div>
          <div className="space-y-1.5 text-[11px] border-t border-slate-800/80 pt-2 text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">Latent Std (&sigma;):</span>
              <span className="font-mono">{entry.reconstruction_latent_std.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Separation:</span>
              <span className="font-mono">{entry.reconstruction_separation.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Compactness:</span>
              <span className="font-mono">{entry.reconstruction_compactness.toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Comparative Takeaway Note */}
      <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 text-xs text-slate-400 flex items-start gap-2">
        <span className="text-violet-400 font-bold">Research Takeaway:</span>
        <span>
          Reconstruction-pretrained features retain spatial edge and high-frequency structures needed for pixel decoding,
          yielding competitive linear probe performance (74.8% on ViT) while maintaining lower semantic cluster separation than label-supervised features.
        </span>
      </div>
    </div>
  );
}
