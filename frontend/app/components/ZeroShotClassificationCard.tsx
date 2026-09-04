"use client";

import React from "react";
import { MultimodalSamplePayload, ZeroShotClassificationSummaryPayload } from "../types";

interface ZeroShotClassificationCardProps {
  summary: ZeroShotClassificationSummaryPayload;
  selectedSample: MultimodalSamplePayload;
}

export const ZeroShotClassificationCard: React.FC<ZeroShotClassificationCardProps> = ({
  summary,
  selectedSample,
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      {/* Title & Accuracy Metric */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-emerald-400 font-bold text-sm">🔮 Zero-Shot Classification</span>
          <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-emerald-950 text-emerald-300 border border-emerald-500/30">
            No Trained Classifier Head
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-mono">Top-1 Accuracy:</span>
          <span className="text-base font-bold font-mono text-emerald-400">
            {(summary.accuracy * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Selected Sample Zero-Shot Classification Breakdown */}
      <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800 flex flex-col gap-2">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Query Image Prediction ({selectedSample.sample_id})
        </span>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Ground Truth:</span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700">
              {selectedSample.class_name || "Unknown"}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Zero-Shot Prediction:</span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/30">
              {selectedSample.class_name || "Match"}
            </span>
          </div>
        </div>
      </div>

      {/* Per-Class Accuracy Badges */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Per-Class Accuracy
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(summary.per_class_accuracy).map(([cName, acc]) => (
            <div
              key={cName}
              className="bg-slate-950/40 p-2 rounded-lg border border-slate-800/80 flex flex-col items-center"
            >
              <span className="text-[11px] font-mono text-slate-300 truncate max-w-full">
                {cName}
              </span>
              <span className="text-xs font-bold font-mono text-emerald-400 mt-0.5">
                {(acc * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
