"use client";

import React from "react";
import { TemporalObjectiveComparisonPayload } from "../types";

interface TemporalObjectiveComparisonCardProps {
  comparisons: TemporalObjectiveComparisonPayload[];
}

export const TemporalObjectiveComparisonCard: React.FC<
  TemporalObjectiveComparisonCardProps
> = ({ comparisons }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-sm font-bold text-slate-100">
            Pretraining Objective Comparison on Video Tasks
          </h2>
          <p className="text-xs text-slate-400">
            Supervised vs SimCLR vs Masked Reconstruction vs Scratch transfer dynamics
          </p>
        </div>
        <span className="text-xs font-mono text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/30">
          Cross-Objective Study
        </span>
      </div>

      {/* Grid of Objective Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {comparisons.map((item) => {
          const isRecon = item.objective === "reconstruction";
          const isSuper = item.objective === "supervised";
          const isSimclr = item.objective === "simclr";

          const accentColor = isRecon
            ? "border-violet-500/50 bg-violet-950/20 text-violet-300"
            : isSuper
            ? "border-cyan-500/50 bg-cyan-950/20 text-cyan-300"
            : isSimclr
            ? "border-indigo-500/50 bg-indigo-950/20 text-indigo-300"
            : "border-slate-700 bg-slate-950/40 text-slate-400";

          return (
            <div
              key={item.objective}
              className={`rounded-2xl border p-4 flex flex-col justify-between space-y-3 bg-slate-950/80 ${accentColor}`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold font-mono tracking-wide uppercase">
                    {item.label}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                  {item.description}
                </p>
              </div>

              {/* KPI metrics */}
              <div className="space-y-2 pt-2 border-t border-slate-800 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-[10px] uppercase">Frozen Probe Acc:</span>
                  <span className="font-bold text-slate-200">
                    {(item.frozen_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-[10px] uppercase">Fine-Tuned Acc:</span>
                  <span className="font-bold text-emerald-400">
                    {(item.finetune_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-[10px] uppercase">Temporal Consistency:</span>
                  <span className="font-bold text-cyan-300">
                    {item.temporal_consistency.toFixed(3)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-[10px] uppercase">Sequence Drift:</span>
                  <span className="font-bold text-amber-300">
                    {item.sequence_drift.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
