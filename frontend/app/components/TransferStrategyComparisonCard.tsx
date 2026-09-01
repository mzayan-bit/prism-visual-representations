"use client";

import React from "react";
import {
  TransferLearningReportPayload,
  TransferStrategyComparisonPayload,
} from "../types";

interface TransferStrategyComparisonCardProps {
  currentReport: TransferLearningReportPayload;
  comparison: TransferStrategyComparisonPayload | null;
}

export function TransferStrategyComparisonCard({
  currentReport,
  comparison,
}: TransferStrategyComparisonCardProps) {
  if (!comparison) {
    return null;
  }

  const strategies = [
    {
      id: "scratch_baseline",
      name: "Scratch Baseline",
      accuracy: comparison.scratch_accuracy,
      gain: 0.0,
      trainablePct: "100%",
      desc: "Random initialization on target data without source pretraining",
      badge: "BASELINE",
      badgeColor: "bg-slate-700/40 text-slate-300 border-slate-600/40",
    },
    {
      id: "linear_probe",
      name: "Linear Probe",
      accuracy: comparison.linear_probe_accuracy,
      gain: comparison.linear_probe_gain,
      trainablePct: `${(currentReport.freeze_plan.trainable_fraction * 100).toFixed(0)}%`,
      desc: "Strictly frozen source backbone; only classifier head updated",
      badge: "FEATURE REUSE",
      badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
    },
    {
      id: "partial_fine_tune",
      name: "Partial Fine-Tune",
      accuracy: comparison.partial_fine_tune_accuracy,
      gain: comparison.partial_fine_tune_gain,
      trainablePct: "50-60%",
      desc: "Frozen early spatial stages; adapted late semantics & classifier",
      badge: "ADAPTIVE",
      badgeColor: "bg-indigo-500/20 text-indigo-300 border-indigo-500/30",
    },
    {
      id: "full_fine_tune",
      name: "Full Fine-Tune",
      accuracy: comparison.full_fine_tune_accuracy,
      gain: comparison.full_fine_tune_gain,
      trainablePct: "100%",
      desc: "End-to-end backpropagation across all backbone parameters",
      badge: "END-TO-END",
      badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    },
  ];

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg mb-6">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
            Transfer Strategy Comparison & Representation Gain Matrix
          </h3>
          <p className="text-xs text-slate-400">
            Controlled evaluation of target performance gains relative to scratch initialization
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {strategies.map((strat) => {
          const isCurrent = currentReport.strategy === strat.id;
          const gainFormatted =
            strat.gain >= 0
              ? `+${(strat.gain * 100).toFixed(1)}%`
              : `${(strat.gain * 100).toFixed(1)}%`;

          return (
            <div
              key={strat.id}
              className={`p-4 rounded-xl border transition-all ${
                isCurrent
                  ? "bg-slate-800/90 border-cyan-500/80 shadow-md ring-1 ring-cyan-500/30"
                  : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium border ${strat.badgeColor}`}
                >
                  {strat.badge}
                </span>
                {isCurrent && (
                  <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider font-mono">
                    ACTIVE
                  </span>
                )}
              </div>

              <div className="text-sm font-bold text-white mb-1">
                {strat.name}
              </div>

              <div className="flex items-baseline gap-2 mb-2">
                <span className="text-2xl font-bold font-mono text-white">
                  {(strat.accuracy * 100).toFixed(1)}%
                </span>
                {strat.id !== "scratch_baseline" && (
                  <span
                    className={`text-xs font-mono font-bold ${
                      strat.gain >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {gainFormatted} Δ
                  </span>
                )}
              </div>

              <div className="text-[11px] text-slate-400 mb-3 min-h-[32px] line-clamp-2">
                {strat.desc}
              </div>

              <div className="border-t border-slate-800/80 pt-2 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>Trainable Ratio</span>
                <span className="text-slate-200 font-semibold">
                  {strat.trainablePct}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
