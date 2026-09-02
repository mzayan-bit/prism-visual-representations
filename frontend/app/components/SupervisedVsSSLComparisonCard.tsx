"use client";

import React from "react";
import { SupervisedVsSSLComparisonPayload } from "../types";

interface SupervisedVsSSLComparisonCardProps {
  comparison: SupervisedVsSSLComparisonPayload;
}

export function SupervisedVsSSLComparisonCard({
  comparison,
}: SupervisedVsSSLComparisonCardProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Supervised vs SimCLR vs Scratch Comparison
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Downstream linear probe accuracy under identical target dataset and evaluation protocol.
          </p>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-1 rounded border border-slate-800">
          Architecture: {comparison.architecture}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        {/* Scratch Baseline */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3.5">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-slate-400">Scratch Baseline</span>
            <span className="text-[10px] text-slate-500 font-mono">Fresh Init</span>
          </div>
          <div className="text-2xl font-bold text-slate-200 font-mono mt-1">
            {(comparison.scratch_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-slate-500 mt-2">
            Target trained from random initialization
          </div>
        </div>

        {/* Self-Supervised SimCLR */}
        <div className="bg-slate-950 border border-indigo-900/60 rounded-lg p-3.5 ring-1 ring-indigo-500/20">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-semibold text-indigo-400">SimCLR Pretraining</span>
            <span className="text-[10px] text-indigo-300 font-mono bg-indigo-950/80 px-1.5 py-0.5 rounded border border-indigo-800/40">
              Unlabeled SSL
            </span>
          </div>
          <div className="text-2xl font-bold text-white font-mono mt-1">
            {(comparison.ssl_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-emerald-400 font-medium mt-2 flex items-center gap-1">
            <span>&Delta; vs Scratch:</span>
            <span className="font-mono">
              +{ (comparison.accuracy_gain_ssl_vs_scratch * 100).toFixed(1) }%
            </span>
          </div>
        </div>

        {/* Supervised Pretraining */}
        <div className="bg-slate-950 border border-emerald-900/60 rounded-lg p-3.5">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-semibold text-emerald-400">Supervised Pretraining</span>
            <span className="text-[10px] text-emerald-300 font-mono bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-800/40">
              Class Labels
            </span>
          </div>
          <div className="text-2xl font-bold text-white font-mono mt-1">
            {(comparison.supervised_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-slate-400 mt-2 flex items-center gap-1">
            <span>SSL Gap:</span>
            <span className="font-mono text-amber-400">
              {(comparison.accuracy_gap_ssl_vs_supervised * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Methodological Context Alert */}
      <div className="text-xs text-slate-400 bg-slate-950 border border-slate-800/80 rounded-lg p-3 flex items-start gap-2">
        <span className="text-indigo-400 font-bold">&bull;</span>
        <div>
          <span className="text-slate-200 font-medium">Research Takeaway: </span>
          SimCLR representations capture {((comparison.ssl_accuracy / comparison.supervised_accuracy) * 100).toFixed(1)}% of fully supervised feature linear-discriminative power without utilizing any class labels during pretraining.
        </div>
      </div>
    </div>
  );
}
