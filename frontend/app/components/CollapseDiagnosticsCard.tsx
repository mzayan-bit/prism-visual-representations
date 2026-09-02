"use client";

import React from "react";
import { RepresentationCollapseSummaryPayload } from "../types";

interface CollapseDiagnosticsCardProps {
  collapse: RepresentationCollapseSummaryPayload;
}

export function CollapseDiagnosticsCard({ collapse }: CollapseDiagnosticsCardProps) {
  const healthyFraction = Math.max(0, 1.0 - collapse.near_zero_variance_fraction);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Representation Collapse Diagnostics
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Audit feature dimensional diversity and prevent complete or dimensional collapse.
          </p>
        </div>
        <span
          className={`text-xs font-semibold px-2.5 py-1 rounded-md border ${
            collapse.is_collapsed
              ? "bg-rose-950/80 text-rose-400 border-rose-800/60"
              : "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
          }`}
        >
          {collapse.is_collapsed ? "COLLAPSE DETECTED" : "HEALTHY REPRESENTATION"}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {/* Metric 1 */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
            Mean Feature Std (&sigma;)
          </div>
          <div className="text-lg font-bold text-white font-mono mt-1">
            {collapse.mean_feature_std.toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Min bound: &ge; 0.05
          </div>
        </div>

        {/* Metric 2 */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
            Active Dimensions
          </div>
          <div className="text-lg font-bold text-emerald-400 font-mono mt-1">
            {collapse.total_dimensions - collapse.near_zero_variance_dimensions} / {collapse.total_dimensions}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            {(healthyFraction * 100).toFixed(1)}% active channels
          </div>
        </div>

        {/* Metric 3 */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
            Angular Spread
          </div>
          <div className="text-lg font-bold text-indigo-400 font-mono mt-1">
            {collapse.distinct_sample_cosine_spread.toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Pairwise distinct cosine
          </div>
        </div>

        {/* Metric 4 */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
            Alignment Dist (Sq)
          </div>
          <div className="text-lg font-bold text-purple-400 font-mono mt-1">
            {collapse.mean_positive_alignment_distance.toFixed(4)}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Positive pair divergence
          </div>
        </div>
      </div>

      {/* Progress Bar for Channel Variance */}
      <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5">
          <span>Active Dimensionality Health</span>
          <span className="font-mono text-emerald-400 font-medium">
            {collapse.total_dimensions - collapse.near_zero_variance_dimensions} Active Dimensions
          </span>
        </div>
        <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden flex">
          <div
            className="h-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${healthyFraction * 100}%` }}
          />
          <div
            className="h-full bg-rose-500 transition-all duration-500"
            style={{ width: `${collapse.near_zero_variance_fraction * 100}%` }}
          />
        </div>
        {collapse.warnings.length > 0 && (
          <div className="mt-3 space-y-1">
            {collapse.warnings.map((w, idx) => (
              <div key={idx} className="text-[11px] text-amber-400/90 bg-amber-950/40 border border-amber-800/40 px-2.5 py-1 rounded">
                &bull; {w}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
