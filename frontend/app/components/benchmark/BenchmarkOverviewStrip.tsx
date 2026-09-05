"use client";

import React from "react";
import { CampaignCoverageSummaryPayload } from "../../benchmarkData";

interface BenchmarkOverviewStripProps {
  coverage: CampaignCoverageSummaryPayload;
  totalObservations: number;
}

export const BenchmarkOverviewStrip: React.FC<BenchmarkOverviewStripProps> = ({
  coverage,
  totalObservations,
}) => {
  const compPct = (coverage.completion_fraction * 100).toFixed(1);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 px-6 py-4 bg-slate-950 border-b border-slate-900">
      {/* Planned Factor Combinations */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-slate-800 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Planned Combos</span>
        <div className="text-xl font-bold font-mono text-white mt-1">
          {coverage.planned_experiments_count}
        </div>
        <span className="text-[10px] text-slate-500">Across 6 dimensions</span>
      </div>

      {/* Completed Factor Combinations */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-emerald-900/40 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400">Completed Runs</span>
        <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
          {coverage.completed_experiments_count}
        </div>
        <span className="text-[10px] text-emerald-500/80">Strictly observed</span>
      </div>

      {/* Partial Factor Combinations */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-amber-900/40 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-amber-400">Partial Runs</span>
        <div className="text-xl font-bold font-mono text-amber-400 mt-1">
          {coverage.partial_experiments_count}
        </div>
        <span className="text-[10px] text-amber-500/80">Missing some metrics</span>
      </div>

      {/* Missing Factor Combinations */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-rose-900/40 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-rose-400">Missing Combos</span>
        <div className="text-xl font-bold font-mono text-rose-400 mt-1">
          {coverage.missing_experiments_count}
        </div>
        <span className="text-[10px] text-rose-500/80">Evidence gap target</span>
      </div>

      {/* Conceptually Not Applicable */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-slate-800 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Excluded (N/A)</span>
        <div className="text-xl font-bold font-mono text-slate-400 mt-1">
          {coverage.not_applicable_count}
        </div>
        <span className="text-[10px] text-slate-500">Domain-incompatible</span>
      </div>

      {/* Completion Percentage */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-cyan-900/50 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400">Completion</span>
        <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
          {compPct}%
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full mt-1.5 overflow-hidden">
          <div
            className="bg-cyan-400 h-full rounded-full transition-all duration-500"
            style={{ width: `${Math.min(100, Math.max(0, Number(compPct)))}%` }}
          />
        </div>
      </div>

      {/* Evaluated Seeds */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-indigo-900/40 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-400">RNG Seeds</span>
        <div className="text-xl font-bold font-mono text-indigo-300 mt-1">
          {coverage.evaluated_seeds_count}
        </div>
        <span className="text-[10px] text-indigo-400/80">42, 100, 2024</span>
      </div>

      {/* Total Registered Observations */}
      <div className="p-3 bg-slate-900/90 rounded-xl border border-purple-900/40 flex flex-col justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-purple-400">Total Cells</span>
        <div className="text-xl font-bold font-mono text-purple-300 mt-1">
          {totalObservations}
        </div>
        <span className="text-[10px] text-purple-400/80">Canonical cells</span>
      </div>
    </div>
  );
};
