"use client";

import React from "react";
import { EvidenceGapPayload, MissingExperimentPlanPayload } from "../../benchmarkData";

interface EvidenceGapPanelCardProps {
  gaps: EvidenceGapPayload[];
  missingPlan: MissingExperimentPlanPayload;
}

export const EvidenceGapPanelCard: React.FC<EvidenceGapPanelCardProps> = ({
  gaps,
  missingPlan,
}) => {
  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>🔍</span> Evidence Gaps & Missing Experiment Planner
          </h3>
          <p className="text-xs text-slate-400">
            Explicit tracking of missing factor combinations, single-seed evaluations, and required runs
          </p>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-2.5 py-1 rounded-md bg-amber-950 border border-amber-700 text-amber-300 font-bold">
            {gaps.length} Gaps Detected
          </span>
          <span className="px-2.5 py-1 rounded-md bg-cyan-950 border border-cyan-800 text-cyan-300">
            {missingPlan.estimated_work_units} Work Units
          </span>
        </div>
      </div>

      {/* Gaps List */}
      <div className="space-y-3">
        {gaps.length === 0 ? (
          <div className="p-6 bg-slate-950 rounded-xl border border-emerald-900/40 text-center font-mono text-xs text-emerald-400">
            ✨ Complete Benchmark Coverage! No experimental evidence gaps detected.
          </div>
        ) : (
          gaps.map((gap) => (
            <div
              key={gap.gap_id}
              className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 flex flex-col space-y-2 font-mono text-xs"
            >
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1.5">
                <span className="font-bold text-amber-400">[{gap.gap_id}]</span>
                <span className="text-[10px] text-slate-500">RQ: {gap.research_question_id}</span>
              </div>

              <div className="text-slate-300">{gap.rationale}</div>

              <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400 pt-1">
                <span>
                  Missing Metric:{" "}
                  <code className="text-cyan-400 font-bold">
                    {gap.missing_metric_id || "full_factor_suite"}
                  </code>
                </span>
                <span>
                  Required Seeds: <strong className="text-slate-200">+{gap.missing_seed_count}</strong>
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
