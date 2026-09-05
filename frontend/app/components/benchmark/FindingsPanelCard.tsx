"use client";

import React from "react";
import { ResearchFindingPayload } from "../../benchmarkData";

interface FindingsPanelCardProps {
  findings: ResearchFindingPayload[];
}

export const FindingsPanelCard: React.FC<FindingsPanelCardProps> = ({
  findings,
}) => {
  const getEvidenceBadge = (strength: string) => {
    switch (strength) {
      case "supported_by_repeated_runs":
        return {
          label: "SUPPORTED BY REPEATED RUNS (N≥3)",
          color: "bg-emerald-950 border-emerald-500 text-emerald-300",
        };
      case "supported_by_single_run":
        return {
          label: "SUPPORTED BY SINGLE RUN (N=1)",
          color: "bg-amber-950 border-amber-500 text-amber-300",
        };
      case "descriptive_only":
        return {
          label: "DESCRIPTIVE ONLY",
          color: "bg-purple-950 border-purple-500 text-purple-300",
        };
      default:
        return {
          label: "INSUFFICIENT EVIDENCE",
          color: "bg-rose-950 border-rose-500 text-rose-300",
        };
    }
  };

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div>
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>💡</span> Evidence-Backed Scientific Findings
        </h3>
        <p className="text-xs text-slate-400">
          Template-generated claims strictly grounded in observed benchmark cells with explicit caveats
        </p>
      </div>

      <div className="space-y-3">
        {findings.map((f) => {
          const badge = getEvidenceBadge(f.evidence_strength);

          return (
            <div
              key={f.finding_id}
              className="p-4 bg-slate-950 rounded-xl border border-slate-800/80 hover:border-slate-700 transition-all flex flex-col space-y-2.5"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/60 pb-2">
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-xs font-bold text-cyan-400">[{f.finding_id}]</span>
                  <span className="text-xs text-slate-400">RQ: {f.research_question_id}</span>
                </div>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${badge.color}`}
                >
                  {badge.label}
                </span>
              </div>

              {/* Statement */}
              <div className="text-xs text-slate-200 leading-relaxed font-medium">
                {f.statement}
              </div>

              {/* Effect size and control info */}
              <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono text-slate-400 pt-1">
                {f.effect_size_delta !== null && f.effect_size_delta !== undefined && (
                  <span>
                    Measured Delta:{" "}
                    <strong className="text-cyan-400">{f.effect_size_delta.toFixed(3)}</strong>
                  </span>
                )}
                {f.comparison_audit && (
                  <span>
                    Control:{" "}
                    <strong
                      className={
                        f.comparison_audit.is_strictly_controlled
                          ? "text-emerald-400"
                          : "text-amber-400"
                      }
                    >
                      {f.comparison_audit.status.toUpperCase()}
                    </strong>
                  </span>
                )}
                <span>
                  Support:{" "}
                  <strong className="text-slate-300">
                    {f.supporting_result_ids?.length || 0} cells
                  </strong>
                </span>
              </div>

              {/* Caveats */}
              {f.caveats && f.caveats.length > 0 && (
                <div className="p-2 bg-slate-900/60 rounded-lg border border-slate-800/60 text-[10px] font-mono text-slate-400 space-y-0.5">
                  <span className="text-slate-300 font-bold block">Caveats & Limitations:</span>
                  {f.caveats.map((c, i) => (
                    <div key={i}>• {c}</div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
