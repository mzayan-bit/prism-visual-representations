"use client";

import React, { useState } from "react";
import { TemporalCandidateFailurePayload } from "../types";

interface TemporalFailureExplorerProps {
  failures: TemporalCandidateFailurePayload[];
  onSelectSample: (sampleId: string) => void;
}

export const TemporalFailureExplorer: React.FC<TemporalFailureExplorerProps> = ({
  failures,
  onSelectSample,
}) => {
  const [filterType, setFilterType] = useState<string>("all");

  const filteredFailures =
    filterType === "all"
      ? failures
      : failures.filter((f) => f.failure_type === filterType);

  const failureTypes = Array.from(new Set(failures.map((f) => f.failure_type)));

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-sm font-bold text-slate-100">
            Temporal Failure Taxonomy Explorer
          </h2>
          <p className="text-xs text-slate-400">
            Investigate edge-cases, order anomalies, and sensitivity failure patterns
          </p>
        </div>

        {/* Filter */}
        <div className="flex flex-wrap gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setFilterType("all")}
            className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
              filterType === "all"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            All ({failures.length})
          </button>
          {failureTypes.map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
                filterType === t
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {t.replace(/_/g, " ").toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Failure Cards List */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {filteredFailures.map((f, idx) => (
          <div
            key={idx}
            onClick={() => onSelectSample(f.sample_id)}
            className="cursor-pointer bg-slate-950 border border-slate-800 hover:border-amber-500/50 p-3 rounded-xl transition-all space-y-2 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-cyan-300 group-hover:text-amber-300 transition-colors">
                {f.sample_id}
              </span>
              <span
                className={`text-[9px] font-mono font-semibold px-2 py-0.5 rounded border uppercase ${
                  f.severity === "high"
                    ? "bg-rose-950 text-rose-300 border-rose-500/30"
                    : f.severity === "medium"
                    ? "bg-amber-950 text-amber-300 border-amber-500/30"
                    : "bg-slate-800 text-slate-300 border-slate-700"
                }`}
              >
                {f.severity} severity
              </span>
            </div>

            <div className="text-[11px] font-mono text-slate-400">
              Type: <span className="text-slate-200">{f.failure_type.replace(/_/g, " ")}</span>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              {f.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
