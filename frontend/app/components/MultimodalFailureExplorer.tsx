"use client";

import React, { useState } from "react";
import { MultimodalCandidateFailurePayload } from "../types";

interface MultimodalFailureExplorerProps {
  failures: MultimodalCandidateFailurePayload[];
  onSelectSampleId: (id: string) => void;
}

export const MultimodalFailureExplorer: React.FC<MultimodalFailureExplorerProps> = ({
  failures,
  onSelectSampleId,
}) => {
  const [selectedFilter, setSelectedFilter] = useState<string>("all");

  const failureTypes = Array.from(new Set(failures.map((f) => f.failure_type)));
  const filtered =
    selectedFilter === "all"
      ? failures
      : failures.filter((f) => f.failure_type === selectedFilter);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-red-400 font-bold text-sm">🚨 Multimodal Diagnostic Failure Explorer</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400 font-medium">Filter:</span>
          <select
            value={selectedFilter}
            onChange={(e) => setSelectedFilter(e.target.value)}
            className="bg-slate-950 text-slate-200 rounded px-2 py-1 border border-slate-700 text-xs font-mono focus:outline-none focus:border-red-500"
          >
            <option value="all">All Failure Types ({failures.length})</option>
            {failureTypes.map((ft) => (
              <option key={ft} value={ft}>
                {ft.toUpperCase().replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex flex-col gap-2 max-h-56 overflow-y-auto">
        {filtered.map((fail, idx) => {
          const isHigh = fail.severity === "high";

          return (
            <div
              key={idx}
              onClick={() => onSelectSampleId(fail.sample_id)}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                isHigh
                  ? "bg-red-950/30 border-red-500/40 hover:bg-red-950/50"
                  : "bg-slate-950/40 border-slate-800 hover:bg-slate-800/40"
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-slate-200">
                    {fail.sample_id}
                  </span>
                  <span
                    className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded font-semibold ${
                      isHigh
                        ? "bg-red-500/20 text-red-300 border border-red-500/30"
                        : "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    }`}
                  >
                    {fail.failure_type.replace(/_/g, " ")}
                  </span>
                </div>
                <span className="text-[10px] text-slate-500 font-mono">
                  Severity: {fail.severity.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-slate-300">{fail.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
