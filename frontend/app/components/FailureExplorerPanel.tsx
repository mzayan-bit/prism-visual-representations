"use client";

import React, { useState } from "react";
import { CandidateFailureCase } from "../types";

interface FailureExplorerPanelProps {
  failures: CandidateFailureCase[];
  selectedSampleId: string | null;
  onSelectSample: (sampleId: string) => void;
}

export const FailureExplorerPanel: React.FC<FailureExplorerPanelProps> = ({
  failures,
  selectedSampleId,
  onSelectSample,
}) => {
  const [filterKind, setFilterKind] = useState<string>("all");

  const kinds = Array.from(new Set(failures.map((f) => f.failure_kind)));
  const filteredFailures =
    filterKind === "all"
      ? failures
      : failures.filter((f) => f.failure_kind === filterKind);

  return (
    <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-zinc-800/80">
        <div>
          <h3 className="text-xs font-mono font-bold text-rose-300 flex items-center gap-1.5">
            <span>&#9888; Candidate Failure Explorer</span>
            <span className="bg-rose-950/80 text-rose-400 border border-rose-800 px-1.5 py-0.2 rounded text-[10px]">
              {failures.length} found
            </span>
          </h3>
          <p className="text-[10px] text-zinc-400">
            Ambiguous manifold points, cross-class neighbors, & centroid outliers
          </p>
        </div>

        {/* Filter Dropdown */}
        {kinds.length > 0 && (
          <select
            value={filterKind}
            onChange={(e) => setFilterKind(e.target.value)}
            className="bg-zinc-950 border border-zinc-800 text-zinc-300 font-mono text-[11px] rounded px-2 py-1 focus:outline-none cursor-pointer"
          >
            <option value="all">All Types ({failures.length})</option>
            {kinds.map((k) => (
              <option key={k} value={k}>
                {k} ({failures.filter((f) => f.failure_kind === k).length})
              </option>
            ))}
          </select>
        )}
      </div>

      {/* List */}
      {failures.length === 0 ? (
        <div className="text-center py-8 text-zinc-500 font-mono text-xs">
          <div className="text-emerald-400 text-base mb-1">&#10003;</div>
          <div className="text-zinc-300 font-bold">Zero Ambiguities Detected</div>
          <p className="text-[11px] text-zinc-500 mt-1">
            All samples exhibit 100% same-class nearest neighbors and clear centroid separation.
          </p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[260px] overflow-y-auto pr-1">
          {filteredFailures.map((failure, idx) => {
            const isSelected = selectedSampleId === failure.sample_id;
            return (
              <div
                key={`${failure.sample_id}-${failure.failure_kind}-${idx}`}
                onClick={() => onSelectSample(failure.sample_id)}
                className={`p-2.5 rounded-lg border text-xs font-mono cursor-pointer transition-colors ${
                  isSelected
                    ? "bg-rose-950/40 border-rose-500/80 shadow-sm"
                    : "bg-zinc-950/60 border-zinc-800/80 hover:border-zinc-700"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-zinc-200">{failure.sample_id}</span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-zinc-800 text-zinc-300">
                      Class {failure.label}
                    </span>
                  </div>
                  <span className="text-[10px] font-bold text-rose-400 bg-rose-950/60 px-1.5 py-0.5 rounded border border-rose-900/60">
                    {failure.failure_kind}
                  </span>
                </div>
                <p className="text-[11px] text-zinc-400 line-clamp-2">
                  {failure.description}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
