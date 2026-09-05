"use client";

import React, { useState } from "react";
import { ParetoAnalysisPayload } from "../../benchmarkData";

interface ParetoExplorerCardProps {
  pareto: ParetoAnalysisPayload;
  onSelectExperiment?: (expId: string) => void;
}

export const ParetoExplorerCard: React.FC<ParetoExplorerCardProps> = ({
  pareto,
  onSelectExperiment,
}) => {
  const [selectedExp, setSelectedExp] = useState<string | null>(null);

  const nonDominatedSet = new Set(pareto.non_dominated_experiment_ids);

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>📈</span> Multi-Objective Pareto Frontier Analysis
          </h3>
          <p className="text-xs text-slate-400">
            Non-dominated models across: {pareto.metric_ids.join(", ")}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="px-2.5 py-1 rounded-md bg-emerald-950 border border-emerald-700 text-emerald-300 font-bold">
            {pareto.non_dominated_experiment_ids.length} Non-Dominated
          </span>
          <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-400">
            {pareto.candidate_experiment_ids.length} Candidates
          </span>
        </div>
      </div>

      {/* Candidates Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {pareto.candidate_experiment_ids.map((expId) => {
          const isNonDom = nonDominatedSet.has(expId);
          const isSelected = selectedExp === expId;

          return (
            <button
              key={expId}
              onClick={() => {
                setSelectedExp(expId);
                if (onSelectExperiment) onSelectExperiment(expId);
              }}
              className={`p-3 rounded-xl border text-left transition-all font-mono flex flex-col justify-between space-y-2 ${
                isNonDom
                  ? "bg-emerald-950/40 border-emerald-500/80 hover:border-emerald-400 shadow-lg shadow-emerald-950/20"
                  : "bg-slate-950/80 border-slate-800 hover:border-slate-700 text-slate-400"
              } ${isSelected ? "ring-2 ring-cyan-400" : ""}`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-bold ${isNonDom ? "text-emerald-300" : "text-slate-300"}`}>
                  {expId}
                </span>
                {isNonDom && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-900 text-emerald-200 font-bold">
                    PARETO
                  </span>
                )}
              </div>
              <div className="text-[10px] text-slate-500">
                {isNonDom ? "Optimal Tradeoff Point" : "Dominated by frontier"}
              </div>
            </button>
          );
        })}
      </div>

      {selectedExp && pareto.dominated_relationships[selectedExp] && (
        <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 animate-fadeIn">
          Configuration <strong className="text-amber-400">{selectedExp}</strong> is strictly dominated by:{" "}
          <span className="text-emerald-400 font-bold">
            {pareto.dominated_relationships[selectedExp].join(", ")}
          </span>
        </div>
      )}
    </div>
  );
};
