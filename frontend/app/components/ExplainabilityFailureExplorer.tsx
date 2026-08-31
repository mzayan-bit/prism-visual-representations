"use client";

import React, { useState } from "react";
import {
  ExplainabilitySamplePayload,
  ExplanationFailureFlag,
} from "../types";

interface ExplainabilityFailureExplorerProps {
  samples: ExplainabilitySamplePayload[];
  selectedArch: string;
  onSelectSample: (sampleId: string) => void;
}

export const ExplainabilityFailureExplorer: React.FC<ExplainabilityFailureExplorerProps> = ({
  samples,
  selectedArch,
  onSelectSample,
}) => {
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  // Collect all flagged failure records
  const allFlags: Array<{
    sampleId: string;
    sampleName: string;
    trueClass: string;
    flag: ExplanationFailureFlag;
  }> = [];

  samples.forEach((sample) => {
    const archFlags = sample.failure_flags[selectedArch] || [];
    archFlags.forEach((flag) => {
      allFlags.push({
        sampleId: sample.sample_id,
        sampleName: sample.sample_id.replace("sample_", "Sample #"),
        trueClass: sample.class_name,
        flag,
      });
    });
  });

  const filteredFlags =
    categoryFilter === "all"
      ? allFlags
      : allFlags.filter((f) => f.flag.category === categoryFilter);

  const getSeverityBadge = (sev: string) => {
    switch (sev) {
      case "critical":
        return "bg-rose-950/80 text-rose-300 border-rose-700/60 font-bold";
      case "high":
        return "bg-amber-950/80 text-amber-300 border-amber-700/60 font-bold";
      case "medium":
        return "bg-cyan-950/80 text-cyan-300 border-cyan-700/60 font-semibold";
      default:
        return "bg-slate-800 text-slate-400 border-slate-700";
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl mb-8">
      {/* Header & Filter Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800/80">
        <div>
          <h3 className="text-sm font-black text-white font-mono flex items-center gap-2">
            <span>⚠️</span> EXPLANATION FAILURE & DIAGNOSTIC TAXONOMY
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Systematic detection of signal degradation, cross-method disagreement, and corruption shifts.
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-mono text-slate-400">Filter:</label>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Diagnostic Patterns ({allFlags.length})</option>
            <option value="method_disagreement">Method Disagreement</option>
            <option value="attribution_shift_under_corruption">Attribution Shift under Corruption</option>
            <option value="prediction_flip_with_stable_attribution">Prediction Flip with Stable Attribution</option>
            <option value="large_attribution_shift_with_stable_prediction">Large Shift with Stable Prediction</option>
            <option value="diffuse_attribution">Diffuse Attribution</option>
            <option value="localized_single_region">Localized Single Region</option>
            <option value="low_attribution_signal">Low Attribution Signal</option>
          </select>
        </div>
      </div>

      {/* Failure Records Table */}
      {filteredFlags.length === 0 ? (
        <div className="p-8 text-center bg-slate-950/40 border border-dashed border-slate-800 rounded-xl">
          <span className="text-2xl mb-2 block">✨</span>
          <span className="text-xs font-bold text-slate-400 font-mono">No flagged failure patterns under this filter</span>
          <p className="text-[11px] text-slate-500 mt-1">Attribution signals are stable and within normal bounds.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-left">
                <th className="p-3">Sample</th>
                <th className="p-3">Category</th>
                <th className="p-3">Severity</th>
                <th className="p-3">Description</th>
                <th className="p-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredFlags.map((item, idx) => (
                <tr
                  key={`${item.sampleId}-${item.flag.category}-${idx}`}
                  className="border-b border-slate-800/40 hover:bg-slate-800/30 transition-all cursor-pointer"
                  onClick={() => onSelectSample(item.sampleId)}
                >
                  <td className="p-3 font-bold text-slate-200">
                    <div>{item.sampleName}</div>
                    <div className="text-[10px] text-slate-500 font-normal">{item.trueClass}</div>
                  </td>
                  <td className="p-3 font-semibold text-cyan-300">
                    {item.flag.category.replace(/_/g, " ")}
                  </td>
                  <td className="p-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] border uppercase ${getSeverityBadge(item.flag.severity)}`}>
                      {item.flag.severity}
                    </span>
                  </td>
                  <td className="p-3 text-slate-400 max-w-md text-[11px]">
                    {item.flag.description}
                  </td>
                  <td className="p-3 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectSample(item.sampleId);
                      }}
                      className="px-2.5 py-1 rounded bg-slate-950 hover:bg-cyan-950 text-cyan-400 border border-slate-800 hover:border-cyan-700 text-[10px] font-bold transition-all"
                    >
                      Inspect →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
