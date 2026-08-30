"use client";

import React, { useState } from "react";
import { RobustnessFailureRecord } from "../types";

interface RobustnessFailureExplorerProps {
  failures: RobustnessFailureRecord[];
  onSelectSample: (sampleId: string) => void;
  selectedSampleId: string | null;
}

export default function RobustnessFailureExplorer({
  failures,
  onSelectSample,
  selectedSampleId,
}: RobustnessFailureExplorerProps) {
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");

  const categories = Array.from(new Set(failures.map((f) => f.category)));

  const filteredFailures = failures.filter((f) => {
    const matchesCat = filterCategory === "all" || f.category === filterCategory;
    const matchesQuery =
      searchQuery === "" ||
      f.sample_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      f.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCat && matchesQuery;
  });

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>⚠️</span> Robustness Failure Taxonomy & Anomaly Explorer
          </h2>
          <p className="text-xs text-slate-400">
            Identified prediction flips, manifold collapses, and severe representation shifts
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-2">
          {/* Category Filter */}
          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="bg-slate-950/60 border border-slate-800 text-xs text-slate-200 px-3 py-1.5 rounded-lg outline-none cursor-pointer"
          >
            <option value="all">All Categories ({failures.length})</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat.replace(/_/g, " ").toUpperCase()} (
                {failures.filter((f) => f.category === cat).length})
              </option>
            ))}
          </select>

          {/* Search Input */}
          <input
            type="text"
            placeholder="Search sample..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-slate-950/60 border border-slate-800 text-xs text-slate-200 px-3 py-1.5 rounded-lg outline-none placeholder:text-slate-500 w-36"
          />
        </div>
      </div>

      {/* Failure Records Table */}
      {filteredFailures.length === 0 ? (
        <div className="p-6 rounded-xl bg-slate-950/40 border border-slate-800/80 text-center text-slate-400 text-xs">
          No failures match current filters.
        </div>
      ) : (
        <div className="overflow-x-auto max-h-[360px] overflow-y-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="sticky top-0 bg-slate-950/90 backdrop-blur-sm z-10">
              <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                <th className="py-2 px-3">Sample ID</th>
                <th className="py-2 px-3">Category</th>
                <th className="py-2 px-3">Severity</th>
                <th className="py-2 px-3">Description</th>
                <th className="py-2 px-3">Drift (L2)</th>
                <th className="py-2 px-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {filteredFailures.map((failure, idx) => {
                const isSelected = selectedSampleId === failure.sample_id;

                return (
                  <tr
                    key={`${failure.sample_id}-${idx}`}
                    className={`transition-colors ${
                      isSelected
                        ? "bg-cyan-500/10 border-l-2 border-cyan-400"
                        : "hover:bg-slate-800/40"
                    }`}
                  >
                    <td className="py-2.5 px-3 font-mono font-bold text-white">
                      {failure.sample_id}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                        {failure.category.replace(/_/g, " ")}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="px-1.5 py-0.5 rounded text-[11px] font-bold bg-slate-800 text-slate-300">
                        Sev {failure.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 max-w-xs truncate">
                      {failure.description}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-cyan-300 font-bold">
                      {failure.metrics.drift ? failure.metrics.drift.toFixed(3) : "—"}
                    </td>
                    <td className="py-2.5 px-3">
                      <button
                        onClick={() => onSelectSample(failure.sample_id)}
                        className={`px-2 py-1 rounded text-[11px] font-semibold transition-all ${
                          isSelected
                            ? "bg-cyan-600 text-white"
                            : "bg-slate-800 hover:bg-slate-700 text-slate-300"
                        }`}
                      >
                        {isSelected ? "Inspecting" : "Inspect"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
