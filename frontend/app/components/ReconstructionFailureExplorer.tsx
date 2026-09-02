"use client";

import React, { useState } from "react";
import { ReconstructionFailureCasePayload } from "../types";

interface ReconstructionFailureExplorerProps {
  failureCases: ReconstructionFailureCasePayload[];
}

export function ReconstructionFailureExplorer({
  failureCases,
}: ReconstructionFailureExplorerProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  const categories = [
    "all",
    ...Array.from(new Set(failureCases.map((f) => f.category))),
  ];

  const filteredCases =
    selectedCategory === "all"
      ? failureCases
      : failureCases.filter((f) => f.category === selectedCategory);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            Reconstruction Failure Explorer
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Diagnosed failure modes categorized under PRISM failure taxonomy.
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex items-center gap-1.5 overflow-x-auto bg-slate-950 p-1 rounded-lg border border-slate-800">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-2.5 py-1 rounded text-xs font-medium uppercase tracking-wider transition-all whitespace-nowrap ${
                selectedCategory === cat
                  ? "bg-rose-900/80 text-rose-200 font-semibold border border-rose-700/80 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {cat.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Failure Cases Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {filteredCases.map((fCase) => (
          <div
            key={fCase.sample_id}
            className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 text-xs flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="font-mono font-semibold text-slate-300">
                  {fCase.sample_id}
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-950/80 text-rose-400 border border-rose-800/60 uppercase">
                  MSE: {fCase.reconstruction_mse.toFixed(4)}
                </span>
              </div>
              <div className="text-[11px] font-semibold text-rose-300 uppercase tracking-wider mb-1.5">
                {fCase.category.replace(/_/g, " ")}
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                {fCase.description}
              </p>
            </div>

            {fCase.patch_index !== null && (
              <div className="mt-3 pt-2 border-t border-slate-900 text-[10px] text-slate-500 font-mono">
                Failed Patch Index: #{fCase.patch_index}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
