"use client";

import React from "react";
import { MultimodalObjectiveComparisonPayload } from "../types";

interface MultimodalObjectiveComparisonCardProps {
  comparisons: MultimodalObjectiveComparisonPayload[];
}

export const MultimodalObjectiveComparisonCard: React.FC<
  MultimodalObjectiveComparisonCardProps
> = ({ comparisons }) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-bold text-sm">⚖️ Pretraining Objective Comparison</span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          Vision-Language vs Unimodal SSL & Supervised
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="pb-2 font-semibold">Objective</th>
              <th className="pb-2 font-semibold">Supervision</th>
              <th className="pb-2 font-semibold text-right">Linear Probe</th>
              <th className="pb-2 font-semibold text-right">Zero-Shot</th>
              <th className="pb-2 font-semibold text-right">Retrieval R@1</th>
              <th className="pb-2 font-semibold text-right">Eff. Dim</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {comparisons.map((c) => {
              const isVL = c.objective === "vision_language";

              return (
                <tr
                  key={c.objective}
                  className={`hover:bg-slate-800/30 transition-colors ${
                    isVL ? "bg-cyan-950/20 font-semibold" : ""
                  }`}
                >
                  <td className="py-2.5 flex items-center gap-1.5">
                    {isVL && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                    <span className={isVL ? "text-cyan-300" : "text-slate-200"}>
                      {c.objective.toUpperCase().replace("_", "-")}
                    </span>
                  </td>
                  <td className="py-2.5 text-slate-400">{c.label_supervision}</td>
                  <td className="py-2.5 text-right font-bold text-slate-200">
                    {c.linear_probe_accuracy !== null
                      ? `${(c.linear_probe_accuracy * 100).toFixed(1)}%`
                      : "N/A"}
                  </td>
                  <td className="py-2.5 text-right font-bold">
                    {c.zero_shot_accuracy !== null ? (
                      <span className="text-emerald-400">
                        {(c.zero_shot_accuracy * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-slate-600">N/A</span>
                    )}
                  </td>
                  <td className="py-2.5 text-right font-bold">
                    {c.image_to_text_r1 !== null ? (
                      <span className="text-cyan-400">
                        {(c.image_to_text_r1 * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-slate-600">N/A</span>
                    )}
                  </td>
                  <td className="py-2.5 text-right text-indigo-400">
                    {c.effective_dimensionality.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
