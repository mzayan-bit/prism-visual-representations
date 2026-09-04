"use client";

import React from "react";

interface MultimodalRobustnessCardProps {
  robustness: {
    corruptions: string[];
    results: Record<
      string,
      {
        corruption: string;
        severity: number;
        mean_paired_cosine: number;
        cosine_drop: number;
        mean_visual_drift: number;
        mean_alignment_drift: number;
        image_to_text_r1: number;
        image_to_text_r3: number;
        image_to_text_mrr: number;
        zero_shot_accuracy: number | null;
      }
    >;
  };
}

export const MultimodalRobustnessCard: React.FC<MultimodalRobustnessCardProps> = ({
  robustness,
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-amber-400 font-bold text-sm">🛡️ Alignment Robustness Under Visual Corruption</span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          Fixed Text Modality • Corrupted Image
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="pb-2 font-semibold">Corruption</th>
              <th className="pb-2 font-semibold text-right">Paired Cosine</th>
              <th className="pb-2 font-semibold text-right">Cosine Drop</th>
              <th className="pb-2 font-semibold text-right">Visual Drift</th>
              <th className="pb-2 font-semibold text-right">Align Drift</th>
              <th className="pb-2 font-semibold text-right">Retrieval R@1</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {robustness.corruptions.map((cName) => {
              const res = robustness.results[cName];
              if (!res) return null;
              const isClean = cName === "clean";

              return (
                <tr
                  key={cName}
                  className={`hover:bg-slate-800/30 transition-colors ${
                    isClean ? "bg-slate-950/40 font-semibold" : ""
                  }`}
                >
                  <td className="py-2.5 flex items-center gap-1.5">
                    <span className={isClean ? "text-emerald-400" : "text-slate-200"}>
                      {cName.toUpperCase().replace("_", " ")}
                    </span>
                  </td>
                  <td className="py-2.5 text-right font-bold text-slate-200">
                    {res.mean_paired_cosine.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-right font-bold">
                    <span className={res.cosine_drop > 0.15 ? "text-red-400" : "text-slate-400"}>
                      {res.cosine_drop.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-2.5 text-right text-cyan-400 font-bold">
                    {res.mean_visual_drift.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-right text-indigo-400 font-bold">
                    {res.mean_alignment_drift.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-right font-bold text-emerald-400">
                    {(res.image_to_text_r1 * 100).toFixed(1)}%
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
