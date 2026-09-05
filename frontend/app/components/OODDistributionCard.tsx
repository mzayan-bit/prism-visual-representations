"use client";

import React from "react";
import { OODBinaryEvaluationSummaryPayload, UncertaintySampleItemPayload } from "../types";

interface OODDistributionCardProps {
  evaluation?: OODBinaryEvaluationSummaryPayload;
  activeOODEval?: OODBinaryEvaluationSummaryPayload;
  samples: UncertaintySampleItemPayload[];
  selectedScoreMethod?: string;
  selectedMethod?: string;
}

export const OODDistributionCard: React.FC<OODDistributionCardProps> = ({
  evaluation,
  activeOODEval,
  samples,
  selectedScoreMethod,
  selectedMethod,
}) => {
  const currentEval = evaluation || activeOODEval;
  const currentMethod = selectedScoreMethod || selectedMethod || "msp";

  if (!currentEval) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h2 className="text-sm font-bold text-slate-100">OOD Score Distribution</h2>
        <p className="text-xs text-slate-400 mt-2">OOD scoring evaluation not available.</p>
      </div>
    );
  }

  // Create 10 score bins [0.0..1.0] or scaled range
  const numBins = 10;
  const idBins = new Array(numBins).fill(0);
  const oodBins = new Array(numBins).fill(0);

  const idScores: number[] = [];
  const oodScores: number[] = [];

  samples.forEach((s) => {
    let score = s.msp_score;
    if (selectedMethod === "nearest_class_centroid_distance") {
      score = Math.min(s.centroid_distance / 3.5, 1.0);
    } else if (selectedMethod === "knn_representation_distance") {
      score = Math.min(s.knn_distance / 3.5, 1.0);
    } else if (selectedMethod === "predictive_entropy") {
      score = Math.min(s.entropy / 1.1, 1.0);
    }

    const binIdx = Math.min(Math.floor(score * numBins), numBins - 1);
    if (s.category === "IN_DISTRIBUTION") {
      idBins[binIdx] += 1;
      idScores.push(score);
    } else {
      oodBins[binIdx] += 1;
      oodScores.push(score);
    }
  });

  const maxFreq = Math.max(...idBins, ...oodBins, 1);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              OOD Novelty Score Separation
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800/40">
              {currentMethod.replace(/_/g, " ").toUpperCase()}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Polarity Normalized: Higher Score = More Out-of-Distribution Like
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-cyan-500 rounded-sm"></div>
            <span className="text-slate-300">In-Distribution (ID)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-amber-500 rounded-sm"></div>
            <span className="text-slate-300">Out-of-Distribution (OOD)</span>
          </div>
        </div>
      </div>

      {/* Distribution Chart */}
      <div className="my-4">
        <div className="h-44 flex items-end gap-2 bg-slate-950/60 p-4 rounded-xl border border-slate-800/70 relative">
          {/* Decision Threshold Line */}
          <div
            className="absolute top-2 bottom-6 w-0.5 bg-rose-500 border-r border-dashed border-rose-400 z-10"
            style={{ left: `${Math.min(Math.max(currentEval.threshold * 100, 10), 90)}%` }}
          >
            <span className="absolute -top-1 -translate-x-1/2 text-[9px] font-mono bg-rose-950 text-rose-300 px-1 py-0.5 rounded border border-rose-700">
              θ = {currentEval.threshold.toFixed(2)}
            </span>
          </div>

          {Array.from({ length: numBins }).map((_, i) => {
            const low = (i / numBins).toFixed(1);
            const high = ((i + 1) / numBins).toFixed(1);
            const idCount = idBins[i];
            const oodCount = oodBins[i];

            const idHeight = (idCount / maxFreq) * 100;
            const oodHeight = (oodCount / maxFreq) * 100;

            return (
              <div key={i} className="flex-1 flex flex-col items-center h-full justify-end group">
                <div className="w-full flex items-end justify-center gap-1 h-full">
                  {/* ID Bar */}
                  <div
                    title={`Score [${low}, ${high}]: ${idCount} ID Samples`}
                    className="w-1/2 bg-cyan-500/80 hover:bg-cyan-400 rounded-t transition-all"
                    style={{ height: `${Math.max(idHeight, idCount > 0 ? 6 : 0)}%` }}
                  />
                  {/* OOD Bar */}
                  <div
                    title={`Score [${low}, ${high}]: ${oodCount} OOD Samples`}
                    className="w-1/2 bg-amber-500/80 hover:bg-amber-400 rounded-t transition-all"
                    style={{ height: `${Math.max(oodHeight, oodCount > 0 ? 6 : 0)}%` }}
                  />
                </div>
                <span className="text-[10px] font-mono text-slate-500 mt-1">
                  {low}
                </span>
              </div>
            );
          })}
        </div>
        <div className="text-center text-[10px] font-mono text-slate-400 mt-1">
          Normalized OOD Novelty Score [0.0 → 1.0] (Higher = More OOD-like)
        </div>
      </div>

      {/* Metrics Readout */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Mean ID Score</div>
          <div className="text-base font-bold font-mono text-cyan-400 mt-0.5">
            {currentEval.mean_id_score.toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-500">N={currentEval.id_sample_count} ID Samples</div>
        </div>

        <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Mean OOD Score</div>
          <div className="text-base font-bold font-mono text-amber-400 mt-0.5">
            {currentEval.mean_ood_score.toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-500">N={currentEval.ood_sample_count} OOD Samples</div>
        </div>

        <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Separation Gap</div>
          <div className="text-base font-bold font-mono text-emerald-400 mt-0.5">
            +{currentEval.score_separation_gap.toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-500">Δ(OOD - ID) Novelty Margin</div>
        </div>
      </div>
    </div>
  );
};
