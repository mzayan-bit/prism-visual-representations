"use client";

import React from "react";
import { CalibrationReportPayload, UncertaintySampleItemPayload } from "../types";

interface ConfidenceHistogramCardProps {
  report: CalibrationReportPayload;
  samples: UncertaintySampleItemPayload[];
}

export const ConfidenceHistogramCard: React.FC<ConfidenceHistogramCardProps> = ({
  report,
  samples,
}) => {
  const numBins = 10;
  const correctBins = new Array(numBins).fill(0);
  const incorrectBins = new Array(numBins).fill(0);
  const oodBins = new Array(numBins).fill(0);

  samples.forEach((s) => {
    const binIdx = Math.min(Math.floor(s.confidence * numBins), numBins - 1);
    if (s.category === "IN_DISTRIBUTION") {
      if (s.is_correct) {
        correctBins[binIdx] += 1;
      } else {
        incorrectBins[binIdx] += 1;
      }
    } else {
      oodBins[binIdx] += 1;
    }
  });

  const maxFreq = Math.max(
    ...correctBins,
    ...incorrectBins,
    ...oodBins,
    1
  );

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              Confidence & Entropy Distributions
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/40">
              CORRECT vs INCORRECT vs OOD
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Distribution of max softmax probability and entropy partitioned by correctness
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-emerald-500 rounded-sm"></div>
            <span className="text-slate-300">Correct ID</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-rose-500 rounded-sm"></div>
            <span className="text-slate-300">Incorrect ID</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-amber-500 rounded-sm"></div>
            <span className="text-slate-300">OOD</span>
          </div>
        </div>
      </div>

      {/* Histogram Visualization */}
      <div className="my-4">
        <div className="h-44 flex items-end gap-2 bg-slate-950/60 p-4 rounded-xl border border-slate-800/70">
          {Array.from({ length: numBins }).map((_, i) => {
            const low = (i / numBins).toFixed(1);
            const high = ((i + 1) / numBins).toFixed(1);
            const cCount = correctBins[i];
            const eCount = incorrectBins[i];
            const oCount = oodBins[i];

            const cHeight = (cCount / maxFreq) * 100;
            const eHeight = (eCount / maxFreq) * 100;
            const oHeight = (oCount / maxFreq) * 100;

            return (
              <div key={i} className="flex-1 flex flex-col items-center h-full justify-end group">
                <div className="w-full flex items-end justify-center gap-0.5 h-full">
                  {/* Correct bar */}
                  <div
                    title={`Confidence [${low}, ${high}]: ${cCount} Correct ID`}
                    className="w-1/3 bg-emerald-500/80 hover:bg-emerald-400 rounded-t transition-all"
                    style={{ height: `${Math.max(cHeight, cCount > 0 ? 6 : 0)}%` }}
                  />
                  {/* Incorrect bar */}
                  <div
                    title={`Confidence [${low}, ${high}]: ${eCount} Incorrect ID`}
                    className="w-1/3 bg-rose-500/80 hover:bg-rose-400 rounded-t transition-all"
                    style={{ height: `${Math.max(eHeight, eCount > 0 ? 6 : 0)}%` }}
                  />
                  {/* OOD bar */}
                  <div
                    title={`Confidence [${low}, ${high}]: ${oCount} OOD Samples`}
                    className="w-1/3 bg-amber-500/80 hover:bg-amber-400 rounded-t transition-all"
                    style={{ height: `${Math.max(oHeight, oCount > 0 ? 6 : 0)}%` }}
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
          Max Softmax Probability Range [0.0 → 1.0]
        </div>
      </div>

      {/* Statistical Summary Strip */}
      {(() => {
        const correctSummary = report.correct_subset_summary || report.correct_predictions_summary || {
          sample_count: 0,
          mean_max_probability: 0,
          median_max_probability: 0,
          mean_entropy: 0,
          mean_normalized_entropy: 0,
        };
        const errorSummary = report.error_subset_summary || report.incorrect_predictions_summary || {
          sample_count: 0,
          mean_max_probability: 0,
          median_max_probability: 0,
          mean_entropy: 0,
          mean_normalized_entropy: 0,
        };
        return (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
              <div className="text-[11px] font-bold text-emerald-400 flex items-center justify-between">
                <span>Correct Predictions (N={correctSummary.sample_count}):</span>
                <span className="font-mono">{(correctSummary.mean_max_probability * 100).toFixed(1)}% Conf</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1 flex justify-between font-mono">
                <span>Median Conf: {(correctSummary.median_max_probability * 100).toFixed(1)}%</span>
                <span>Entropy: {correctSummary.mean_entropy.toFixed(3)} nats</span>
              </div>
            </div>

            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
              <div className="text-[11px] font-bold text-rose-400 flex items-center justify-between">
                <span>Incorrect Predictions (N={errorSummary.sample_count}):</span>
                <span className="font-mono">{(errorSummary.mean_max_probability * 100).toFixed(1)}% Conf</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1 flex justify-between font-mono">
                <span>Median Conf: {(errorSummary.median_max_probability * 100).toFixed(1)}%</span>
                <span>Entropy: {errorSummary.mean_entropy.toFixed(3)} nats</span>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
