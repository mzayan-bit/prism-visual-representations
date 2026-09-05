"use client";

import React, { useState } from "react";
import {
  CorruptionUncertaintyCurvePayload,
  PredictionFlipUncertaintyPayload,
} from "../types";

interface CorruptionUncertaintyCardProps {
  curves: CorruptionUncertaintyCurvePayload[];
  predictionFlips: PredictionFlipUncertaintyPayload[];
  selectedCorruption: string;
  onSelectCorruption: (corruption: string) => void;
}

export const CorruptionUncertaintyCard: React.FC<CorruptionUncertaintyCardProps> = ({
  curves,
  predictionFlips,
  selectedCorruption,
  onSelectCorruption,
}) => {
  const [activeMetricView, setActiveMetricView] = useState<
    "all" | "confidence" | "entropy" | "drift" | "ece"
  >("all");
  const [filterFlipSeverity, setFilterFlipSeverity] = useState<number | null>(null);

  const activeCurve =
    curves.find((c) => c.corruption_type === selectedCorruption) || curves[0];

  const filteredFlips = predictionFlips.filter((f) => {
    const matchesCorr = f.corruption_type === selectedCorruption;
    const matchesSev =
      filterFlipSeverity === null || f.severity === filterFlipSeverity;
    return matchesCorr && matchesSev;
  });

  if (!activeCurve) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-400">
        No corruption uncertainty curves available.
      </div>
    );
  }

  // SVG dimensions for trend chart
  const width = 640;
  const height = 220;
  const padLeft = 45;
  const padRight = 30;
  const padTop = 20;
  const padBottom = 30;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const severities = activeCurve.severities; // e.g. [1, 2, 3, 4, 5]
  const numPts = severities.length;

  const getX = (idx: number) => padLeft + (idx / Math.max(1, numPts - 1)) * plotW;
  const getY = (val: number, min = 0, max = 1) => {
    const clamped = Math.max(min, Math.min(max, val));
    const norm = (clamped - min) / (max - min || 1);
    return padTop + (1 - norm) * plotH;
  };

  const makeLinePath = (data: number[], min = 0, max = 1) => {
    return data
      .map((val, i) => `${i === 0 ? "M" : "L"} ${getX(i)},${getY(val, min, max)}`)
      .join(" ");
  };

  const maxDrift = Math.max(
    1.0,
    ...activeCurve.mean_representation_drifts.map((d) => d || 0)
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400 font-bold text-sm">
            🌪️
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-200">
              Corruption Uncertainty & Robustness Dynamics
            </h2>
            <p className="text-xs text-slate-400">
              Tracking predictive degradation, confidence collapse, and representation drift across perturbation severities
            </p>
          </div>
        </div>

        {/* Corruption Selector Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {curves.map((curve) => (
            <button
              key={curve.corruption_type}
              onClick={() => onSelectCorruption(curve.corruption_type)}
              className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all ${
                selectedCorruption === curve.corruption_type
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/50 shadow-sm"
                  : "bg-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-700/60"
              }`}
            >
              {curve.corruption_type.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Metric Cards Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[11px] text-slate-400">Confidence Slope</div>
          <div className="text-base font-mono font-bold text-cyan-400 mt-0.5">
            {activeCurve.confidence_slope.toFixed(4)}
            <span className="text-[10px] text-slate-500 font-normal ml-1">/level</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            Rate of confidence decay
          </div>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[11px] text-slate-400">Entropy Monotonicity</div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span
              className={`text-sm font-semibold ${
                activeCurve.is_monotonic_entropy
                  ? "text-emerald-400"
                  : "text-amber-400"
              }`}
            >
              {activeCurve.is_monotonic_entropy ? "✓ Monotonic" : "⚠ Non-Monotonic"}
            </span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            Entropy increases with severity
          </div>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[11px] text-slate-400">Accuracy @ Sev 5</div>
          <div className="text-base font-mono font-bold text-rose-400 mt-0.5">
            {(
              (activeCurve.accuracies[activeCurve.accuracies.length - 1] ?? 0) *
              100
            ).toFixed(1)}
            %
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            Baseline: {((activeCurve.accuracies[0] ?? 0) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
          <div className="text-[11px] text-slate-400">Peak Rep. Drift</div>
          <div className="text-base font-mono font-bold text-amber-400 mt-0.5">
            {(
              activeCurve.mean_representation_drifts[
                activeCurve.mean_representation_drifts.length - 1
              ] ?? 0
            ).toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            Euclidean feature shift
          </div>
        </div>
      </div>

      {/* Trajectory Multi-Line SVG Chart */}
      <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <div className="text-xs font-semibold text-slate-300">
            Degradation Curves Across Severity Levels (1 to 5)
          </div>
          {/* Chart Filter Toggles */}
          <div className="flex items-center gap-1.5 text-[11px]">
            {(
              [
                { id: "all", label: "All" },
                { id: "confidence", label: "Accuracy & Conf" },
                { id: "entropy", label: "Entropy & ECE" },
                { id: "drift", label: "Rep Drift" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveMetricView(tab.id)}
                className={`px-2 py-0.5 rounded transition-all ${
                  activeMetricView === tab.id
                    ? "bg-slate-700 text-cyan-300 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto max-w-[640px] mx-auto text-slate-400 font-mono text-[10px]"
          >
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
              <g key={tick}>
                <line
                  x1={padLeft}
                  y1={getY(tick, 0, 1)}
                  x2={width - padRight}
                  y2={getY(tick, 0, 1)}
                  stroke="#334155"
                  strokeDasharray="3 3"
                  strokeWidth="0.5"
                />
                <text
                  x={padLeft - 6}
                  y={getY(tick, 0, 1) + 3}
                  textAnchor="end"
                  fill="#64748b"
                >
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Severity vertical grid lines */}
            {severities.map((sev, idx) => (
              <g key={sev}>
                <line
                  x1={getX(idx)}
                  y1={padTop}
                  x2={getX(idx)}
                  y2={height - padBottom}
                  stroke="#334155"
                  strokeDasharray="2 2"
                  strokeWidth="0.5"
                />
                <text
                  x={getX(idx)}
                  y={height - padBottom + 15}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontWeight="600"
                >
                  Sev {sev}
                </text>
              </g>
            ))}

            {/* Accuracy Line (Emerald) */}
            {(activeMetricView === "all" || activeMetricView === "confidence") && (
              <path
                d={makeLinePath(activeCurve.accuracies, 0, 1)}
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeLinecap="round"
              />
            )}

            {/* Mean Confidence Line (Cyan) */}
            {(activeMetricView === "all" || activeMetricView === "confidence") && (
              <path
                d={makeLinePath(activeCurve.mean_confidences, 0, 1)}
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2"
                strokeDasharray="4 2"
                strokeLinecap="round"
              />
            )}

            {/* Mean Entropy Line (Amber) */}
            {(activeMetricView === "all" || activeMetricView === "entropy") && (
              <path
                d={makeLinePath(activeCurve.mean_entropies, 0, 1)}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2"
                strokeLinecap="round"
              />
            )}

            {/* ECE Line (Rose) */}
            {(activeMetricView === "all" || activeMetricView === "entropy") && (
              <path
                d={makeLinePath(activeCurve.eces, 0, 0.5)}
                fill="none"
                stroke="#f43f5e"
                strokeWidth="1.5"
                strokeDasharray="2 2"
                strokeLinecap="round"
              />
            )}

            {/* Representation Drift Line (Purple) */}
            {(activeMetricView === "all" || activeMetricView === "drift") && (
              <path
                d={makeLinePath(activeCurve.mean_representation_drifts, 0, maxDrift)}
                fill="none"
                stroke="#a855f7"
                strokeWidth="2"
                strokeLinecap="round"
              />
            )}

            {/* Data point dots */}
            {severities.map((sev, idx) => (
              <g key={`dots-${sev}`}>
                {(activeMetricView === "all" || activeMetricView === "confidence") && (
                  <circle
                    cx={getX(idx)}
                    cy={getY(activeCurve.accuracies[idx] ?? 0, 0, 1)}
                    r="4"
                    fill="#10b981"
                    stroke="#0f172a"
                    strokeWidth="1.5"
                  />
                )}
                {(activeMetricView === "all" || activeMetricView === "confidence") && (
                  <circle
                    cx={getX(idx)}
                    cy={getY(activeCurve.mean_confidences[idx] ?? 0, 0, 1)}
                    r="3.5"
                    fill="#06b6d4"
                    stroke="#0f172a"
                    strokeWidth="1.5"
                  />
                )}
                {(activeMetricView === "all" || activeMetricView === "entropy") && (
                  <circle
                    cx={getX(idx)}
                    cy={getY(activeCurve.mean_entropies[idx] ?? 0, 0, 1)}
                    r="3.5"
                    fill="#f59e0b"
                    stroke="#0f172a"
                    strokeWidth="1.5"
                  />
                )}
                {(activeMetricView === "all" || activeMetricView === "drift") && (
                  <circle
                    cx={getX(idx)}
                    cy={getY(
                      activeCurve.mean_representation_drifts[idx] ?? 0,
                      0,
                      maxDrift
                    )}
                    r="3.5"
                    fill="#a855f7"
                    stroke="#0f172a"
                    strokeWidth="1.5"
                  />
                )}
              </g>
            ))}
          </svg>
        </div>

        {/* Chart Legend */}
        <div className="flex flex-wrap items-center justify-center gap-4 mt-3 pt-2 border-t border-slate-800 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-emerald-500 rounded" />
            <span className="text-slate-300">Accuracy</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-cyan-400 border-b border-dashed border-cyan-400" />
            <span className="text-slate-300">Mean Confidence</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-amber-500 rounded" />
            <span className="text-slate-300">Entropy</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-rose-500 border-b border-dotted border-rose-500" />
            <span className="text-slate-300">ECE (0-0.5 scale)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-purple-500 rounded" />
            <span className="text-slate-300">Rep Drift (0-{maxDrift.toFixed(1)})</span>
          </div>
        </div>
      </div>

      {/* Prediction Flip Analysis Table */}
      <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-xs font-semibold text-slate-200">
              Prediction Flip Diagnostics ({selectedCorruption})
            </h3>
            <p className="text-[11px] text-slate-400">
              Samples whose predicted class flipped under corruption, tracking post-flip overconfidence
            </p>
          </div>

          {/* Severity Filter */}
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-500 text-[11px]">Severity:</span>
            <button
              onClick={() => setFilterFlipSeverity(null)}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                filterFlipSeverity === null
                  ? "bg-slate-700 text-slate-200 font-semibold"
                  : "bg-slate-800/80 text-slate-400 hover:text-slate-300"
              }`}
            >
              All
            </button>
            {[1, 2, 3, 4, 5].map((sev) => (
              <button
                key={sev}
                onClick={() => setFilterFlipSeverity(sev)}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                  filterFlipSeverity === sev
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                    : "bg-slate-800/80 text-slate-400 hover:text-slate-300"
                }`}
              >
                S{sev}
              </button>
            ))}
          </div>
        </div>

        {filteredFlips.length === 0 ? (
          <div className="text-center py-6 text-slate-500 text-xs italic">
            No prediction flips recorded for this corruption/severity configuration.
          </div>
        ) : (
          <div className="overflow-x-auto max-h-60 overflow-y-auto">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] text-slate-400 uppercase bg-slate-900/60 sticky top-0">
                  <th className="py-2 px-2.5">Sample ID</th>
                  <th className="py-2 px-2">Sev</th>
                  <th className="py-2 px-2">Clean Class</th>
                  <th className="py-2 px-2">Corrupted Class</th>
                  <th className="py-2 px-2 text-right">Clean Conf</th>
                  <th className="py-2 px-2 text-right">Post Conf</th>
                  <th className="py-2 px-2 text-right">Post Entropy</th>
                  <th className="py-2 px-2 text-right">Rep Drift</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-[11px]">
                {filteredFlips.map((flip, idx) => {
                  const isHighConfError = flip.corrupted_confidence > 0.8;
                  return (
                    <tr
                      key={`${flip.sample_id}-${flip.severity}-${idx}`}
                      className="hover:bg-slate-800/40 transition-colors"
                    >
                      <td className="py-1.5 px-2.5 text-slate-300 font-semibold">
                        {flip.sample_id}
                      </td>
                      <td className="py-1.5 px-2">
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px] border border-slate-700">
                          S{flip.severity}
                        </span>
                      </td>
                      <td className="py-1.5 px-2 text-emerald-400">
                        Class {flip.clean_prediction}
                      </td>
                      <td className="py-1.5 px-2 text-rose-400 font-semibold">
                        Class {flip.corrupted_prediction}
                      </td>
                      <td className="py-1.5 px-2 text-right text-slate-400">
                        {(flip.clean_confidence * 100).toFixed(1)}%
                      </td>
                      <td
                        className={`py-1.5 px-2 text-right font-semibold ${
                          isHighConfError ? "text-rose-400" : "text-amber-300"
                        }`}
                      >
                        {(flip.corrupted_confidence * 100).toFixed(1)}%
                        {isHighConfError && (
                          <span
                            className="ml-1 text-[9px] text-rose-400"
                            title="Overconfident flip error"
                          >
                            ⚠️
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 px-2 text-right text-slate-300">
                        {flip.corrupted_entropy.toFixed(3)}
                      </td>
                      <td className="py-1.5 px-2 text-right text-purple-400">
                        {flip.representation_drift.toFixed(3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
