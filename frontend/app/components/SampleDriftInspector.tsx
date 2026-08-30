"use client";

import React from "react";
import { SampleNeighborhood, SampleRepresentationDrift } from "../types";

interface SampleDriftInspectorProps {
  sampleDrift: SampleRepresentationDrift | null;
  cleanNeighborhood: SampleNeighborhood | null;
  corruptedNeighborhood: SampleNeighborhood | null;
  classNames?: string[];
}

export default function SampleDriftInspector({
  sampleDrift,
  cleanNeighborhood,
  corruptedNeighborhood,
  classNames = ["Class 0", "Class 1", "Class 2"],
}: SampleDriftInspectorProps) {
  if (!sampleDrift) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm flex items-center justify-center min-h-[300px]">
        Select a sample in the PCA plot or Failure table to inspect its individual drift.
      </div>
    );
  }

  const labelIdx =
    typeof sampleDrift.label === "number"
      ? sampleDrift.label
      : parseInt(String(sampleDrift.label), 10) || 0;
  const trueClassName = classNames[labelIdx] || `Class ${labelIdx}`;

  const cleanPredName =
    classNames[sampleDrift.clean_prediction] || `Class ${sampleDrift.clean_prediction}`;
  const corrPredName =
    classNames[sampleDrift.corrupted_prediction] ||
    `Class ${sampleDrift.corrupted_prediction}`;

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-white tracking-tight">
              Sample Inspector: {sampleDrift.sample_id}
            </h2>
            <span className="px-2 py-0.5 text-xs font-semibold rounded bg-slate-800 text-slate-300">
              Ground Truth: {trueClassName}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Individual representation displacement and decision flip details
          </p>
        </div>

        {/* Prediction Flip Badge */}
        <div>
          {sampleDrift.prediction_changed ? (
            <span className="px-2.5 py-1 text-xs font-bold rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/30">
              ⚠️ Prediction Flipped
            </span>
          ) : (
            <span className="px-2.5 py-1 text-xs font-bold rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              ✓ Prediction Preserved
            </span>
          )}
        </div>
      </div>

      {/* Grid: Predictions & Loss Comparison */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        {/* Clean Condition */}
        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="text-xs text-slate-400 font-medium mb-1 flex items-center justify-between">
            <span>Clean Prediction</span>
            <span
              className={
                sampleDrift.clean_correct ? "text-emerald-400" : "text-rose-400"
              }
            >
              {sampleDrift.clean_correct ? "✓ Correct" : "✕ Wrong"}
            </span>
          </div>
          <div className="text-lg font-bold text-white mb-2">{cleanPredName}</div>
          <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/60">
            <span>Loss: {sampleDrift.clean_loss.toFixed(4)}</span>
          </div>
        </div>

        {/* Corrupted Condition */}
        <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="text-xs text-slate-400 font-medium mb-1 flex items-center justify-between">
            <span>Corrupted Prediction</span>
            <span
              className={
                sampleDrift.corrupted_correct
                  ? "text-emerald-400"
                  : "text-rose-400"
              }
            >
              {sampleDrift.corrupted_correct ? "✓ Correct" : "✕ Wrong"}
            </span>
          </div>
          <div className="text-lg font-bold text-white mb-2">{corrPredName}</div>
          <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/60">
            <span>Loss: {sampleDrift.corrupted_loss.toFixed(4)}</span>
            <span
              className={`font-semibold ${
                sampleDrift.corrupted_loss > sampleDrift.clean_loss
                  ? "text-amber-400"
                  : "text-emerald-400"
              }`}
            >
              Δ +{(sampleDrift.corrupted_loss - sampleDrift.clean_loss).toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Quantitative Vector Drift Metrics */}
      <div className="grid grid-cols-3 gap-2.5 mb-4 text-center">
        <div className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800">
          <div className="text-xs text-slate-400">Euclidean Drift</div>
          <div className="text-sm font-bold text-cyan-300 mt-0.5">
            {sampleDrift.euclidean_drift.toFixed(4)}
          </div>
        </div>
        <div className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800">
          <div className="text-xs text-slate-400">Cosine Similarity</div>
          <div className="text-sm font-bold text-indigo-300 mt-0.5">
            {sampleDrift.cosine_similarity.toFixed(4)}
          </div>
        </div>
        <div className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800">
          <div className="text-xs text-slate-400">Relative Norm Δ</div>
          <div className="text-sm font-bold text-purple-300 mt-0.5">
            {(sampleDrift.relative_norm_change * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Nearest Neighbor Structure Breakdown */}
      {cleanNeighborhood && corruptedNeighborhood && (
        <div className="mt-4 pt-3 border-t border-slate-800">
          <div className="text-xs font-bold text-slate-300 mb-2 flex items-center justify-between">
            <span>Local Neighborhood Structure (k={cleanNeighborhood.neighbors.length})</span>
            <span className="text-xs text-slate-400">
              Consistency: {(cleanNeighborhood.same_class_fraction * 100).toFixed(0)}% →{" "}
              {(corruptedNeighborhood.same_class_fraction * 100).toFixed(0)}%
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            {/* Clean Neighbors */}
            <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400 font-semibold block mb-1">Clean Neighbors:</span>
              <div className="space-y-1">
                {cleanNeighborhood.neighbors.map((nb, i) => (
                  <div key={i} className="flex items-center justify-between text-slate-300">
                    <span className="font-mono text-[11px]">{nb.neighbor_sample_id}</span>
                    <span
                      className={`px-1 py-0.2 rounded text-[10px] ${
                        nb.same_class
                          ? "bg-emerald-500/20 text-emerald-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      Cls {nb.neighbor_label} (d={nb.distance.toFixed(2)})
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Corrupted Neighbors */}
            <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400 font-semibold block mb-1">Corrupted Neighbors:</span>
              <div className="space-y-1">
                {corruptedNeighborhood.neighbors.map((nb, i) => (
                  <div key={i} className="flex items-center justify-between text-slate-300">
                    <span className="font-mono text-[11px]">{nb.neighbor_sample_id}</span>
                    <span
                      className={`px-1 py-0.2 rounded text-[10px] ${
                        nb.same_class
                          ? "bg-emerald-500/20 text-emerald-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      Cls {nb.neighbor_label} (d={nb.distance.toFixed(2)})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
