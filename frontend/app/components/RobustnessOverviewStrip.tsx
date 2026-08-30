"use client";

import React from "react";
import { CorruptionEvaluationSummary } from "../types";

interface RobustnessOverviewStripProps {
  evaluation: CorruptionEvaluationSummary | null;
}

export default function RobustnessOverviewStrip({
  evaluation,
}: RobustnessOverviewStripProps) {
  if (!evaluation) {
    return (
      <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-slate-400 text-sm">
        No evaluation data available for selected parameters.
      </div>
    );
  }

  const cleanAccPct = (evaluation.clean_accuracy * 100).toFixed(1);
  const corrAccPct = (evaluation.corrupted_accuracy * 100).toFixed(1);
  const accDropPct = (evaluation.absolute_accuracy_drop * 100).toFixed(1);
  const consistencyPct = (
    evaluation.prediction_consistency_fraction * 100
  ).toFixed(1);
  const neighborOverlapPct = (
    evaluation.geometry_drift.neighborhood_drift.mean_neighbor_overlap_ratio * 100
  ).toFixed(1);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {/* 1. Accuracy Metric Card */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm relative overflow-hidden flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span>Top-1 Accuracy</span>
          <span className="text-emerald-400">Clean: {cleanAccPct}%</span>
        </div>
        <div className="my-2">
          <div className="text-2xl font-bold text-white tracking-tight">
            {corrAccPct}%
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span
              className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                evaluation.absolute_accuracy_drop > 0
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                  : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
              }`}
            >
              {evaluation.absolute_accuracy_drop > 0 ? "↓" : "↑"}{" "}
              {Math.abs(Number(accDropPct))}% drop
            </span>
          </div>
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
          <div
            className="bg-cyan-500 h-full transition-all"
            style={{ width: `${corrAccPct}%` }}
          />
        </div>
      </div>

      {/* 2. Loss Metric Card */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm relative overflow-hidden flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span>Cross-Entropy Loss</span>
          <span className="text-slate-400">Clean: {evaluation.clean_loss.toFixed(3)}</span>
        </div>
        <div className="my-2">
          <div className="text-2xl font-bold text-white tracking-tight">
            {evaluation.corrupted_loss.toFixed(3)}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span
              className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                evaluation.loss_increase > 0
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                  : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
              }`}
            >
              +{evaluation.loss_increase.toFixed(3)} loss
            </span>
          </div>
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
          <div
            className="bg-amber-500 h-full transition-all"
            style={{
              width: `${Math.min(100, (evaluation.corrupted_loss / (evaluation.clean_loss + 1.0)) * 50)}%`,
            }}
          />
        </div>
      </div>

      {/* 3. Representation Drift Card */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm relative overflow-hidden flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span>Representation Drift</span>
          <span className="text-cyan-400">L2 Distance</span>
        </div>
        <div className="my-2">
          <div className="text-2xl font-bold text-cyan-300 tracking-tight">
            {evaluation.representation_drift.mean_euclidean_drift.toFixed(3)}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5 text-xs text-slate-400">
            <span>± {evaluation.representation_drift.std_euclidean_drift.toFixed(3)} std</span>
          </div>
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
          <div className="bg-cyan-400 h-full w-3/4" />
        </div>
      </div>

      {/* 4. Cosine Similarity Card */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm relative overflow-hidden flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span>Cosine Alignment</span>
          <span className="text-slate-400">cos(z, z&apos;)</span>
        </div>
        <div className="my-2">
          <div className="text-2xl font-bold text-white tracking-tight">
            {evaluation.representation_drift.mean_cosine_similarity.toFixed(3)}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5 text-xs text-slate-400">
            <span>Dist: {evaluation.representation_drift.mean_cosine_distance.toFixed(3)}</span>
          </div>
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
          <div
            className="bg-indigo-400 h-full transition-all"
            style={{
              width: `${Math.max(0, evaluation.representation_drift.mean_cosine_similarity * 100)}%`,
            }}
          />
        </div>
      </div>

      {/* 5. k-NN Neighbor Retention */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm relative overflow-hidden flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span>Neighbor Retention</span>
          <span className="text-slate-400">k={evaluation.geometry_drift.neighborhood_drift.k}</span>
        </div>
        <div className="my-2">
          <div className="text-2xl font-bold text-emerald-300 tracking-tight">
            {neighborOverlapPct}%
          </div>
          <div className="flex items-center gap-1.5 mt-0.5 text-xs text-slate-400">
            <span>Flips: {(evaluation.geometry_drift.neighborhood_drift.nearest_neighbor_label_flip_fraction * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
          <div
            className="bg-emerald-500 h-full transition-all"
            style={{ width: `${neighborOverlapPct}%` }}
          />
        </div>
      </div>

      {/* 6. Prediction Consistency */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm relative overflow-hidden flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
          <span>Consistency</span>
          <span className="text-slate-400">Unchanged</span>
        </div>
        <div className="my-2">
          <div className="text-2xl font-bold text-purple-300 tracking-tight">
            {consistencyPct}%
          </div>
          <div className="flex items-center gap-1.5 mt-0.5 text-xs text-slate-400">
            <span>{evaluation.predictions_changed_count} changed</span>
          </div>
        </div>
        <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
          <div
            className="bg-purple-500 h-full transition-all"
            style={{ width: `${consistencyPct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
