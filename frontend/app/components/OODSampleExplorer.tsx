"use client";

import React from "react";
import { UncertaintySampleItemPayload } from "../types";

interface OODSampleExplorerProps {
  samples: UncertaintySampleItemPayload[];
  selectedSampleId?: string | null;
  onSelectSampleId?: (id: string) => void;
  selectedOODMethod?: string;
  selectedScoreMethod?: string;
  threshold?: number;
}

export const OODSampleExplorer: React.FC<OODSampleExplorerProps> = ({
  samples,
  selectedSampleId,
  onSelectSampleId,
  selectedOODMethod,
  selectedScoreMethod,
  threshold,
}) => {
  const [internalSelectedId, setInternalSelectedId] = React.useState<string>(
    samples[0]?.sample_id || ""
  );

  const activeId = selectedSampleId ?? internalSelectedId;
  const activeSample =
    samples.find((s) => s.sample_id === activeId) || samples[0];

  const handleSelect = (id: string) => {
    setInternalSelectedId(id);
    if (onSelectSampleId) onSelectSampleId(id);
  };

  const currentMethod = (selectedScoreMethod || selectedOODMethod || "msp").replace(
    /_/g,
    " "
  ).toUpperCase();

  if (!activeSample) {
    return null;
  }

  const isID = activeSample.category === "IN_DISTRIBUTION";
  const categoryBadgeClass = isID
    ? "bg-cyan-950 text-cyan-300 border-cyan-800"
    : activeSample.category === "NEAR_OOD"
    ? "bg-purple-950 text-purple-300 border-purple-800"
    : "bg-amber-950 text-amber-300 border-amber-800";

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header with Sample Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h2 className="text-sm font-bold text-slate-100">
            Sample Novelty & Uncertainty Inspector
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Diagnostic examination of individual sample representation geometry and predictive confidence
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400 font-mono">Sample:</span>
          <select
            id="select-sample-inspector"
            value={activeSample.sample_id}
            onChange={(e) => handleSelect(e.target.value)}
            className="bg-slate-950 text-indigo-300 font-mono font-bold rounded px-2.5 py-1 border border-slate-700 focus:outline-none focus:border-indigo-500 cursor-pointer"
          >
            {samples.map((s) => (
              <option key={s.sample_id} value={s.sample_id}>
                {s.sample_id} ({s.category})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Inspector Details */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 my-4 items-center">
        {/* Synthetic Visual Representation Card */}
        <div className="sm:col-span-4 flex flex-col items-center justify-center p-4 bg-slate-950/80 rounded-xl border border-slate-800">
          <div className="w-24 h-24 rounded-lg bg-slate-900 border border-slate-700 flex items-center justify-center shadow-inner relative overflow-hidden">
            {activeSample.category === "IN_DISTRIBUTION" || activeSample.category === "in_distribution" ? (
              <div className="text-3xl">
                {activeSample.predicted_class === 0 ? "⏹️" : activeSample.predicted_class === 1 ? "⏺️" : "🔼"}
              </div>
            ) : activeSample.category === "NEAR_OOD" || activeSample.category === "near_ood" ? (
              <div className="text-3xl">🔷</div>
            ) : (
              <div className="text-3xl">⭐</div>
            )}
            <span
              className={`absolute bottom-1 right-1 text-[9px] font-mono px-1.5 py-0.5 rounded border ${
                activeSample.is_ood_detected
                  ? "bg-rose-950 text-rose-300 border-rose-800"
                  : "bg-emerald-950 text-emerald-300 border-emerald-800"
              }`}
            >
              {activeSample.is_ood_detected ? "OOD FLAGGED" : "ID ACCEPTED"}
            </span>
          </div>
          <span className={`mt-2.5 text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${categoryBadgeClass}`}>
            {activeSample.category}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">{activeSample.sample_id}</span>
        </div>

        {/* Prediction & Confidence Breakdown */}
        <div className="sm:col-span-4 space-y-2.5 text-xs font-mono">
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] uppercase text-slate-400">Class Predictions</div>
            <div className="flex justify-between mt-1">
              <span className="text-slate-400">Predicted:</span>
              <span className="font-bold text-cyan-300">Class {activeSample.predicted_class}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Ground Truth:</span>
              <span className="font-bold text-slate-200">
                {activeSample.true_class !== null ? `Class ${activeSample.true_class}` : "None (OOD)"}
              </span>
            </div>
          </div>

          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] uppercase text-slate-400">Uncertainty Diagnostics</div>
            <div className="flex justify-between mt-1">
              <span className="text-slate-400">Confidence:</span>
              <span className="font-bold text-emerald-400">{(activeSample.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Entropy H(p):</span>
              <span className="font-bold text-indigo-400">{activeSample.entropy.toFixed(3)} nats</span>
            </div>
          </div>
        </div>

        {/* Representation Space Geometry */}
        <div className="sm:col-span-4 space-y-2.5 text-xs font-mono">
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] uppercase text-slate-400">Centroid Geometry</div>
            <div className="flex justify-between mt-1">
              <span className="text-slate-400">Nearest Centroid:</span>
              <span className="font-bold text-amber-400">Class {activeSample.nearest_centroid_class}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Centroid Dist ||h-μ||:</span>
              <span className="font-bold text-amber-300">{activeSample.centroid_distance.toFixed(3)}</span>
            </div>
          </div>

          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] uppercase text-slate-400">Neighborhood & Score</div>
            <div className="flex justify-between mt-1">
              <span className="text-slate-400">kNN Dist (k=5):</span>
              <span className="font-bold text-cyan-400">{activeSample.knn_distance.toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">{currentMethod} Score:</span>
              <span className="font-bold text-rose-400">
                {activeSample.msp_score.toFixed(3)}
                {threshold !== undefined && (
                  <span className="text-slate-500 font-normal text-[10px] ml-1">
                    (τ = {threshold.toFixed(2)})
                  </span>
                )}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
