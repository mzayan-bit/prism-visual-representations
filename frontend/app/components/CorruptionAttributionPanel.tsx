"use client";

import React, { useState } from "react";
import { AttributionHeatmapCard } from "./AttributionHeatmapCard";
import { AttributionMethod, ExplainabilitySamplePayload } from "../types";

interface CorruptionAttributionPanelProps {
  sample: ExplainabilitySamplePayload;
  selectedArch: string;
}

export const CorruptionAttributionPanel: React.FC<CorruptionAttributionPanelProps> = ({
  sample,
  selectedArch,
}) => {
  const [selectedMethod, setSelectedMethod] = useState<AttributionMethod>("input_gradient");

  const cleanAttribution = sample.attributions[selectedArch]?.[selectedMethod];
  const corruptedAttribution = sample.corrupted_attributions[selectedArch]?.[selectedMethod];
  const driftSummary = sample.drift_summaries[selectedArch]?.[selectedMethod];

  const cleanPred = sample.predictions[selectedArch];
  const corrPred = sample.corrupted_predictions[selectedArch];

  const availableMethods: AttributionMethod[] =
    selectedArch === "vit"
      ? ["input_gradient", "gradient_x_input", "occlusion_sensitivity", "vit_attention"]
      : ["input_gradient", "gradient_x_input", "occlusion_sensitivity", "grad_cam"];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl mb-8">
      {/* Header & Method Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800/80">
        <div>
          <h3 className="text-sm font-black text-white font-mono flex items-center gap-2">
            <span>🌪️</span> CORRUPTION STABILITY & ATTRIBUTION DRIFT
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Measuring how spatial evidence shifts when input undergoes distribution corruption (Gaussian Noise σ=0.15).
          </p>
        </div>

        {/* Method Selector Pills */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono">
          {availableMethods.map((m) => (
            <button
              key={m}
              onClick={() => setSelectedMethod(m)}
              className={`px-3 py-1.5 rounded-lg font-bold transition-all capitalize ${
                selectedMethod === m
                  ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {m.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Side-by-Side Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Clean Condition */}
        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
            <span className="text-xs font-bold font-mono text-emerald-400">CLEAN SOURCE CONDITION</span>
            {cleanPred && (
              <span className="text-[11px] font-mono text-slate-300">
                Pred: <strong className="text-white">{cleanPred.predicted_name}</strong> ({Math.round(cleanPred.confidence * 100)}%)
              </span>
            )}
          </div>
          <AttributionHeatmapCard
            title="Clean Attribution"
            result={cleanAttribution}
            imageTensor={sample.image_tensor}
          />
        </div>

        {/* Corrupted Condition */}
        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
            <span className="text-xs font-bold font-mono text-rose-400">CORRUPTED CONDITION (NOISE)</span>
            {corrPred && (
              <span className="text-[11px] font-mono text-slate-300">
                Pred: <strong className="text-white">{corrPred.predicted_name}</strong> ({Math.round(corrPred.confidence * 100)}%)
              </span>
            )}
          </div>
          <AttributionHeatmapCard
            title="Corrupted Attribution"
            result={corruptedAttribution}
            imageTensor={sample.corrupted_image_tensor || sample.image_tensor}
          />
        </div>
      </div>

      {/* Quantitative Drift Dashboard */}
      {driftSummary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t border-slate-800/80 text-xs font-mono">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block">MAP COSINE SIMILARITY</span>
            <span className="text-sm font-bold text-cyan-400">
              {driftSummary.attribution_cosine_similarity.toFixed(3)}
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">Clean vs Corrupted</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block">TOP 10% MASK RETENTION</span>
            <span className="text-sm font-bold text-emerald-400">
              {(driftSummary.top_10_percent_mask_overlap * 100).toFixed(1)}%
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">Jaccard Overlap</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block">CENTER OF MASS SHIFT</span>
            <span className="text-sm font-bold text-amber-400">
              {driftSummary.center_of_mass_displacement.toFixed(2)} px
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">Euclidean Displacement</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block">PREDICTION STATUS</span>
            <span className={`text-xs font-bold block ${driftSummary.prediction_preserved ? "text-emerald-400" : "text-rose-400"}`}>
              {driftSummary.prediction_preserved ? "PRESERVED" : "FLIPPED"}
            </span>
            <span className="text-[10px] text-slate-500 block mt-1">
              {driftSummary.clean_predicted_class} → {driftSummary.corrupted_predicted_class}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
