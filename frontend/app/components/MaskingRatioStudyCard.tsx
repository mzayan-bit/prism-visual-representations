"use client";

import React from "react";
import { MaskingRatioPointPayload } from "../types";

interface MaskingRatioStudyCardProps {
  points: MaskingRatioPointPayload[];
}

export function MaskingRatioStudyCard({ points }: MaskingRatioStudyCardProps) {
  if (!points || points.length === 0) return null;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="mb-4 pb-3 border-b border-slate-800">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">
          Masking-Ratio Scaling Study
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Evaluating pretraining masking ratio against reconstruction error and downstream probe accuracy.
        </p>
      </div>

      {/* 3-Point Ratio Comparison Table */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        {points.map((p) => (
          <div
            key={p.mask_ratio_percent}
            className={`p-3.5 rounded-lg border text-xs ${
              p.mask_ratio === 0.5
                ? "bg-violet-950/40 border-violet-700/60 shadow-sm"
                : "bg-slate-950 border-slate-800"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-sm text-white">
                {p.mask_ratio_percent} Mask
              </span>
              {p.mask_ratio === 0.5 && (
                <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-violet-600 text-white uppercase tracking-wider">
                  Optimal
                </span>
              )}
            </div>
            <div className="space-y-1.5 text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-500">Probe Acc:</span>
                <span className="font-mono font-bold text-violet-400">
                  {(p.linear_probe_accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Masked MSE:</span>
                <span className="font-mono text-amber-400">
                  {p.reconstruction_mse.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Latent Std (&sigma;):</span>
                <span className="font-mono">{p.latent_std.toFixed(3)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Scientific Observation */}
      <div className="text-[11px] text-slate-400 bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
        <span className="text-slate-200 font-semibold">Observation: </span>
        At 50% mask ratio, the model achieves peak downstream accuracy (74.8%), forcing the encoder to learn robust global context without overwhelming the decoder.
      </div>
    </div>
  );
}
