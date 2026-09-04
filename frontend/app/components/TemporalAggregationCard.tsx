"use client";

import React from "react";
import {
  TemporalAggregationType,
  TemporalVideoSamplePayload,
} from "../types";

interface TemporalAggregationCardProps {
  aggregator: TemporalAggregationType;
  sample: TemporalVideoSamplePayload | undefined;
  activeFrameIndex: number;
}

export const TemporalAggregationCard: React.FC<TemporalAggregationCardProps> = ({
  aggregator,
  sample,
  activeFrameIndex,
}) => {
  const attentionWeights = sample?.attention_weights || [];
  const hiddenNorms = sample?.hidden_norms || [];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">🧮</span>
          <div>
            <h2 className="text-sm font-bold text-slate-100">
              Temporal Aggregation Dynamics
            </h2>
            <p className="text-xs text-slate-400">
              Internal state allocation across the sequence length
            </p>
          </div>
        </div>
        <span className="text-xs font-mono font-bold text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/30">
          {aggregator.replace(/_/g, " ").toUpperCase()}
        </span>
      </div>

      {/* Aggregator-Specific Visualizations */}
      {aggregator === "learned_temporal_pooling" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Learned Attention Weights (α_t)</span>
            <span className="text-slate-500 font-mono text-[11px]">
              Σ α_t = 1.000 (Softmax)
            </span>
          </div>

          <div className="grid grid-cols-4 gap-2">
            {attentionWeights.map((w, idx) => {
              const isActive = idx === activeFrameIndex;
              return (
                <div
                  key={idx}
                  className={`p-2.5 rounded-xl border flex flex-col justify-between h-28 transition-all ${
                    isActive
                      ? "bg-slate-950 border-amber-500 shadow-md ring-1 ring-amber-500/30"
                      : "bg-slate-950 border-slate-800"
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] font-mono">
                    <span className={isActive ? "text-amber-400 font-bold" : "text-slate-400"}>
                      t = {idx}
                    </span>
                    <span className="font-bold text-slate-200">
                      {(w * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Weight Fill Bar */}
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-auto mb-1">
                    <div
                      className="bg-amber-400 h-full rounded-full transition-all duration-300"
                      style={{ width: `${Math.max(4, w * 100)}%` }}
                    />
                  </div>

                  <div className="text-[9px] font-mono text-slate-500 text-right">
                    α = {w.toFixed(3)}
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-[11px] text-slate-400 bg-slate-950 p-2.5 rounded-xl border border-slate-800/80 leading-relaxed">
            <span className="text-amber-300 font-semibold font-mono">Note:</span> Softmax weights represent temporal feature aggregation salience rather than causal explanations.
          </p>
        </div>
      )}

      {aggregator === "simple_rnn" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Recurrent Hidden State Norms (||h_t||_2)</span>
            <span className="text-cyan-400 font-mono text-[11px]">
              h_0 = 0 • BPTT Active
            </span>
          </div>

          <div className="grid grid-cols-4 gap-2">
            {hiddenNorms.map((n, idx) => {
              const isActive = idx === activeFrameIndex;
              return (
                <div
                  key={idx}
                  className={`p-2.5 rounded-xl border flex flex-col justify-between h-28 transition-all ${
                    isActive
                      ? "bg-slate-950 border-cyan-500 shadow-md ring-1 ring-cyan-500/30"
                      : "bg-slate-950 border-slate-800"
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] font-mono">
                    <span className={isActive ? "text-cyan-400 font-bold" : "text-slate-400"}>
                      t = {idx}
                    </span>
                    <span className="font-bold text-slate-200">{n.toFixed(3)}</span>
                  </div>

                  {/* Norm Fill Bar */}
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-auto mb-1">
                    <div
                      className="bg-cyan-400 h-full rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(100, Math.max(4, n * 80))}%` }}
                    />
                  </div>

                  <div className="text-[9px] font-mono text-slate-500 text-right">
                    tanh(Wx + Wh + b)
                  </div>
                </div>
              );
            })}
          </div>

          <div className="text-[11px] font-mono text-slate-400 bg-slate-950 p-2.5 rounded-xl border border-slate-800/80">
            h_t = tanh(W_x x_t + W_h h_(t-1) + b)
          </div>
        </div>
      )}

      {(aggregator === "mean_pool" ||
        aggregator === "max_pool" ||
        aggregator === "last_frame") && (
        <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 space-y-2 text-xs">
          <div className="font-bold text-slate-200">
            {aggregator === "mean_pool" && "Uniform Average Pooling Across Time"}
            {aggregator === "max_pool" && "Feature-Wise Maximum Activation Across Time"}
            {aggregator === "last_frame" && "Terminal Frame Representation Selection"}
          </div>
          <p className="text-slate-400 text-[11px] leading-relaxed">
            {aggregator === "mean_pool" &&
              "Computes z = (1/T) Σ h_t. Provides a parameter-free, order-invariant baseline that distributes upstream gradient equally across all observations."}
            {aggregator === "max_pool" &&
              "Computes z_d = max_t h_(t, d). Routes upstream gradient strictly to the deterministic argmax timestep for each representation channel."}
            {aggregator === "last_frame" &&
              "Selects z = h_(T-1). Intentionally discards earlier temporal history to test if multi-frame aggregation outperforms a static snapshot."}
          </p>
        </div>
      )}
    </div>
  );
};
