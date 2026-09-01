"use client";

import React from "react";
import { LayerTransferProbePayload } from "../types";

interface LayerTransferabilityPanelProps {
  probes: LayerTransferProbePayload[];
  architecture: string;
}

export function LayerTransferabilityPanel({
  probes,
  architecture,
}: LayerTransferabilityPanelProps) {
  if (!probes || probes.length === 0) {
    return (
      <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex items-center justify-center min-h-[260px]">
        <span className="text-xs text-slate-500 font-mono">
          No layer transferability probes available for {architecture.toUpperCase()}.
        </span>
      </div>
    );
  }

  // Find best performing probed layer
  const bestProbe = [...probes].sort((a, b) => b.val_accuracy - a.val_accuracy)[0];

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-full">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
            Layer Transferability Probes Across Depth
          </h3>
          <p className="text-xs text-slate-400">
            Linear classification accuracy achieved on representations extracted at each layer
          </p>
        </div>
        {bestProbe && (
          <div className="text-right">
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Peak Transfer Layer
            </div>
            <div className="text-xs font-bold text-cyan-300 font-mono capitalize">
              {bestProbe.layer_name.replace("_", " ")} ({(bestProbe.val_accuracy * 100).toFixed(1)}%)
            </div>
          </div>
        )}
      </div>

      {/* Layer Probe Rows */}
      <div className="space-y-3 flex-1 overflow-y-auto max-h-[300px] pr-1">
        {probes.map((probe, idx) => {
          const isBest = probe.layer_name === bestProbe?.layer_name;
          const valPct = (probe.val_accuracy * 100).toFixed(1);
          const trainPct = (probe.train_accuracy * 100).toFixed(1);

          return (
            <div
              key={probe.layer_name}
              className={`p-3 rounded-lg border transition-all ${
                isBest
                  ? "bg-slate-950/80 border-indigo-500/60 ring-1 ring-indigo-500/20"
                  : "bg-slate-950/50 border-slate-800/80 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-slate-800 text-[10px] font-mono text-slate-300 flex items-center justify-center">
                    {idx + 1}
                  </span>
                  <span className="text-xs font-bold text-white font-mono capitalize">
                    {probe.layer_name.replace("_", " ")}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    ({probe.representation_dim}-dim)
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-xs font-bold font-mono text-cyan-300">
                    Val Acc: {valPct}%
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    Train: {trainPct}%
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden border border-slate-800/80 mb-2">
                <div
                  style={{ width: `${valPct}%` }}
                  className={`h-full transition-all duration-500 ${
                    isBest ? "bg-cyan-400" : "bg-indigo-500"
                  }`}
                />
              </div>

              {/* Footer Meta */}
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                <span>Probe Params: {probe.probe_parameters_count.toLocaleString()}</span>
                <span>Best Epoch: {probe.best_epoch + 1}/{probe.epochs_trained}</span>
                <span>Time: {probe.duration_seconds.toFixed(2)}s</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
