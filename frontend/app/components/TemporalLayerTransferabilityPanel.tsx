"use client";

import React, { useState } from "react";
import { TemporalLayerProfileRecord } from "../types";

interface TemporalLayerTransferabilityPanelProps {
  layerProfiles: Record<string, TemporalLayerProfileRecord[]>;
}

export const TemporalLayerTransferabilityPanel: React.FC<
  TemporalLayerTransferabilityPanelProps
> = ({ layerProfiles }) => {
  const architectures = Object.keys(layerProfiles);
  const [selectedArch, setSelectedArch] = useState<string>(
    architectures[0] || "resnet"
  );

  const activeProfiles = layerProfiles[selectedArch] || [];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-sm font-bold text-slate-100">
            Layer-Wise Temporal Transferability
          </h2>
          <p className="text-xs text-slate-400">
            Which encoder depth provides the most reusable signal for temporal aggregation?
          </p>
        </div>

        {/* Architecture Selector */}
        <div className="flex gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          {architectures.map((arch) => (
            <button
              key={arch}
              onClick={() => setSelectedArch(arch)}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-colors ${
                selectedArch === arch
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {arch.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Visual Depth vs Accuracy Chart */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        {activeProfiles.map((p) => (
          <div
            key={p.layer_name}
            className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col justify-between h-36"
          >
            <div>
              <div className="flex items-center justify-between text-xs font-mono font-bold text-slate-200">
                <span>{p.layer_name}</span>
                <span className="text-cyan-400 text-[10px]">
                  D={p.feature_dim}
                </span>
              </div>
              <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                Depth: {(p.depth_fraction * 100).toFixed(0)}%
              </div>
            </div>

            <div className="space-y-1.5 mt-auto">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <span className="text-slate-400">Video Acc:</span>
                <span className="font-bold text-emerald-400">
                  {(p.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-emerald-400 h-full rounded-full"
                  style={{ width: `${p.accuracy * 100}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[10px] font-mono pt-1">
                <span className="text-slate-400">Consistency:</span>
                <span className="font-bold text-amber-300">
                  {p.consistency.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
