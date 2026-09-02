"use client";

import React from "react";
import { SSLLayerProbePointPayload } from "../types";

interface SSLLayerProbePanelProps {
  probes: SSLLayerProbePointPayload[];
}

export function SSLLayerProbePanel({ probes }: SSLLayerProbePanelProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Layer-Wise Transferability Probes
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Linear probing accuracy across encoder layer depths for SimCLR vs Supervised representations.
          </p>
        </div>
      </div>

      <div className="space-y-3">
        {probes.map((probe) => {
          const sslPct = probe.ssl_accuracy * 100;
          const supPct = probe.supervised_accuracy * 100;

          return (
            <div
              key={probe.layer_id}
              className="bg-slate-950 border border-slate-800/80 rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold text-white">
                    {probe.layer_id}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                    dim: {probe.representation_dim}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs font-mono">
                  <span className="text-indigo-400 font-medium">SSL: {sslPct.toFixed(1)}%</span>
                  <span className="text-emerald-400 font-medium">Sup: {supPct.toFixed(1)}%</span>
                </div>
              </div>

              {/* Progress Bars */}
              <div className="space-y-1.5">
                {/* SimCLR */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400 w-10 font-mono">SimCLR</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                      style={{ width: `${sslPct}%` }}
                    />
                  </div>
                </div>

                {/* Supervised */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400 w-10 font-mono">Supervised</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                      style={{ width: `${supPct}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
