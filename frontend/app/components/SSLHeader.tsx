"use client";

import React from "react";
import { SSLMetadataPayload } from "../types";

interface SSLHeaderProps {
  metadata: SSLMetadataPayload;
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  selectedTemp: number;
  onSelectTemp: (temp: number) => void;
}

export function SSLHeader({
  metadata,
  selectedArch,
  onSelectArch,
  selectedTemp,
  onSelectTemp,
}: SSLHeaderProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6 shadow-sm">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
              PHASE 18
            </span>
            <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-indigo-950/80 text-indigo-400 border border-indigo-800/60">
              {metadata.method}
            </span>
            <span className="text-xs text-slate-500 font-mono">
              {metadata.experiment_id}
            </span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            Self-Supervised Representation Learning Laboratory
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Study visual representations learned without class supervision via instance-level
            contrastive learning (SimCLR), evaluating alignment, dimensional collapse, and downstream linear transfer.
          </p>
        </div>

        {/* Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Architecture Selector */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1">
            {metadata.architectures.map((arch) => (
              <button
                key={arch}
                onClick={() => onSelectArch(arch)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium uppercase tracking-wider transition-all ${
                  selectedArch.toLowerCase() === arch.toLowerCase()
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {arch}
              </button>
            ))}
          </div>

          {/* Temperature Selector */}
          <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
            <span className="text-slate-400 font-medium">Temperature &tau;:</span>
            <div className="flex items-center gap-1">
              {metadata.temperatures.map((t) => (
                <button
                  key={t}
                  onClick={() => onSelectTemp(t)}
                  className={`px-2 py-0.5 rounded text-xs font-mono transition-all ${
                    selectedTemp === t
                      ? "bg-emerald-600 text-white font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
