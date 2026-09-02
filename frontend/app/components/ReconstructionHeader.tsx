"use client";

import React from "react";
import { ReconstructionMetadataPayload } from "../types";

interface ReconstructionHeaderProps {
  metadata: ReconstructionMetadataPayload;
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  selectedMethod: string;
  onSelectMethod: (method: string) => void;
  selectedMaskRatio: number;
  onSelectMaskRatio: (ratio: number) => void;
  selectedCorruption: string;
  onSelectCorruption: (corr: string) => void;
}

export function ReconstructionHeader({
  metadata,
  selectedArch,
  onSelectArch,
  selectedMethod,
  onSelectMethod,
  selectedMaskRatio,
  onSelectMaskRatio,
  selectedCorruption,
  onSelectCorruption,
}: ReconstructionHeaderProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6 shadow-sm">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-violet-950/80 text-violet-400 border border-violet-800/60">
              PHASE 19
            </span>
            <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
              {selectedMethod.replace(/_/g, " ").toUpperCase()}
            </span>
            <span className="text-xs text-slate-500 font-mono">
              {metadata.experiment_id}
            </span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            Reconstruction & Masked Representation Learning Laboratory
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Analyze visual representations emerging from masked patch reconstruction and denoising autoencoding,
            comparing spatial reconstruction fidelity against downstream semantic utility and probe performance.
          </p>
        </div>

        {/* Control Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Method Selector */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1">
            <button
              onClick={() => onSelectMethod("masked_patch_reconstruction")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                selectedMethod === "masked_patch_reconstruction"
                  ? "bg-violet-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Masked Patch (MIM)
            </button>
            <button
              onClick={() => onSelectMethod("denoising_autoencoder")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                selectedMethod === "denoising_autoencoder"
                  ? "bg-cyan-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Denoising (DAE)
            </button>
          </div>

          {/* Architecture Selector */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1">
            {metadata.architectures.map((arch) => (
              <button
                key={arch}
                onClick={() => onSelectArch(arch)}
                className={`px-2.5 py-1.5 rounded-md text-xs font-medium uppercase tracking-wider transition-all ${
                  selectedArch.toLowerCase() === arch.toLowerCase()
                    ? "bg-slate-700 text-white shadow-sm font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {arch}
              </button>
            ))}
          </div>

          {/* Mask Ratio Selector (if masked patch) */}
          {selectedMethod === "masked_patch_reconstruction" && (
            <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
              <span className="text-slate-400 font-medium">Mask Ratio:</span>
              <div className="flex items-center gap-1">
                {metadata.mask_ratios.map((r) => (
                  <button
                    key={r}
                    onClick={() => onSelectMaskRatio(r)}
                    className={`px-2 py-0.5 rounded text-xs font-mono transition-all ${
                      selectedMaskRatio === r
                        ? "bg-violet-600 text-white font-semibold"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {Math.round(r * 100)}%
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Corruption Type Selector (if denoising) */}
          {selectedMethod === "denoising_autoencoder" && (
            <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
              <span className="text-slate-400 font-medium">Corruption:</span>
              <div className="flex items-center gap-1">
                {["gaussian_noise", "occlusion", "resolution"].map((corr) => (
                  <button
                    key={corr}
                    onClick={() => onSelectCorruption(corr)}
                    className={`px-2 py-0.5 rounded text-xs capitalize transition-all ${
                      selectedCorruption === corr
                        ? "bg-cyan-600 text-white font-semibold"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {corr.replace(/_/g, " ")}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
