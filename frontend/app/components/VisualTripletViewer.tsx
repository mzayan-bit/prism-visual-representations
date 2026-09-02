"use client";

import React, { useState } from "react";
import { VisualTripletSamplePayload } from "../types";

interface VisualTripletViewerProps {
  triplets: VisualTripletSamplePayload[];
  selectedMethod: string;
}

export function VisualTripletViewer({
  triplets,
  selectedMethod,
}: VisualTripletViewerProps) {
  const [selectedSampleIndex, setSelectedSampleIndex] = useState<number>(0);

  if (!triplets || triplets.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-400">
        No visual reconstruction samples available for the selected configuration.
      </div>
    );
  }

  const sample = triplets[Math.min(selectedSampleIndex, triplets.length - 1)];

  // Helper to render 3-channel 8x8 image onto 64px square CSS grid
  const renderImageGrid = (
    img: number[][][],
    title: string,
    badgeColor: string,
    isMaskedView: boolean = false
  ) => {
    const c = img.length;
    const h = img[0].length;
    const w = img[0][0].length;

    return (
      <div className="flex flex-col items-center bg-slate-950 p-3 rounded-lg border border-slate-800/80">
        <span
          className={`text-[11px] font-semibold uppercase tracking-wider mb-2 px-2 py-0.5 rounded border ${badgeColor}`}
        >
          {title}
        </span>
        <div
          className="grid gap-[1px] bg-slate-900 p-1 rounded border border-slate-800 relative shadow-inner"
          style={{
            gridTemplateColumns: `repeat(${w}, minmax(0, 1fr))`,
            width: "140px",
            height: "140px",
          }}
        >
          {Array.from({ length: h }).map((_, r) =>
            Array.from({ length: w }).map((_, col) => {
              const red = Math.round((img[0]?.[r]?.[col] ?? 0) * 255);
              const green = Math.round((img[1]?.[r]?.[col] ?? 0) * 255);
              const blue = Math.round((img[2]?.[r]?.[col] ?? 0) * 255);

              // Check if patch is masked (4x4 patch in 8x8 image)
              const patchIdx = Math.floor(r / 4) * 2 + Math.floor(col / 4);
              const isMaskedPatch =
                isMaskedView && sample.masked_patch_indices.includes(patchIdx);

              return (
                <div
                  key={`${r}-${col}`}
                  className={`w-full h-full transition-all ${
                    isMaskedPatch ? "border border-violet-500/40" : ""
                  }`}
                  style={{
                    backgroundColor: `rgb(${red}, ${green}, ${blue})`,
                  }}
                  title={`[${r},${col}] RGB: (${red}, ${green}, ${blue})`}
                />
              );
            })
          )}
        </div>
        <span className="text-[10px] text-slate-500 font-mono mt-1.5">
          {h}x{w}x{c} px
        </span>
      </div>
    );
  };

  // Helper to render 2D spatial error map [H x W]
  const renderErrorHeatmap = (errorMap: number[][]) => {
    const h = errorMap.length;
    const w = errorMap[0].length;

    return (
      <div className="flex flex-col items-center bg-slate-950 p-3 rounded-lg border border-slate-800/80">
        <span className="text-[11px] font-semibold uppercase tracking-wider mb-2 px-2 py-0.5 rounded border bg-rose-950/80 text-rose-300 border-rose-800/60">
          Spatial Error Map
        </span>
        <div
          className="grid gap-[1px] bg-slate-900 p-1 rounded border border-slate-800 shadow-inner"
          style={{
            gridTemplateColumns: `repeat(${w}, minmax(0, 1fr))`,
            width: "140px",
            height: "140px",
          }}
        >
          {Array.from({ length: h }).map((_, r) =>
            Array.from({ length: w }).map((_, col) => {
              const err = errorMap[r]?.[col] ?? 0;
              // Heatmap scale: 0 -> dark slate, 0.05 -> amber, 0.15+ -> bright rose/red
              const intensity = Math.min(1.0, err / 0.12);
              const red = Math.round(30 + intensity * 225);
              const green = Math.round(40 + (1 - intensity) * 80);
              const blue = Math.round(60 + (1 - intensity) * 120);

              return (
                <div
                  key={`err-${r}-${col}`}
                  className="w-full h-full transition-all"
                  style={{
                    backgroundColor: `rgb(${red}, ${green}, ${blue})`,
                  }}
                  title={`Error [${r},${col}]: ${err.toFixed(4)}`}
                />
              );
            })
          )}
        </div>
        <div className="flex items-center gap-1 mt-1.5 text-[10px] text-slate-500">
          <span className="text-slate-400">0.0</span>
          <div className="w-12 h-1.5 rounded bg-gradient-to-r from-slate-700 via-amber-600 to-rose-500" />
          <span className="text-rose-400">High</span>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>Visual Triplet & Spatial Reconstruction Fidelity</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Inspect clean target, corrupted/masked model inputs, decoded reconstruction, and pixel error distributions.
          </p>
        </div>

        {/* Sample Selector Chips */}
        <div className="flex items-center gap-1.5 overflow-x-auto">
          {triplets.map((t, idx) => (
            <button
              key={t.sample_id}
              onClick={() => setSelectedSampleIndex(idx)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 ${
                selectedSampleIndex === idx
                  ? "bg-violet-600 text-white font-semibold shadow-sm"
                  : "bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800"
              }`}
            >
              <span>{t.class_name}</span>
              {t.failure_category && (
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Triplet + Heatmap Quad Layout */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        {renderImageGrid(
          sample.original_image,
          "1. Clean Target (y)",
          "bg-emerald-950/80 text-emerald-300 border-emerald-800/60"
        )}
        {renderImageGrid(
          sample.corrupted_or_masked_image,
          selectedMethod === "masked_patch_reconstruction"
            ? "2. Masked Input (x_m)"
            : "2. Corrupted Input (x~)",
          "bg-indigo-950/80 text-indigo-300 border-indigo-800/60",
          selectedMethod === "masked_patch_reconstruction"
        )}
        {renderImageGrid(
          sample.reconstructed_image,
          "3. Reconstruction (y^)",
          "bg-cyan-950/80 text-cyan-300 border-cyan-800/60"
        )}
        {renderErrorHeatmap(sample.error_map)}
      </div>

      {/* Sample Metrics Summary Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950 px-4 py-2.5 rounded-lg border border-slate-800/80 text-xs">
        <div className="flex items-center gap-4">
          <div>
            <span className="text-slate-500 mr-1.5">Sample ID:</span>
            <span className="text-slate-300 font-mono font-semibold">
              {sample.sample_id}
            </span>
          </div>
          <div>
            <span className="text-slate-500 mr-1.5">Class Label:</span>
            <span className="text-slate-300 capitalize">{sample.class_name}</span>
          </div>
          {sample.masked_patch_indices.length > 0 && (
            <div>
              <span className="text-slate-500 mr-1.5">Masked Patches:</span>
              <span className="text-violet-300 font-mono font-semibold">
                [{sample.masked_patch_indices.join(", ")}]
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div>
            <span className="text-slate-500 mr-1.5">Sample MSE:</span>
            <span
              className={`font-mono font-bold ${
                sample.sample_mse > 0.05 ? "text-amber-400" : "text-emerald-400"
              }`}
            >
              {sample.sample_mse.toFixed(4)}
            </span>
          </div>
          {sample.failure_category && (
            <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-950/90 text-rose-300 border border-rose-800/80">
              {sample.failure_category.replace(/_/g, " ").toUpperCase()}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
