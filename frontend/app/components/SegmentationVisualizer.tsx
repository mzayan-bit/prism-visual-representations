"use client";

import React, { useState } from "react";
import { SpatialSegmentationSamplePayload } from "../types";

interface SegmentationVisualizerProps {
  samples: SpatialSegmentationSamplePayload[];
}

const CLASS_COLORS: string[] = [
  "#1e293b", // Class 0: Background (Slate-800)
  "#10b981", // Class 1: Emerald-500
  "#06b6d4", // Class 2: Cyan-500
  "#f59e0b", // Class 3: Amber-500
  "#8b5cf6", // Class 4: Violet-500
];

const CLASS_NAMES: string[] = [
  "Background",
  "Object Alpha",
  "Object Beta",
  "Object Gamma",
  "Object Delta",
];

export const SegmentationVisualizer: React.FC<SegmentationVisualizerProps> = ({
  samples,
}) => {
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number>(0);

  if (!samples || samples.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500 font-mono text-xs">
        No segmentation sample visualization data available.
      </div>
    );
  }

  const sample = samples[selectedSampleIdx] || samples[0];
  const imgData = sample.image; // [C, H, W]
  const cImg = imgData.length;
  const hImg = sample.ground_truth_mask.length;
  const wImg = sample.ground_truth_mask[0].length;
  const gtMask = sample.ground_truth_mask;
  const predMask = sample.predicted_mask;

  // Compute sample-level pixel accuracy
  let correctPixels = 0;
  const totalPixels = hImg * wImg;
  for (let r = 0; r < hImg; r++) {
    for (let c = 0; c < wImg; c++) {
      if (gtMask[r][c] === predMask[r][c]) {
        correctPixels++;
      }
    }
  }
  const sampleAccuracy = totalPixels > 0 ? (correctPixels / totalPixels) * 100 : 0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl shadow-slate-950/40">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">🎨</span>
          <h3 className="text-sm font-bold text-slate-100 font-mono tracking-tight">
            SEGMENTATION VISUALIZER // DENSE PREDICTION & ERROR MAP
          </h3>
        </div>

        {/* Sample Accuracy Badge */}
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-slate-400">Sample Accuracy:</span>
          <span
            className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
              sampleAccuracy >= 80
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                : "bg-amber-500/10 text-amber-400 border border-amber-500/30"
            }`}
          >
            {sampleAccuracy.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* 4-way Multi-view Grid: Input, Ground Truth, Predicted, Error Map */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
        {/* View 1: Input RGB Image */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col items-center">
          <span className="text-[10px] font-mono font-bold text-slate-400 mb-2">
            1. INPUT IMAGE
          </span>
          <div className="w-28 h-28 sm:w-32 sm:h-32 border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow">
            <svg
              viewBox={`0 0 ${wImg} ${hImg}`}
              className="w-full h-full"
              style={{ imageRendering: "pixelated" }}
            >
              {Array.from({ length: hImg }).map((_, r) =>
                Array.from({ length: wImg }).map((_, c) => {
                  const red = Math.round(
                    Math.max(0, Math.min(255, (imgData[0][r][c] || 0) * 255))
                  );
                  const green = Math.round(
                    Math.max(
                      0,
                      Math.min(255, ((cImg > 1 ? imgData[1][r][c] : imgData[0][r][c]) || 0) * 255)
                    )
                  );
                  const blue = Math.round(
                    Math.max(
                      0,
                      Math.min(255, ((cImg > 2 ? imgData[2][r][c] : imgData[0][r][c]) || 0) * 255)
                    )
                  );
                  return (
                    <rect
                      key={`${r}-${c}`}
                      x={c}
                      y={r}
                      width={1}
                      height={1}
                      fill={`rgb(${red}, ${green}, ${blue})`}
                    />
                  );
                })
              )}
            </svg>
          </div>
        </div>

        {/* View 2: Ground Truth Mask */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col items-center">
          <span className="text-[10px] font-mono font-bold text-emerald-400 mb-2">
            2. GROUND TRUTH
          </span>
          <div className="w-28 h-28 sm:w-32 sm:h-32 border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow">
            <svg
              viewBox={`0 0 ${wImg} ${hImg}`}
              className="w-full h-full"
              style={{ imageRendering: "pixelated" }}
            >
              {gtMask.map((row, r) =>
                row.map((clsId, c) => (
                  <rect
                    key={`${r}-${c}`}
                    x={c}
                    y={r}
                    width={1}
                    height={1}
                    fill={CLASS_COLORS[clsId % CLASS_COLORS.length]}
                  />
                ))
              )}
            </svg>
          </div>
        </div>

        {/* View 3: Predicted Segmentation Mask */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col items-center">
          <span className="text-[10px] font-mono font-bold text-cyan-400 mb-2">
            3. PREDICTION
          </span>
          <div className="w-28 h-28 sm:w-32 sm:h-32 border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow">
            <svg
              viewBox={`0 0 ${wImg} ${hImg}`}
              className="w-full h-full"
              style={{ imageRendering: "pixelated" }}
            >
              {predMask.map((row, r) =>
                row.map((clsId, c) => (
                  <rect
                    key={`${r}-${c}`}
                    x={c}
                    y={r}
                    width={1}
                    height={1}
                    fill={CLASS_COLORS[clsId % CLASS_COLORS.length]}
                  />
                ))
              )}
            </svg>
          </div>
        </div>

        {/* View 4: Error Difference Map */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col items-center">
          <span className="text-[10px] font-mono font-bold text-rose-400 mb-2">
            4. ERROR MAP
          </span>
          <div className="w-28 h-28 sm:w-32 sm:h-32 border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow">
            <svg
              viewBox={`0 0 ${wImg} ${hImg}`}
              className="w-full h-full"
              style={{ imageRendering: "pixelated" }}
            >
              {gtMask.map((row, r) =>
                row.map((gtCls, c) => {
                  const isMatch = gtCls === predMask[r][c];
                  return (
                    <rect
                      key={`${r}-${c}`}
                      x={c}
                      y={r}
                      width={1}
                      height={1}
                      fill={isMatch ? "#0f172a" : "#ef4444"}
                    />
                  );
                })
              )}
            </svg>
          </div>
        </div>
      </div>

      {/* Class Legend & Sample Switcher Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-3 border-t border-slate-800">
        {/* Class Palette Legend */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-mono text-slate-400 mr-1">Classes:</span>
          {Array.from({ length: sample.num_classes }).map((_, idx) => (
            <div
              key={idx}
              className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 px-2 py-1 rounded text-xs"
            >
              <span
                className="w-2.5 h-2.5 rounded-sm"
                style={{ backgroundColor: CLASS_COLORS[idx % CLASS_COLORS.length] }}
              ></span>
              <span className="text-slate-300 font-mono text-[11px]">
                {CLASS_NAMES[idx] || `Class ${idx}`}
              </span>
            </div>
          ))}
        </div>

        {/* Sample Selector */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span className="text-[11px] font-mono text-slate-400">Sample:</span>
          {samples.map((s, idx) => (
            <button
              key={s.sample_id}
              onClick={() => setSelectedSampleIdx(idx)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-all ${
                selectedSampleIdx === idx
                  ? "bg-teal-600 text-white shadow-md shadow-teal-600/30"
                  : "bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800"
              }`}
            >
              #{idx + 1}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
