"use client";

import React, { useState } from "react";
import { SpatialDetectionSamplePayload } from "../types";

interface DetectionVisualizerProps {
  samples: SpatialDetectionSamplePayload[];
}

export const DetectionVisualizer: React.FC<DetectionVisualizerProps> = ({
  samples,
}) => {
  const [selectedSampleIdx, setSelectedSampleIdx] = useState<number>(0);
  const [displayMode, setDisplayMode] = useState<"both" | "gt" | "pred">("both");
  const [scoreThreshold, setScoreThreshold] = useState<number>(0.1);

  if (!samples || samples.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500 font-mono text-xs">
        No detection sample visualization data available.
      </div>
    );
  }

  const sample = samples[selectedSampleIdx] || samples[0];
  const imgData = sample.image; // [C, H, W]
  const cImg = imgData.length;
  const hImg = imgData[0].length;
  const wImg = imgData[0][0].length;

  const filteredPreds = sample.predicted_boxes.filter(
    (b) => b.confidence >= scoreThreshold
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl shadow-slate-950/40">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">📦</span>
          <h3 className="text-sm font-bold text-slate-100 font-mono tracking-tight">
            DETECTION VISUALIZER // BOUNDING BOX OVERLAYS
          </h3>
        </div>

        {/* Display Mode Toggles */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => setDisplayMode("both")}
              className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all ${
                displayMode === "both"
                  ? "bg-slate-800 text-amber-400"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Both
            </button>
            <button
              onClick={() => setDisplayMode("gt")}
              className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all ${
                displayMode === "gt"
                  ? "bg-slate-800 text-emerald-400"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Ground Truth
            </button>
            <button
              onClick={() => setDisplayMode("pred")}
              className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all ${
                displayMode === "pred"
                  ? "bg-slate-800 text-cyan-400"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Predictions
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Visual Box Canvas View */}
        <div className="lg:col-span-6 flex flex-col items-center">
          <div className="relative w-72 h-72 rounded-xl overflow-hidden border border-slate-800 bg-slate-950 shadow-inner">
            {/* Render RGB Pixels */}
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

            {/* Bounding Box Overlays (Percentage based) */}
            {(displayMode === "both" || displayMode === "gt") &&
              sample.ground_truth_boxes.map((gt, idx) => {
                const [xMin, yMin, xMax, yMax] = gt.box;
                return (
                  <div
                    key={`gt-${idx}`}
                    className="absolute border-2 border-emerald-500 bg-emerald-500/10 pointer-events-none transition-all"
                    style={{
                      left: `${xMin * 100}%`,
                      top: `${yMin * 100}%`,
                      width: `${(xMax - xMin) * 100}%`,
                      height: `${(yMax - yMin) * 100}%`,
                    }}
                  >
                    <span className="absolute -top-5 left-0 bg-emerald-950/90 text-emerald-400 border border-emerald-500/40 text-[9px] font-mono font-bold px-1 rounded">
                      GT: Class {gt.class_id}
                    </span>
                  </div>
                );
              })}

            {(displayMode === "both" || displayMode === "pred") &&
              filteredPreds.map((p, idx) => {
                const [xMin, yMin, xMax, yMax] = p.box;
                return (
                  <div
                    key={`pred-${idx}`}
                    className="absolute border-2 border-cyan-400 bg-cyan-400/15 pointer-events-none transition-all"
                    style={{
                      left: `${xMin * 100}%`,
                      top: `${yMin * 100}%`,
                      width: `${(xMax - xMin) * 100}%`,
                      height: `${(yMax - yMin) * 100}%`,
                    }}
                  >
                    <span className="absolute -bottom-5 right-0 bg-cyan-950/90 text-cyan-300 border border-cyan-400/40 text-[9px] font-mono font-bold px-1 rounded">
                      Pred {p.class_id} ({Math.round(p.confidence * 100)}%)
                    </span>
                  </div>
                );
              })}
          </div>

          {/* Sample Selector Buttons */}
          <div className="flex items-center gap-2 mt-4">
            <span className="text-[11px] font-mono text-slate-400">Sample:</span>
            {samples.map((s, idx) => (
              <button
                key={s.sample_id}
                onClick={() => setSelectedSampleIdx(idx)}
                className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-all ${
                  selectedSampleIdx === idx
                    ? "bg-amber-600 text-white shadow-md shadow-amber-600/30"
                    : "bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800"
                }`}
              >
                #{idx + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Box Inspection & Legend List */}
        <div className="lg:col-span-6 space-y-4">
          {/* Confidence Slider */}
          <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-mono text-slate-400 text-[11px]">
                CONFIDENCE THRESHOLD
              </span>
              <span className="font-mono font-bold text-cyan-400">
                {Math.round(scoreThreshold * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="0.9"
              step="0.05"
              value={scoreThreshold}
              onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
          </div>

          {/* Ground Truth Objects Card */}
          <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl">
            <div className="text-[11px] font-mono font-bold text-emerald-400 mb-2 flex items-center justify-between">
              <span>GROUND TRUTH TARGETS ({sample.ground_truth_boxes.length})</span>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            </div>
            {sample.ground_truth_boxes.length === 0 ? (
              <div className="text-xs text-slate-500 italic">No target objects (Background-only image).</div>
            ) : (
              <div className="space-y-1.5">
                {sample.ground_truth_boxes.map((gt, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs bg-slate-900/80 px-2.5 py-1.5 rounded border border-slate-800/80"
                  >
                    <span className="font-semibold text-slate-200">
                      Class {gt.class_id} ({gt.class_name || `Object ${gt.class_id}`})
                    </span>
                    <span className="font-mono text-[10px] text-slate-400">
                      [{gt.box.map((v) => v.toFixed(2)).join(", ")}]
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Predicted Detections Card */}
          <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-xl">
            <div className="text-[11px] font-mono font-bold text-cyan-400 mb-2 flex items-center justify-between">
              <span>ACTIVE PREDICTIONS ({filteredPreds.length})</span>
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            </div>
            {filteredPreds.length === 0 ? (
              <div className="text-xs text-slate-500 italic">
                No predictions above confidence threshold.
              </div>
            ) : (
              <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                {filteredPreds.map((p, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs bg-slate-900/80 px-2.5 py-1.5 rounded border border-slate-800/80"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-cyan-300">
                        Class {p.class_id}
                      </span>
                      <span className="bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-[9px] font-mono px-1 rounded">
                        {Math.round(p.confidence * 100)}%
                      </span>
                    </div>
                    <span className="font-mono text-[10px] text-slate-400">
                      [{p.box.map((v) => v.toFixed(2)).join(", ")}]
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
