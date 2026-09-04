"use client";

import React, { useState } from "react";
import { SpatialLayerTransferabilityRecord, SpatialTaskType } from "../types";

interface SpatialLayerTransferabilityPanelProps {
  layerRecords: SpatialLayerTransferabilityRecord[];
  activeTask: SpatialTaskType;
}

export const SpatialLayerTransferabilityPanel: React.FC<
  SpatialLayerTransferabilityPanelProps
> = ({ layerRecords, activeTask }) => {
  const [selectedMetric, setSelectedMetric] = useState<"detection" | "segmentation">(
    activeTask === "object_detection" ? "detection" : "segmentation"
  );

  if (!layerRecords || layerRecords.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500 font-mono text-xs">
        No layer transferability data available.
      </div>
    );
  }

  const maxIoU = Math.max(
    ...layerRecords.map((r) =>
      selectedMetric === "detection" ? r.detection_mean_iou : r.segmentation_mean_iou
    ),
    0.01
  );

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl shadow-slate-950/40">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">🪜</span>
          <h3 className="text-sm font-bold text-slate-100 font-mono tracking-tight">
            LAYER-WISE SPATIAL TRANSFERABILITY ACROSS DEPTH
          </h3>
        </div>

        {/* Metric Selector */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setSelectedMetric("detection")}
            className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all ${
              selectedMetric === "detection"
                ? "bg-slate-800 text-amber-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Detection mIoU
          </button>
          <button
            onClick={() => setSelectedMetric("segmentation")}
            className={`px-2.5 py-1 text-[11px] font-bold rounded transition-all ${
              selectedMetric === "segmentation"
                ? "bg-slate-800 text-teal-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Segmentation mIoU
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {layerRecords.map((rec) => {
          const val =
            selectedMetric === "detection"
              ? rec.detection_mean_iou
              : rec.segmentation_mean_iou;
          const pct = Math.round((val / maxIoU) * 100);

          return (
            <div
              key={rec.layer}
              className="bg-slate-950 border border-slate-800/80 rounded-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-slate-700 transition-colors"
            >
              {/* Layer Title & Meta */}
              <div className="sm:w-1/3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-slate-200">
                    {rec.layer}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    (Depth #{rec.depth_index})
                  </span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                  Resolution:{" "}
                  <strong className="text-slate-300">{rec.feature_resolution}</strong>{" "}
                  | Channels:{" "}
                  <strong className="text-slate-300">{rec.feature_channels}</strong>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="sm:w-1/2 flex items-center gap-3">
                <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-800">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-500 ${
                      selectedMetric === "detection"
                        ? "bg-gradient-to-r from-amber-600 to-amber-400"
                        : "bg-gradient-to-r from-teal-600 to-teal-400"
                    }`}
                    style={{ width: `${Math.max(5, Math.min(100, pct))}%` }}
                  ></div>
                </div>
                <span className="font-mono text-xs font-bold text-slate-100 min-w-[50px] text-right">
                  {(val * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
