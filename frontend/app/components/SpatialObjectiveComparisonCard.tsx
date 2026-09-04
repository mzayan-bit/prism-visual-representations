"use client";

import React from "react";
import {
  PretrainingObjectiveType,
  SpatialTaskType,
  SpatialTransferReportPayload,
} from "../types";

interface SpatialObjectiveComparisonCardProps {
  comparison: Record<PretrainingObjectiveType, SpatialTransferReportPayload | null>;
  taskType: SpatialTaskType;
}

const OBJECTIVE_CONFIG: Record<
  PretrainingObjectiveType,
  { name: string; icon: string; border: string; accent: string; badge: string }
> = {
  supervised: {
    name: "Supervised Classification",
    icon: "🏷️",
    border: "border-cyan-500/40",
    accent: "text-cyan-400",
    badge: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
  },
  simclr: {
    name: "SimCLR Contrastive SSL",
    icon: "🌌",
    border: "border-indigo-500/40",
    accent: "text-indigo-400",
    badge: "bg-indigo-500/10 text-indigo-400 border-indigo-500/30",
  },
  reconstruction: {
    name: "Reconstruction (MIM)",
    icon: "🧩",
    border: "border-violet-500/40",
    accent: "text-violet-400",
    badge: "bg-violet-500/10 text-violet-400 border-violet-500/30",
  },
  scratch: {
    name: "Random Scratch Baseline",
    icon: "🎲",
    border: "border-slate-700",
    accent: "text-slate-400",
    badge: "bg-slate-800 text-slate-300 border-slate-700",
  },
};

export const SpatialObjectiveComparisonCard: React.FC<
  SpatialObjectiveComparisonCardProps
> = ({ comparison, taskType }) => {
  const objectives: PretrainingObjectiveType[] = [
    "supervised",
    "simclr",
    "reconstruction",
    "scratch",
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 mb-6 shadow-xl shadow-slate-950/40">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">⚖️</span>
          <h3 className="text-sm font-bold text-slate-100 font-mono tracking-tight">
            CROSS-PRETRAINING OBJECTIVE TRANSFER COMPARISON
          </h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          Target Task:{" "}
          <strong className="text-slate-200">
            {taskType === "object_detection" ? "Detection" : "Segmentation"}
          </strong>
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {objectives.map((objKey) => {
          const cfg = OBJECTIVE_CONFIG[objKey];
          const report = comparison[objKey];

          const meanIoU =
            taskType === "object_detection"
              ? report?.detection_metrics?.mean_iou ?? 0
              : report?.segmentation_metrics?.mean_iou ?? 0;

          const secondaryMetric =
            taskType === "object_detection"
              ? {
                  label: "Precision @ 0.5",
                  val: `${Math.round(
                    (report?.detection_metrics?.precision ?? 0) * 100
                  )}%`,
                }
              : {
                  label: "Pixel Accuracy",
                  val: `${Math.round(
                    (report?.segmentation_metrics?.pixel_accuracy ?? 0) * 100
                  )}%`,
                };

          const cosineDrift = report?.spatial_representation_drift_cosine ?? 0;
          const trainableFraction = report?.trainable_fraction ?? 0;
          const featureRes = report?.feature_resolution ?? "N/A";

          return (
            <div
              key={objKey}
              className={`bg-slate-950 border ${cfg.border} rounded-xl p-4 flex flex-col justify-between transition-all hover:border-slate-600 shadow-md`}
            >
              <div>
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-base">{cfg.icon}</span>
                    <span className="text-xs font-bold text-slate-200">
                      {cfg.name}
                    </span>
                  </div>
                </div>

                {/* Main Metric: Mean IoU */}
                <div className="mb-4 bg-slate-900/90 p-3 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] font-mono text-slate-400 mb-1 flex items-center justify-between">
                    <span>DOWNSTREAM MEAN IOU</span>
                    <span className="font-mono text-xs font-bold text-amber-400">
                      {(meanIoU * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-amber-500 to-emerald-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(100, Math.max(0, meanIoU * 100))}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sub-Metrics Grid */}
                <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                  <div className="bg-slate-900/50 p-2 rounded border border-slate-800/50">
                    <span className="block text-[10px] font-mono text-slate-400">
                      {secondaryMetric.label}
                    </span>
                    <span className="font-mono font-bold text-slate-200">
                      {secondaryMetric.val}
                    </span>
                  </div>
                  <div className="bg-slate-900/50 p-2 rounded border border-slate-800/50">
                    <span className="block text-[10px] font-mono text-slate-400">
                      Feature Res
                    </span>
                    <span className="font-mono font-bold text-slate-200">
                      {featureRes}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                  <div className="bg-slate-900/50 p-2 rounded border border-slate-800/50">
                    <span className="block text-[10px] font-mono text-slate-400">
                      Spatial Drift
                    </span>
                    <span className="font-mono font-bold text-slate-200">
                      {cosineDrift.toFixed(3)}
                    </span>
                  </div>
                  <div className="bg-slate-900/50 p-2 rounded border border-slate-800/50">
                    <span className="block text-[10px] font-mono text-slate-400">
                      Trainable %
                    </span>
                    <span className="font-mono font-bold text-slate-200">
                      {Math.round(trainableFraction * 100)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Status / Probe Tag */}
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span>Probe Strategy:</span>
                <span className="text-slate-300 font-semibold">Frozen Probe</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
