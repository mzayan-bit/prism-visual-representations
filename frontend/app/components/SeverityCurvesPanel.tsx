"use client";

import React, { useState } from "react";
import { CorruptionSeverityCurve } from "../types";

interface SeverityCurvesPanelProps {
  curve: CorruptionSeverityCurve | null;
  corruptionName: string;
}

export default function SeverityCurvesPanel({
  curve,
  corruptionName,
}: SeverityCurvesPanelProps) {
  const [activeMetric, setActiveMetric] = useState<"accuracy" | "loss" | "drift" | "consistency">("accuracy");

  if (!curve) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm">
        No severity curve data available.
      </div>
    );
  }

  // SVG Chart dimensions
  const width = 600;
  const height = 260;
  const padLeft = 50;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const severities = curve.severities;
  const numSteps = severities.length;

  const getPoints = () => {
    if (activeMetric === "accuracy") {
      return curve.accuracy_trajectory.map((val, idx) => ({
        sev: severities[idx],
        val,
        normY: 1.0 - Math.max(0, Math.min(1.0, val)),
        display: `${(val * 100).toFixed(1)}%`,
      }));
    } else if (activeMetric === "loss") {
      const maxLoss = Math.max(2.0, ...curve.loss_trajectory);
      return curve.loss_trajectory.map((val, idx) => ({
        sev: severities[idx],
        val,
        normY: 1.0 - Math.max(0, Math.min(1.0, val / maxLoss)),
        display: val.toFixed(3),
      }));
    } else if (activeMetric === "drift") {
      const maxDrift = Math.max(1.0, ...curve.representation_drift_trajectory);
      return curve.representation_drift_trajectory.map((val, idx) => ({
        sev: severities[idx],
        val,
        normY: 1.0 - Math.max(0, Math.min(1.0, val / maxDrift)),
        display: val.toFixed(3),
      }));
    } else {
      return curve.neighbor_consistency_trajectory.map((val, idx) => ({
        sev: severities[idx],
        val,
        normY: 1.0 - Math.max(0, Math.min(1.0, val)),
        display: `${(val * 100).toFixed(1)}%`,
      }));
    }
  };

  const points = getPoints();

  // Compute SVG polyline string
  const svgPoints = points
    .map((p, idx) => {
      const x = padLeft + (idx / Math.max(1, numSteps - 1)) * chartW;
      const y = padTop + p.normY * chartH;
      return `${x},${y}`;
    })
    .join(" ");

  // Baseline Y position for clean accuracy
  const cleanBaselineY =
    padTop + (1.0 - Math.max(0, Math.min(1.0, curve.clean_accuracy))) * chartH;

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>📉</span> Severity Degradation Trajectory
          </h2>
          <p className="text-xs text-slate-400">
            Systematic response to {corruptionName.replace(/_/g, " ")} across levels (1 = subtle, 5 = severe)
          </p>
        </div>

        {/* Metric Toggles */}
        <div className="flex items-center gap-1.5 bg-slate-950/60 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveMetric("accuracy")}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMetric === "accuracy"
                ? "bg-cyan-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Accuracy
          </button>
          <button
            onClick={() => setActiveMetric("loss")}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMetric === "loss"
                ? "bg-amber-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Loss
          </button>
          <button
            onClick={() => setActiveMetric("drift")}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMetric === "drift"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Drift
          </button>
          <button
            onClick={() => setActiveMetric("consistency")}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
              activeMetric === "consistency"
                ? "bg-emerald-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Consistency
          </button>
        </div>
      </div>

      {/* Trajectory Summary Badges */}
      <div className="flex flex-wrap items-center gap-3 mb-4 text-xs">
        <div className="px-3 py-1 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2">
          <span className="text-slate-400">Clean Baseline:</span>
          <span className="font-bold text-emerald-400">
            {(curve.clean_accuracy * 100).toFixed(1)}%
          </span>
        </div>
        <div className="px-3 py-1 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2">
          <span className="text-slate-400">Area Under Curve (AUC):</span>
          <span className="font-bold text-cyan-300">
            {(curve.area_under_curve * 100).toFixed(1)}%
          </span>
        </div>
        <div className="px-3 py-1 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2">
          <span className="text-slate-400">Total Drop (Sev 1→5):</span>
          <span className="font-bold text-rose-400">
            -{(curve.total_accuracy_drop * 100).toFixed(1)}%
          </span>
        </div>
        <div className="px-3 py-1 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2">
          <span className="text-slate-400">Mean Robust Accuracy:</span>
          <span className="font-bold text-white">
            {(curve.mean_accuracy * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* SVG Trajectory Chart */}
      <div className="w-full overflow-x-auto bg-slate-950/40 rounded-xl p-3 border border-slate-800/80">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-h-[300px]">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((t) => {
            const y = padTop + t * chartH;
            return (
              <g key={t}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="#334155"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={padLeft - 8}
                  y={y + 4}
                  textAnchor="end"
                  fill="#64748b"
                  fontSize="10"
                >
                  {activeMetric === "accuracy" || activeMetric === "consistency"
                    ? `${Math.round((1.0 - t) * 100)}%`
                    : (1.0 - t).toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Clean baseline line if accuracy */}
          {activeMetric === "accuracy" && (
            <g>
              <line
                x1={padLeft}
                y1={cleanBaselineY}
                x2={width - padRight}
                y2={cleanBaselineY}
                stroke="#10b981"
                strokeDasharray="4 4"
                strokeWidth="2"
              />
              <text
                x={width - padRight}
                y={cleanBaselineY - 6}
                textAnchor="end"
                fill="#10b981"
                fontSize="10"
                fontWeight="bold"
              >
                Clean Baseline
              </text>
            </g>
          )}

          {/* Area fill under curve */}
          <polygon
            points={`${padLeft},${padTop + chartH} ${svgPoints} ${
              width - padRight
            },${padTop + chartH}`}
            fill={
              activeMetric === "accuracy"
                ? "url(#cyanGrad)"
                : activeMetric === "loss"
                ? "url(#amberGrad)"
                : "url(#indigoGrad)"
            }
            opacity="0.25"
          />

          {/* Gradient definitions */}
          <defs>
            <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="amberGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="indigoGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Main Trajectory Line */}
          <polyline
            fill="none"
            stroke={
              activeMetric === "accuracy"
                ? "#06b6d4"
                : activeMetric === "loss"
                ? "#f59e0b"
                : activeMetric === "drift"
                ? "#6366f1"
                : "#10b981"
            }
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={svgPoints}
          />

          {/* Points & Labels */}
          {points.map((p, idx) => {
            const x = padLeft + (idx / Math.max(1, numSteps - 1)) * chartW;
            const y = padTop + p.normY * chartH;
            return (
              <g key={idx}>
                {/* Circle point */}
                <circle
                  cx={x}
                  cy={y}
                  r="5"
                  fill="#0f172a"
                  stroke={
                    activeMetric === "accuracy"
                      ? "#06b6d4"
                      : activeMetric === "loss"
                      ? "#f59e0b"
                      : "#6366f1"
                  }
                  strokeWidth="2.5"
                />
                {/* Value tooltip label above point */}
                <text
                  x={x}
                  y={y - 10}
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize="11"
                  fontWeight="bold"
                >
                  {p.display}
                </text>
                {/* X Axis Severity Label */}
                <text
                  x={x}
                  y={padTop + chartH + 20}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="11"
                >
                  Sev {p.sev}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
