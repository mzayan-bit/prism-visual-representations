"use client";

import React, { useState } from "react";
import { TemporalTimelineMetricPayload } from "../types";

interface TemporalRepresentationTimelineProps {
  metrics: TemporalTimelineMetricPayload[];
  activeFrameIndex: number;
  onSelectFrame: (index: number) => void;
}

export const TemporalRepresentationTimeline: React.FC<
  TemporalRepresentationTimelineProps
> = ({ metrics, activeFrameIndex, onSelectFrame }) => {
  const [metricKey, setMetricKey] = useState<
    "adjacent_drift" | "adjacent_cosine_similarity" | "representation_norm" | "motion_displacement"
  >("adjacent_drift");

  if (!metrics || metrics.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500">
        No temporal timeline metrics available.
      </div>
    );
  }

  const values = metrics.map((m) => m[metricKey]);
  const minVal = Math.min(...values, 0);
  const maxVal = Math.max(...values, 1e-4);
  const valRange = maxVal - minVal || 1;

  const chartWidth = 500;
  const chartHeight = 160;
  const padding = 30;

  const points = metrics.map((m, idx) => {
    const x = padding + (idx / Math.max(1, metrics.length - 1)) * (chartWidth - 2 * padding);
    const y = chartHeight - padding - ((m[metricKey] - minVal) / valRange) * (chartHeight - 2 * padding);
    return { x, y, val: m[metricKey], t: m.timestep };
  });

  const pathD = points.reduce((acc, p, idx) => {
    return `${acc} ${idx === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
  }, "");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header & Metric Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              Frame Representation Timeline
            </h2>
            <span className="text-[10px] bg-slate-800 text-slate-400 font-mono px-2 py-0.5 rounded border border-slate-700">
              T = {metrics.length}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Temporal dynamics, Euclidean drift, and cosine stability across timesteps
          </p>
        </div>

        {/* Metric Selector Tabs */}
        <div className="flex flex-wrap gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setMetricKey("adjacent_drift")}
            className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
              metricKey === "adjacent_drift"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Feature Drift (Δh)
          </button>
          <button
            onClick={() => setMetricKey("adjacent_cosine_similarity")}
            className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
              metricKey === "adjacent_cosine_similarity"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Cosine Stability
          </button>
          <button
            onClick={() => setMetricKey("representation_norm")}
            className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
              metricKey === "representation_norm"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Feature Norm (||h||)
          </button>
          <button
            onClick={() => setMetricKey("motion_displacement")}
            className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors ${
              metricKey === "motion_displacement"
                ? "bg-violet-500/20 text-violet-300 border border-violet-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Motion (Δp)
          </button>
        </div>
      </div>

      {/* SVG Timeline Chart */}
      <div className="relative bg-slate-950 rounded-xl p-3 border border-slate-800">
        <svg
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="w-full h-44 overflow-visible"
        >
          {/* Horizontal grid lines */}
          <line
            x1={padding}
            y1={padding}
            x2={chartWidth - padding}
            y2={padding}
            stroke="#334155"
            strokeDasharray="4 4"
            strokeWidth="0.8"
          />
          <line
            x1={padding}
            y1={chartHeight / 2}
            x2={chartWidth - padding}
            y2={chartHeight / 2}
            stroke="#334155"
            strokeDasharray="4 4"
            strokeWidth="0.8"
          />
          <line
            x1={padding}
            y1={chartHeight - padding}
            x2={chartWidth - padding}
            y2={chartHeight - padding}
            stroke="#475569"
            strokeWidth="1"
          />

          {/* Value Labels on Y-axis */}
          <text
            x={padding - 6}
            y={padding + 4}
            textAnchor="end"
            className="text-[9px] font-mono fill-slate-500"
          >
            {maxVal.toFixed(2)}
          </text>
          <text
            x={padding - 6}
            y={chartHeight - padding + 3}
            textAnchor="end"
            className="text-[9px] font-mono fill-slate-500"
          >
            {minVal.toFixed(2)}
          </text>

          {/* Metric Curve */}
          <path
            d={pathD}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Active Frame Indicator Line */}
          {points[activeFrameIndex] && (
            <line
              x1={points[activeFrameIndex].x}
              y1={padding}
              x2={points[activeFrameIndex].x}
              y2={chartHeight - padding}
              stroke="#06b6d4"
              strokeWidth="1.5"
              strokeDasharray="2 2"
            />
          )}

          {/* Point markers */}
          {points.map((p, idx) => {
            const isActive = idx === activeFrameIndex;
            return (
              <g
                key={idx}
                className="cursor-pointer group"
                onClick={() => onSelectFrame(idx)}
              >
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={isActive ? 6 : 4}
                  className={`transition-all ${
                    isActive
                      ? "fill-cyan-400 stroke-slate-950 stroke-2"
                      : "fill-amber-400 hover:fill-amber-300 stroke-slate-900 stroke-1"
                  }`}
                />
                <text
                  x={p.x}
                  y={chartHeight - padding + 15}
                  textAnchor="middle"
                  className={`text-[10px] font-mono ${
                    isActive ? "fill-cyan-400 font-bold" : "fill-slate-500"
                  }`}
                >
                  t={p.t}
                </text>
                <text
                  x={p.x}
                  y={p.y - 10}
                  textAnchor="middle"
                  className="text-[9px] font-mono fill-slate-300 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  {p.val.toFixed(3)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* KPI Value Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-2.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">Active Frame Norm</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {metrics[activeFrameIndex]?.representation_norm.toFixed(3) || "0.000"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-2.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">Adjacent Drift (Δh)</div>
          <div className="text-sm font-bold font-mono text-amber-300 mt-0.5">
            {metrics[activeFrameIndex]?.adjacent_drift.toFixed(3) || "0.000"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-2.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">Cosine Stability</div>
          <div className="text-sm font-bold font-mono text-emerald-300 mt-0.5">
            {metrics[activeFrameIndex]?.adjacent_cosine_similarity.toFixed(3) || "1.000"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-2.5">
          <div className="text-[10px] font-mono text-slate-400 uppercase">Motion Displacement</div>
          <div className="text-sm font-bold font-mono text-violet-300 mt-0.5">
            {metrics[activeFrameIndex]?.motion_displacement.toFixed(3) || "0.000"}
          </div>
        </div>
      </div>
    </div>
  );
};
