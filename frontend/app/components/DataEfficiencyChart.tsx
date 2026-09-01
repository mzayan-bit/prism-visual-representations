"use client";

import React, { useState } from "react";
import { DataEfficiencyPointPayload, SampleEfficiencySummaryPayload } from "../types";

interface DataEfficiencyChartProps {
  dataEfficiency: SampleEfficiencySummaryPayload | undefined;
  architecture: string;
}

export function DataEfficiencyChart({
  dataEfficiency,
  architecture,
}: DataEfficiencyChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<DataEfficiencyPointPayload | null>(null);

  if (!dataEfficiency || !dataEfficiency.points || dataEfficiency.points.length === 0) {
    return (
      <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex items-center justify-center min-h-[300px]">
        <span className="text-xs text-slate-500 font-mono">
          No data efficiency trajectory available for {architecture.toUpperCase()}.
        </span>
      </div>
    );
  }

  // Filter series
  const scratchPoints = dataEfficiency.points
    .filter((p) => p.strategy === "scratch_baseline")
    .sort((a, b) => a.data_budget - b.data_budget);

  const probePoints = dataEfficiency.points
    .filter((p) => p.strategy === "linear_probe")
    .sort((a, b) => a.data_budget - b.data_budget);

  // SVG Chart dimensions
  const width = 500;
  const height = 240;
  const padding = { top: 25, right: 30, bottom: 40, left: 50 };

  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  // Scale functions
  const xScale = (budget: number) => {
    // Budget range [0.05, 1.0]
    const minB = 0.05;
    const maxB = 1.0;
    return padding.left + ((budget - minB) / (maxB - minB)) * innerWidth;
  };

  const yScale = (acc: number) => {
    // Acc range [0.0, 1.0]
    return padding.top + (1.0 - Math.min(1.0, Math.max(0.0, acc))) * innerHeight;
  };

  const createPath = (points: DataEfficiencyPointPayload[]) => {
    if (points.length === 0) return "";
    return points
      .map((p, idx) => {
        const x = xScale(p.data_budget);
        const y = yScale(p.val_accuracy);
        return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const scratchPath = createPath(scratchPoints);
  const probePath = createPath(probePoints);

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-full">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
            Label-Efficiency Transfer Curves
          </h3>
          <p className="text-xs text-slate-400">
            Target performance scaling across nested data budgets (10% → 100%)
          </p>
        </div>
        <div className="text-right">
          <div className="text-[10px] uppercase font-mono text-slate-400">
            Normalized AUC
          </div>
          <div className="text-xs font-bold text-amber-300 font-mono">
            {dataEfficiency.normalized_auc.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-end gap-4 text-xs font-mono mb-2">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-cyan-400 inline-block"></span>
          <span className="text-cyan-300 font-medium">Linear Probe (Transfer)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-slate-400 border-b border-dashed border-slate-400 inline-block"></span>
          <span className="text-slate-400">Scratch Baseline</span>
        </div>
      </div>

      {/* Chart SVG */}
      <div className="relative flex-1 flex items-center justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full max-h-[240px] overflow-visible"
        >
          {/* Grid lines */}
          {[0.0, 0.25, 0.5, 0.75, 1.0].map((tick) => {
            const y = yScale(tick);
            return (
              <g key={tick}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="#334155"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={padding.left - 8}
                  y={y + 4}
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="monospace"
                  textAnchor="end"
                >
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}

          {/* X Axis Ticks */}
          {[0.1, 0.25, 0.5, 1.0].map((b) => {
            const x = xScale(b);
            return (
              <g key={b}>
                <line
                  x1={x}
                  y1={height - padding.bottom}
                  x2={x}
                  y2={height - padding.bottom + 5}
                  stroke="#64748b"
                  strokeWidth="1"
                />
                <text
                  x={x}
                  y={height - padding.bottom + 18}
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="monospace"
                  textAnchor="middle"
                >
                  {Math.round(b * 100)}%
                </text>
              </g>
            );
          })}

          {/* Axis Labels */}
          <text
            x={padding.left + innerWidth / 2}
            y={height - 5}
            fill="#cbd5e1"
            fontSize="10"
            fontFamily="monospace"
            textAnchor="middle"
          >
            Target Training Budget Fraction
          </text>

          {/* Series Lines */}
          <path
            d={scratchPath}
            fill="none"
            stroke="#94a3b8"
            strokeWidth="2"
            strokeDasharray="4 4"
          />

          <path
            d={probePath}
            fill="none"
            stroke="#22d3ee"
            strokeWidth="2.5"
          />

          {/* Scratch Points */}
          {scratchPoints.map((p, idx) => (
            <circle
              key={`scratch-${idx}`}
              cx={xScale(p.data_budget)}
              cy={yScale(p.val_accuracy)}
              r="4"
              className="fill-slate-400 cursor-pointer hover:r-6 transition-all"
              onMouseEnter={() => setHoveredPoint(p)}
              onMouseLeave={() => setHoveredPoint(null)}
            />
          ))}

          {/* Probe Points */}
          {probePoints.map((p, idx) => (
            <circle
              key={`probe-${idx}`}
              cx={xScale(p.data_budget)}
              cy={yScale(p.val_accuracy)}
              r="5"
              className="fill-cyan-400 cursor-pointer hover:r-7 transition-all stroke-2 stroke-slate-900"
              onMouseEnter={() => setHoveredPoint(p)}
              onMouseLeave={() => setHoveredPoint(null)}
            />
          ))}
        </svg>

        {/* Hover Tooltip Overlay */}
        {hoveredPoint && (
          <div className="absolute top-2 right-4 bg-slate-950/95 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs font-mono pointer-events-none z-10">
            <div className="text-white font-bold mb-1 capitalize">
              {hoveredPoint.strategy.replace("_", " ")}
            </div>
            <div className="text-slate-300">
              Budget: {Math.round(hoveredPoint.data_budget * 100)}% ({hoveredPoint.sample_count} samples)
            </div>
            <div className="text-cyan-300 font-semibold">
              Accuracy: {(hoveredPoint.val_accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-slate-400 text-[10px]">
              Loss: {hoveredPoint.val_loss.toFixed(4)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
