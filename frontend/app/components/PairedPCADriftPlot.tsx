"use client";

import React, { useState } from "react";
import { SharedPCAProjectionResult } from "../types";

interface PairedPCADriftPlotProps {
  pcaResult: SharedPCAProjectionResult | null;
  selectedSampleId: string | null;
  onSelectSample: (sampleId: string) => void;
  classNames?: string[];
}

const CLASS_COLORS = [
  "#38bdf8", // Sky blue
  "#f43f5e", // Rose
  "#10b981", // Emerald
  "#a855f7", // Purple
  "#f59e0b", // Amber
  "#06b6d4", // Cyan
];

export default function PairedPCADriftPlot({
  pcaResult,
  selectedSampleId,
  onSelectSample,
  classNames = ["Class 0", "Class 1", "Class 2"],
}: PairedPCADriftPlotProps) {
  const [showVectors, setShowVectors] = useState(true);
  const [hoveredSampleId, setHoveredSampleId] = useState<string | null>(null);

  if (!pcaResult || pcaResult.num_samples === 0) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-slate-400 text-sm">
        No PCA projection data available.
      </div>
    );
  }

  // Find min/max bounds across both clean and corrupted coordinates to ensure identical scaling
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  const allCoords = [
    ...pcaResult.clean_coordinates,
    ...pcaResult.corrupted_coordinates,
  ];

  for (const pt of allCoords) {
    if (pt[0] < minX) minX = pt[0];
    if (pt[0] > maxX) maxX = pt[0];
    if (pt[1] < minY) minY = pt[1];
    if (pt[1] > maxY) maxY = pt[1];
  }

  // Add 15% margin
  const rangeX = Math.max(1e-5, maxX - minX);
  const rangeY = Math.max(1e-5, maxY - minY);
  const marginX = rangeX * 0.15;
  const marginY = rangeY * 0.15;

  const boundMinX = minX - marginX;
  const boundMaxX = maxX + marginX;
  const boundMinY = minY - marginY;
  const boundMaxY = maxY + marginY;

  const width = 640;
  const height = 440;
  const pad = 40;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;

  const mapX = (val: number) =>
    pad + ((val - boundMinX) / (boundMaxX - boundMinX)) * plotW;
  const mapY = (val: number) =>
    pad + (1.0 - (val - boundMinY) / (boundMaxY - boundMinY)) * plotH;

  const evr1 = pcaResult.explained_variance_ratio[0]
    ? (pcaResult.explained_variance_ratio[0] * 100).toFixed(1)
    : "0";
  const evr2 = pcaResult.explained_variance_ratio[1]
    ? (pcaResult.explained_variance_ratio[1] * 100).toFixed(1)
    : "0";

  return (
    <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>🧭</span> Shared PCA Manifold & Drift Vectors
          </h2>
          <p className="text-xs text-slate-400">
            Clean (●) vs Corrupted (✕) positions projected into identical clean basis
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
            <input
              type="checkbox"
              checked={showVectors}
              onChange={(e) => setShowVectors(e.target.checked)}
              className="accent-cyan-500 rounded cursor-pointer"
            />
            <span>Show Drift Vectors (→)</span>
          </label>
        </div>
      </div>

      {/* Class Legend */}
      <div className="flex flex-wrap items-center gap-4 mb-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Classes:</span>
          {classNames.map((name, idx) => {
            const color = CLASS_COLORS[idx % CLASS_COLORS.length];
            return (
              <div key={idx} className="flex items-center gap-1">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-slate-300 font-medium">{name}</span>
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-3 ml-auto text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-300" /> Clean
          </span>
          <span className="flex items-center gap-1 font-bold text-slate-200">
            ✕ Corrupted
          </span>
        </div>
      </div>

      {/* Interactive SVG Plot */}
      <div className="relative w-full overflow-hidden bg-slate-950/80 rounded-xl border border-slate-800">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          {/* Background Grid */}
          <g stroke="#1e293b" strokeWidth="1" strokeDasharray="4 4">
            <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} />
            <line x1={width / 2} y1={pad} x2={width / 2} y2={height - pad} />
          </g>

          {/* Marker definition for arrow heads */}
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
            </marker>
          </defs>

          {/* Displacement Lines / Vectors */}
          {showVectors &&
            pcaResult.sample_ids.map((sid, idx) => {
              const cPt = pcaResult.clean_coordinates[idx];
              const crPt = pcaResult.corrupted_coordinates[idx];
              const x1 = mapX(cPt[0]);
              const y1 = mapY(cPt[1]);
              const x2 = mapX(crPt[0]);
              const y2 = mapY(crPt[1]);
              const isSelected = selectedSampleId === sid;
              const isHovered = hoveredSampleId === sid;

              return (
                <line
                  key={`vec-${sid}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={isSelected || isHovered ? "#38bdf8" : "#475569"}
                  strokeWidth={isSelected || isHovered ? "2.5" : "1.2"}
                  strokeOpacity={isSelected || isHovered ? "1.0" : "0.5"}
                />
              );
            })}

          {/* Clean Coordinate Points (Circles) */}
          {pcaResult.sample_ids.map((sid, idx) => {
            const pt = pcaResult.clean_coordinates[idx];
            const rawLabel = pcaResult.labels[idx];
            const labelIdx =
              typeof rawLabel === "number" ? rawLabel : parseInt(String(rawLabel), 10) || 0;
            const color = CLASS_COLORS[labelIdx % CLASS_COLORS.length];
            const cx = mapX(pt[0]);
            const cy = mapY(pt[1]);
            const isSelected = selectedSampleId === sid;
            const isHovered = hoveredSampleId === sid;

            return (
              <g
                key={`clean-${sid}`}
                className="cursor-pointer transition-all"
                onClick={() => onSelectSample(sid)}
                onMouseEnter={() => setHoveredSampleId(sid)}
                onMouseLeave={() => setHoveredSampleId(null)}
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected || isHovered ? "7" : "4.5"}
                  fill={color}
                  stroke="#0f172a"
                  strokeWidth="1.5"
                />
              </g>
            );
          })}

          {/* Corrupted Coordinate Points (Crosses) */}
          {pcaResult.sample_ids.map((sid, idx) => {
            const pt = pcaResult.corrupted_coordinates[idx];
            const rawLabel = pcaResult.labels[idx];
            const labelIdx =
              typeof rawLabel === "number" ? rawLabel : parseInt(String(rawLabel), 10) || 0;
            const color = CLASS_COLORS[labelIdx % CLASS_COLORS.length];
            const cx = mapX(pt[0]);
            const cy = mapY(pt[1]);
            const isSelected = selectedSampleId === sid;
            const isHovered = hoveredSampleId === sid;
            const arm = isSelected || isHovered ? 6 : 4;

            return (
              <g
                key={`corr-${sid}`}
                className="cursor-pointer transition-all"
                onClick={() => onSelectSample(sid)}
                onMouseEnter={() => setHoveredSampleId(sid)}
                onMouseLeave={() => setHoveredSampleId(null)}
              >
                <line
                  x1={cx - arm}
                  y1={cy - arm}
                  x2={cx + arm}
                  y2={cy + arm}
                  stroke={color}
                  strokeWidth={isSelected || isHovered ? "3" : "2"}
                />
                <line
                  x1={cx - arm}
                  y1={cy + arm}
                  x2={cx + arm}
                  y2={cy - arm}
                  stroke={color}
                  strokeWidth={isSelected || isHovered ? "3" : "2"}
                />
              </g>
            );
          })}

          {/* Axes Labels */}
          <text
            x={width / 2}
            y={height - 10}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="11"
            fontWeight="bold"
          >
            PC 1 ({evr1}% variance)
          </text>
          <text
            x={15}
            y={height / 2}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="11"
            fontWeight="bold"
            transform={`rotate(-90 15 ${height / 2})`}
          >
            PC 2 ({evr2}% variance)
          </text>
        </svg>
      </div>

      {/* Scientific Methodology Note */}
      <div className="mt-3 p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs text-slate-400 flex items-start gap-2">
        <span className="text-cyan-400 font-bold">ℹ️ Basis Invariance:</span>
        <span>
          {pcaResult.basis_note} The geometric displacement vectors accurately measure
          how corruptions push representations outside class manifolds.
        </span>
      </div>
    </div>
  );
}
