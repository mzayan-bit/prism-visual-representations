"use client";

import React, { useState } from "react";
import { SSLGeometryPointPayload } from "../types";

interface SSLGeometryPanelProps {
  points: SSLGeometryPointPayload[];
}

const CLASS_COLORS: Record<number, string> = {
  0: "#38bdf8", // Sky
  1: "#fbbf24", // Amber
  2: "#34d399", // Emerald
  3: "#f472b6", // Pink
  4: "#a78bfa", // Violet
  5: "#fb7185", // Rose
  6: "#818cf8", // Indigo
  7: "#2dd4bf", // Teal
};

export function SSLGeometryPanel({ points }: SSLGeometryPanelProps) {
  const [hoveredPoint, setHoveredPoint] = useState<SSLGeometryPointPayload | null>(null);

  const width = 460;
  const height = 240;
  const pad = 30;

  const minX = -4.0;
  const maxX = 4.0;
  const minY = -4.0;
  const maxY = 4.0;

  const mapCoord = (x: number, y: number) => {
    const cx = pad + ((x - minX) / (maxX - minX)) * (width - 2 * pad);
    const cy = height - pad - ((y - minY) / (maxY - minY)) * (height - 2 * pad);
    return { cx, cy };
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Post-Hoc Representation Geometry (2D PCA)
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Emergent semantic clustering in frozen SimCLR feature space.
          </p>
        </div>
        <span className="text-[10px] font-mono bg-amber-950/60 text-amber-300 border border-amber-800/60 px-2 py-0.5 rounded">
          Unlabeled Pretraining
        </span>
      </div>

      {/* Mandatory Disclaimer */}
      <div className="text-[11px] text-slate-400 bg-slate-950 border border-slate-800/80 rounded-md p-2.5 mb-3 flex items-center gap-2">
        <span className="text-indigo-400 font-bold text-xs">&bull;</span>
        <span>
          <strong className="text-slate-200 font-semibold">Scientific Transparency: </strong>
          Class labels shown here are strictly for post-hoc evaluation and visualization. They were completely excluded during contrastive pretraining.
        </span>
      </div>

      <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56">
          {/* Axis grid */}
          <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} stroke="#334155" strokeDasharray="3,3" strokeWidth="0.8" />
          <line x1={width / 2} y1={pad} x2={width / 2} y2={height - pad} stroke="#334155" strokeDasharray="3,3" strokeWidth="0.8" />

          {/* Scatter points */}
          {points.map((pt, idx) => {
            const { cx, cy } = mapCoord(pt.pca_x, pt.pca_y);
            const color = CLASS_COLORS[pt.class_label] || "#94a3b8";
            const isHovered = hoveredPoint?.sample_id === pt.sample_id;

            return (
              <circle
                key={idx}
                cx={cx}
                cy={cy}
                r={isHovered ? 6 : 4}
                fill={color}
                stroke={isHovered ? "#ffffff" : "#0f172a"}
                strokeWidth={isHovered ? 2 : 1}
                className="cursor-pointer transition-all duration-150"
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            );
          })}
        </svg>

        {hoveredPoint && (
          <div className="absolute top-4 right-4 bg-slate-900 border border-slate-700/80 rounded px-2.5 py-1.5 text-xs shadow-md">
            <div className="font-semibold text-white">{hoveredPoint.class_name}</div>
            <div className="text-[10px] text-slate-400 font-mono">
              PCA: ({hoveredPoint.pca_x.toFixed(2)}, {hoveredPoint.pca_y.toFixed(2)})
            </div>
            <div className="text-[10px] text-slate-500 font-mono">{hoveredPoint.sample_id}</div>
          </div>
        )}

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-2 pt-2 border-t border-slate-800/60 text-[11px]">
          {Object.entries(CLASS_COLORS).map(([clsId, color]) => (
            <div key={clsId} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: color }} />
              <span className="text-slate-300">Class {clsId}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
