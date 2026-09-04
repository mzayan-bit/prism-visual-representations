"use client";

import React from "react";
import { TemporalPCATrajectoryPayload } from "../types";

interface TemporalPCATrajectoryPlotProps {
  trajectory: TemporalPCATrajectoryPayload[];
  activeFrameIndex: number;
  onSelectFrame: (index: number) => void;
}

export const TemporalPCATrajectoryPlot: React.FC<
  TemporalPCATrajectoryPlotProps
> = ({ trajectory, activeFrameIndex, onSelectFrame }) => {
  if (!trajectory || trajectory.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500">
        No PCA trajectory available.
      </div>
    );
  }

  const chartSize = 340;
  const padding = 40;

  const xs = trajectory.map((p) => p.pca_1);
  const ys = trajectory.map((p) => p.pca_2);

  const minX = Math.min(...xs, -1.0);
  const maxX = Math.max(...xs, 1.0);
  const minY = Math.min(...ys, -1.0);
  const maxY = Math.max(...ys, 1.0);

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const screenPoints = trajectory.map((p) => ({
    x: padding + ((p.pca_1 - minX) / rangeX) * (chartSize - 2 * padding),
    y: chartSize - padding - ((p.pca_2 - minY) / rangeY) * (chartSize - 2 * padding),
    t: p.timestep,
    rawX: p.pca_1,
    rawY: p.pca_2,
  }));

  const pathD = screenPoints.reduce((acc, p, idx) => {
    return `${acc} ${idx === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
  }, "");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h2 className="text-sm font-bold text-slate-100">
            Temporal Trajectory in Shared PCA Space
          </h2>
          <p className="text-xs text-slate-400">
            Evolution path h_0 → h_1 → h_2 → ... projected on shared 2D basis
          </p>
        </div>
        <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
          2D PCA Projection
        </span>
      </div>

      {/* 2D Trajectory Canvas */}
      <div className="relative bg-slate-950 rounded-xl p-3 border border-slate-800 flex items-center justify-center">
        <svg
          viewBox={`0 0 ${chartSize} ${chartSize}`}
          className="w-full h-64 overflow-visible"
        >
          {/* Axis lines */}
          <line
            x1={padding}
            y1={chartSize / 2}
            x2={chartSize - padding}
            y2={chartSize / 2}
            stroke="#334155"
            strokeDasharray="3 3"
            strokeWidth="0.8"
          />
          <line
            x1={chartSize / 2}
            y1={padding}
            x2={chartSize / 2}
            y2={chartSize - padding}
            stroke="#334155"
            strokeDasharray="3 3"
            strokeWidth="0.8"
          />

          {/* Path trajectory line */}
          <path
            d={pathD}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="2.5"
            strokeDasharray="4 2"
            strokeLinecap="round"
          />

          {/* Directional connecting segments */}
          {screenPoints.slice(0, -1).map((p, idx) => {
            const nextP = screenPoints[idx + 1];
            return (
              <line
                key={idx}
                x1={p.x}
                y1={p.y}
                x2={nextP.x}
                y2={nextP.y}
                stroke="#38bdf8"
                strokeWidth="2"
                markerEnd="url(#arrow)"
              />
            );
          })}

          {/* Markers */}
          {screenPoints.map((p, idx) => {
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
                  r={isActive ? 8 : 5}
                  className={`transition-all ${
                    isActive
                      ? "fill-amber-400 stroke-slate-950 stroke-2 ring-2 ring-amber-400"
                      : "fill-cyan-400 hover:fill-cyan-300 stroke-slate-900 stroke-1"
                  }`}
                />
                <text
                  x={p.x + 10}
                  y={p.y + 4}
                  className={`text-[10px] font-mono font-bold ${
                    isActive ? "fill-amber-400 font-black" : "fill-slate-400"
                  }`}
                >
                  t={p.t}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="text-[11px] text-slate-400 text-center font-mono">
        Active Frame: <span className="text-amber-400 font-bold">t = {activeFrameIndex}</span> • Coordinates: ({screenPoints[activeFrameIndex]?.rawX.toFixed(2)}, {screenPoints[activeFrameIndex]?.rawY.toFixed(2)})
      </div>
    </div>
  );
};
