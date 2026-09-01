"use client";

import React, { useState } from "react";
import { SharedPCADriftPayload, TransferRepresentationDriftPayload } from "../types";

interface RepresentationRetentionPanelProps {
  driftSummary: TransferRepresentationDriftPayload | null;
  sharedPCA: SharedPCADriftPayload | undefined;
  architecture: string;
}

export function RepresentationRetentionPanel({
  driftSummary,
  sharedPCA,
  architecture,
}: RepresentationRetentionPanelProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!driftSummary && !sharedPCA) {
    return (
      <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex items-center justify-center min-h-[300px]">
        <span className="text-xs text-slate-500 font-mono">
          No representation retention metrics available for {architecture.toUpperCase()}.
        </span>
      </div>
    );
  }

  // PCA Plot Layout
  const width = 440;
  const height = 240;
  const padding = { top: 20, right: 20, bottom: 30, left: 30 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  let allX: number[] = [];
  let allY: number[] = [];

  if (sharedPCA) {
    allX = [...sharedPCA.pre_coordinates.map((c) => c[0]), ...sharedPCA.post_coordinates.map((c) => c[0])];
    allY = [...sharedPCA.pre_coordinates.map((c) => c[1]), ...sharedPCA.post_coordinates.map((c) => c[1])];
  }

  const minX = allX.length ? Math.min(...allX) - 0.2 : -1;
  const maxX = allX.length ? Math.max(...allX) + 0.2 : 1;
  const minY = allY.length ? Math.min(...allY) - 0.2 : -1;
  const maxY = allY.length ? Math.max(...allY) + 0.2 : 1;

  const scaleX = (val: number) => padding.left + ((val - minX) / (maxX - minX || 1)) * innerWidth;
  const scaleY = (val: number) => padding.top + (1.0 - (val - minY) / (maxY - minY || 1)) * innerHeight;

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-full">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            Representation Retention & Shared PCA Drift
          </h3>
          <p className="text-xs text-slate-400">
            Displacement vectors projecting pre-transfer vs post-transfer feature vectors into the shared source PCA basis
          </p>
        </div>
      </div>

      {/* Numerical Retention Summary Cards */}
      {driftSummary && (
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Mean Cosine Sim
            </div>
            <div className="text-sm font-bold text-cyan-300 font-mono mt-0.5">
              {driftSummary.mean_cosine_similarity.toFixed(4)}
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Mean Euclidean Drift
            </div>
            <div className="text-sm font-bold text-amber-300 font-mono mt-0.5">
              {driftSummary.mean_euclidean_drift.toFixed(4)}
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Rel Norm Change
            </div>
            <div className="text-sm font-bold text-slate-200 font-mono mt-0.5">
              {(driftSummary.mean_relative_norm_change * 100).toFixed(2)}%
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Backbone Status
            </div>
            <div className="text-xs font-bold font-mono mt-1 text-emerald-400">
              {driftSummary.is_frozen_backbone ? "STRICTLY FROZEN" : "FINE-TUNED"}
            </div>
          </div>
        </div>
      )}

      {/* Shared PCA Scatter & Drift Arrows */}
      {sharedPCA && (
        <div className="relative flex-1 flex flex-col items-center justify-center">
          {/* Legend */}
          <div className="flex items-center justify-end w-full gap-4 text-xs font-mono mb-1">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block"></span>
              <span className="text-cyan-300">Pre-Transfer</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span>
              <span className="text-emerald-300">Post-Transfer</span>
            </div>
          </div>

          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-full max-h-[220px] overflow-visible bg-slate-950/40 rounded-lg border border-slate-800"
          >
            {/* Axis Lines */}
            <line
              x1={padding.left}
              y1={scaleY(0)}
              x2={width - padding.right}
              y2={scaleY(0)}
              stroke="#334155"
              strokeDasharray="2 2"
            />
            <line
              x1={scaleX(0)}
              y1={padding.top}
              x2={scaleX(0)}
              y2={height - padding.bottom}
              stroke="#334155"
              strokeDasharray="2 2"
            />

            {/* Displacement Arrows & Lines */}
            {sharedPCA.pre_coordinates.map((pre, idx) => {
              const post = sharedPCA.post_coordinates[idx];
              const x1 = scaleX(pre[0]);
              const y1 = scaleY(pre[1]);
              const x2 = scaleX(post[0]);
              const y2 = scaleY(post[1]);
              const isHovered = hoveredIdx === idx;

              return (
                <g key={`disp-${idx}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={isHovered ? "#f59e0b" : "#64748b"}
                    strokeWidth={isHovered ? "2" : "1"}
                    strokeOpacity={isHovered ? 1 : 0.6}
                  />

                  {/* Pre point (cyan) */}
                  <circle
                    cx={x1}
                    cy={y1}
                    r={isHovered ? "5" : "3.5"}
                    className="fill-cyan-400 cursor-pointer"
                    onMouseEnter={() => setHoveredIdx(idx)}
                    onMouseLeave={() => setHoveredIdx(null)}
                  />

                  {/* Post point (emerald) */}
                  <circle
                    cx={x2}
                    cy={y2}
                    r={isHovered ? "5" : "3.5"}
                    className="fill-emerald-400 cursor-pointer"
                    onMouseEnter={() => setHoveredIdx(idx)}
                    onMouseLeave={() => setHoveredIdx(null)}
                  />
                </g>
              );
            })}
          </svg>

          {/* Hover Details */}
          {hoveredIdx !== null && (
            <div className="absolute bottom-2 left-4 bg-slate-950/90 border border-slate-700 p-2 rounded text-[11px] font-mono pointer-events-none z-10 text-slate-300">
              Sample #{hoveredIdx + 1} Displacement: [
              {sharedPCA.displacement_vectors[hoveredIdx][0].toFixed(3)},{" "}
              {sharedPCA.displacement_vectors[hoveredIdx][1].toFixed(3)}]
            </div>
          )}
        </div>
      )}
    </div>
  );
}
