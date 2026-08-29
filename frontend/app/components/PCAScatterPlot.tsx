"use client";

import React, { useMemo, useState } from "react";
import { CentroidGeometryReport, NearestNeighborEntry, ProjectionResult, SampleNeighborhood } from "../types";

interface PCAScatterPlotProps {
  projection: ProjectionResult;
  centroidGeometry: CentroidGeometryReport;
  sampleNeighborhoods: Record<string, SampleNeighborhood>;
  selectedSampleId: string | null;
  onSelectSample: (sampleId: string | null) => void;
}

const CLASS_COLORS = [
  { fill: "rgba(6, 182, 212, 0.8)", stroke: "#06b6d4", name: "Class 0 (Cyan)" },
  { fill: "rgba(168, 85, 247, 0.8)", stroke: "#a855f7", name: "Class 1 (Purple)" },
  { fill: "rgba(16, 185, 129, 0.8)", stroke: "#10b981", name: "Class 2 (Emerald)" },
  { fill: "rgba(245, 158, 11, 0.8)", stroke: "#f59e0b", name: "Class 3 (Amber)" },
  { fill: "rgba(244, 63, 94, 0.8)", stroke: "#f43f5e", name: "Class 4 (Rose)" },
];

export const PCAScatterPlot: React.FC<PCAScatterPlotProps> = ({
  projection,
  centroidGeometry,
  sampleNeighborhoods,
  selectedSampleId,
  onSelectSample,
}) => {
  const [hoveredSampleId, setHoveredSampleId] = useState<string | null>(null);
  const [showCentroids, setShowCentroids] = useState(true);
  const [showNeighborLines, setShowNeighborLines] = useState(true);

  const { coordinates, sample_ids, labels, explained_variance_ratio } = projection;

  // Compute bounding box
  const bounds = useMemo(() => {
    if (!coordinates || coordinates.length === 0) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const [x, y] of coordinates) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const padX = Math.max(0.1, (maxX - minX) * 0.12);
    const padY = Math.max(0.1, (maxY - minY) * 0.12);

    return {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    };
  }, [coordinates]);

  const width = 600;
  const height = 440;
  const padding = 45;

  const scaleX = (val: number) => {
    const range = bounds.maxX - bounds.minX || 1;
    return padding + ((val - bounds.minX) / range) * (width - 2 * padding);
  };

  const scaleY = (val: number) => {
    const range = bounds.maxY - bounds.minY || 1;
    return height - padding - ((val - bounds.minY) / range) * (height - 2 * padding);
  };

  // Build sample map
  const sampleMap = useMemo(() => {
    const map = new Map<
      string,
      { id: string; x: number; y: number; label: string | number; colorIdx: number }
    >();
    sample_ids.forEach((id, idx) => {
      const coord = coordinates[idx] || [0, 0];
      const lbl = labels[idx];
      const colorIdx = typeof lbl === "number" ? lbl % CLASS_COLORS.length : 0;
      map.set(id, {
        id,
        x: coord[0],
        y: coord[1],
        label: lbl,
        colorIdx,
      });
    });
    return map;
  }, [sample_ids, coordinates, labels]);

  const activeNeighborhood = selectedSampleId
    ? sampleNeighborhoods[selectedSampleId]
    : null;
  const hoveredSample = hoveredSampleId ? sampleMap.get(hoveredSampleId) : null;
  const selectedSample = selectedSampleId ? sampleMap.get(selectedSampleId) : null;

  // Compute centroid 2D projections by averaging projected points per class
  const classCentroidPoints = useMemo(() => {
    const classGroups = new Map<string, { sumX: number; sumY: number; count: number }>();
    sampleMap.forEach((pt) => {
      const key = String(pt.label);
      if (!classGroups.has(key)) {
        classGroups.set(key, { sumX: 0, sumY: 0, count: 0 });
      }
      const g = classGroups.get(key)!;
      g.sumX += pt.x;
      g.sumY += pt.y;
      g.count += 1;
    });

    const result: Array<{
      classId: string;
      x: number;
      y: number;
      radius90: number;
      colorIdx: number;
    }> = [];

    classGroups.forEach((g, key) => {
      const meanX = g.sumX / g.count;
      const meanY = g.sumY / g.count;
      const cSummary = centroidGeometry.class_centroids[key];
      const r90 = cSummary ? cSummary.intra_class_radius_90 : 0.5;
      const colorIdx = parseInt(key, 10) || 0;
      result.push({
        classId: key,
        x: meanX,
        y: meanY,
        radius90: r90,
        colorIdx: colorIdx % CLASS_COLORS.length,
      });
    });

    return result;
  }, [sampleMap, centroidGeometry]);

  return (
    <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-4 flex flex-col h-full">
      {/* Header Controls */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80">
        <div>
          <h2 className="text-sm font-bold font-mono text-zinc-100 flex items-center gap-2">
            <span>2D Principal Component Projection</span>
            <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/40">
              PC1 + PC2
            </span>
          </h2>
          <p className="text-[11px] text-zinc-400">
            Click point to inspect local neighborhood &bull; Hover for coordinates
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <label className="flex items-center gap-1.5 cursor-pointer text-zinc-400 hover:text-zinc-200">
            <input
              type="checkbox"
              checked={showCentroids}
              onChange={(e) => setShowCentroids(e.target.checked)}
              className="rounded bg-zinc-800 border-zinc-700 text-cyan-500 focus:ring-0 cursor-pointer"
            />
            <span>Centroids</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer text-zinc-400 hover:text-zinc-200">
            <input
              type="checkbox"
              checked={showNeighborLines}
              onChange={(e) => setShowNeighborLines(e.target.checked)}
              className="rounded bg-zinc-800 border-zinc-700 text-cyan-500 focus:ring-0 cursor-pointer"
            />
            <span>k-NN Rays</span>
          </label>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative flex-1 min-h-[380px] mt-2 flex items-center justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full max-h-[460px] select-none"
        >
          {/* Background Grid */}
          <rect
            x={padding}
            y={padding}
            width={width - 2 * padding}
            height={height - 2 * padding}
            fill="#09090b"
            stroke="#27272a"
            strokeWidth="1"
            rx="6"
          />

          {/* Coordinate Axes */}
          <line
            x1={padding}
            y1={scaleY(0)}
            x2={width - padding}
            y2={scaleY(0)}
            stroke="#3f3f46"
            strokeDasharray="4 4"
            strokeWidth="1"
          />
          <line
            x1={scaleX(0)}
            y1={padding}
            x2={scaleX(0)}
            y2={height - padding}
            stroke="#3f3f46"
            strokeDasharray="4 4"
            strokeWidth="1"
          />

          {/* Axis Labels */}
          <text
            x={width - padding - 8}
            y={scaleY(0) - 8}
            fill="#71717a"
            fontSize="10"
            fontFamily="monospace"
            textAnchor="end"
          >
            PC1 ({((explained_variance_ratio[0] || 0) * 100).toFixed(1)}%) &rarr;
          </text>
          <text
            x={scaleX(0) + 8}
            y={padding + 16}
            fill="#71717a"
            fontSize="10"
            fontFamily="monospace"
          >
            &uarr; PC2 ({((explained_variance_ratio[1] || 0) * 100).toFixed(1)}%)
          </text>

          {/* Class Centroids with dispersion circle */}
          {showCentroids &&
            classCentroidPoints.map((c) => {
              const cx = scaleX(c.x);
              const cy = scaleY(c.y);
              const color = CLASS_COLORS[c.colorIdx];
              return (
                <g key={`centroid-${c.classId}`}>
                  {/* Centroid Crosshair */}
                  <line
                    x1={cx - 8}
                    y1={cy}
                    x2={cx + 8}
                    y2={cy}
                    stroke={color.stroke}
                    strokeWidth="2.5"
                  />
                  <line
                    x1={cx}
                    y1={cy - 8}
                    x2={cx}
                    y2={cy + 8}
                    stroke={color.stroke}
                    strokeWidth="2.5"
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={10}
                    fill="none"
                    stroke={color.stroke}
                    strokeWidth="1.5"
                    strokeDasharray="2 2"
                  />
                  <text
                    x={cx + 12}
                    y={cy - 10}
                    fill={color.stroke}
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    &mu;{c.classId}
                  </text>
                </g>
              );
            })}

          {/* Nearest Neighbor Connection Lines for Selected Point */}
          {showNeighborLines &&
            selectedSample &&
            activeNeighborhood &&
            activeNeighborhood.neighbors.map((n: NearestNeighborEntry) => {
              const nPt = sampleMap.get(n.neighbor_sample_id);
              if (!nPt) return null;
              const x1 = scaleX(selectedSample.x);
              const y1 = scaleY(selectedSample.y);
              const x2 = scaleX(nPt.x);
              const y2 = scaleY(nPt.y);

              return (
                <g key={`ray-${n.neighbor_sample_id}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={n.same_class ? "#10b981" : "#f43f5e"}
                    strokeWidth="1.5"
                    strokeDasharray={n.same_class ? "none" : "3 3"}
                    opacity="0.8"
                  />
                </g>
              );
            })}

          {/* Sample Points */}
          {Array.from(sampleMap.values()).map((pt) => {
            const cx = scaleX(pt.x);
            const cy = scaleY(pt.y);
            const isSelected = selectedSampleId === pt.id;
            const isHovered = hoveredSampleId === pt.id;
            const color = CLASS_COLORS[pt.colorIdx];

            return (
              <g
                key={pt.id}
                className="cursor-pointer transition-transform"
                onClick={() => onSelectSample(isSelected ? null : pt.id)}
                onMouseEnter={() => setHoveredSampleId(pt.id)}
                onMouseLeave={() => setHoveredSampleId(null)}
              >
                {/* Selection Halo */}
                {isSelected && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r="12"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="2"
                    className="animate-ping opacity-75"
                  />
                )}
                {isSelected && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r="10"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="2"
                  />
                )}

                {/* Point */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 6 : isHovered ? 5.5 : 4.5}
                  fill={color.fill}
                  stroke={color.stroke}
                  strokeWidth={isSelected ? "2.5" : "1.5"}
                />
              </g>
            );
          })}
        </svg>

        {/* Hover / Selection Tooltip */}
        {(hoveredSample || selectedSample) && (
          <div className="absolute top-2 right-2 bg-zinc-950/90 border border-zinc-700/80 rounded-lg p-2.5 shadow-xl text-xs font-mono text-zinc-200 pointer-events-none max-w-xs">
            <div className="flex items-center justify-between gap-2 border-b border-zinc-800 pb-1 mb-1.5">
              <span className="font-bold text-cyan-300">
                {(hoveredSample || selectedSample)?.id}
              </span>
              <span
                className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                style={{
                  backgroundColor:
                    CLASS_COLORS[(hoveredSample || selectedSample)!.colorIdx].fill,
                  color: "#ffffff",
                }}
              >
                Class {(hoveredSample || selectedSample)?.label}
              </span>
            </div>
            <div className="text-[11px] space-y-0.5 text-zinc-400">
              <div>
                PC1: {(hoveredSample || selectedSample)?.x.toFixed(4)} &bull; PC2:{" "}
                {(hoveredSample || selectedSample)?.y.toFixed(4)}
              </div>
              {selectedSample && activeNeighborhood && (
                <div className="text-emerald-400 pt-1">
                  5-NN Consistency:{" "}
                  {(activeNeighborhood.same_class_fraction * 100).toFixed(0)}%
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Class Legend */}
      <div className="pt-3 border-t border-zinc-800/80 flex flex-wrap items-center gap-4 text-xs font-mono">
        <span className="text-zinc-500 font-bold">Legend:</span>
        {centroidGeometry.class_order.map((cId, idx) => {
          const color = CLASS_COLORS[idx % CLASS_COLORS.length];
          const summary = centroidGeometry.class_centroids[cId];
          return (
            <div key={cId} className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full border"
                style={{ backgroundColor: color.fill, borderColor: color.stroke }}
              />
              <span className="text-zinc-300 font-medium">
                Class {cId}
                {summary && (
                  <span className="text-zinc-500 ml-1">
                    (n={summary.sample_count})
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
