"use client";

import React, { useState } from "react";
import { LayerGeometryProfile } from "../types";

interface LayerEvolutionPanelProps {
  profile: LayerGeometryProfile | null;
  onSelectLayer?: (layer: string) => void;
}

export const LayerEvolutionPanel: React.FC<LayerEvolutionPanelProps> = ({
  profile,
  onSelectLayer,
}) => {
  const [activeMetric, setActiveMetric] = useState<
    "ratio" | "compactness" | "separation" | "consistency"
  >("ratio");

  if (!profile || profile.layer_points.length === 0) {
    return (
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-8 text-center text-zinc-500 font-mono text-xs">
        No layer evolution profile available for this architecture.
      </div>
    );
  }

  const { layer_points, architecture } = profile;
  const numLayers = layer_points.length;

  // Chart Dimensions
  const width = 640;
  const height = 260;
  const padding = 45;

  // Extract series based on activeMetric
  let series: number[] = [];
  let seriesColor = "#38bdf8";
  let metricLabel = "Separation / Compactness Ratio";
  let isPercentage = false;

  if (activeMetric === "ratio") {
    series = profile.ratio_trend;
    seriesColor = "#c084fc"; // purple
    metricLabel = "Separation-to-Compactness Ratio (x)";
  } else if (activeMetric === "compactness") {
    series = profile.compactness_trend;
    seriesColor = "#06b6d4"; // cyan
    metricLabel = "Intra-Class Compactness (d̄)";
  } else if (activeMetric === "separation") {
    series = profile.separation_trend;
    seriesColor = "#818cf8"; // indigo
    metricLabel = "Inter-Class Separation (Δ)";
  } else if (activeMetric === "consistency") {
    series = profile.consistency_trend.map((v) => v * 100);
    seriesColor = "#10b981"; // emerald
    metricLabel = "k-NN Label Consistency (%)";
    isPercentage = true;
  }

  const minVal = Math.min(...series, 0);
  const maxVal = Math.max(...series, 1);
  const valRange = maxVal - minVal || 1;

  const scaleX = (idx: number) => {
    if (numLayers <= 1) return width / 2;
    return padding + (idx / (numLayers - 1)) * (width - 2 * padding);
  };

  const scaleY = (val: number) => {
    return height - padding - ((val - minVal) / valRange) * (height - 2 * padding);
  };

  // Generate SVG polyline path
  const pointsString = series
    .map((val, idx) => `${scaleX(idx)},${scaleY(val)}`)
    .join(" ");

  return (
    <div className="space-y-4">
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-5 space-y-4">
        {/* Header & Tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-3 border-b border-zinc-800/80">
          <div>
            <h2 className="text-sm font-bold font-mono text-zinc-100 flex items-center gap-2">
              <span>Layer-Wise Representation Geometry Evolution</span>
              <span className="text-[10px] text-purple-400 bg-purple-950/60 px-1.5 py-0.5 rounded border border-purple-800/40">
                {architecture.toUpperCase()} &bull; {numLayers} Probed Layers
              </span>
            </h2>
            <p className="text-xs text-zinc-400">
              Active Metric: <span className="text-zinc-200 font-mono font-bold">{metricLabel}</span> across network depth
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-mono">
            <button
              onClick={() => setActiveMetric("ratio")}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === "ratio"
                  ? "bg-purple-950 text-purple-300 border border-purple-800 font-bold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Sep/Comp Ratio
            </button>
            <button
              onClick={() => setActiveMetric("compactness")}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === "compactness"
                  ? "bg-cyan-950 text-cyan-300 border border-cyan-800 font-bold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Compactness
            </button>
            <button
              onClick={() => setActiveMetric("separation")}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === "separation"
                  ? "bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Separation
            </button>
            <button
              onClick={() => setActiveMetric("consistency")}
              className={`px-2.5 py-1 rounded transition-colors ${
                activeMetric === "consistency"
                  ? "bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Consistency
            </button>
          </div>
        </div>

        {/* SVG Chart */}
        <div className="relative w-full overflow-x-auto">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-h-[300px] select-none">
            {/* Grid Lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
              const yVal = minVal + frac * valRange;
              const yPos = scaleY(yVal);
              return (
                <g key={`grid-${frac}`}>
                  <line
                    x1={padding}
                    y1={yPos}
                    x2={width - padding}
                    y2={yPos}
                    stroke="#27272a"
                    strokeDasharray="2 2"
                    strokeWidth="1"
                  />
                  <text
                    x={padding - 8}
                    y={yPos + 3}
                    fill="#71717a"
                    fontSize="9"
                    fontFamily="monospace"
                    textAnchor="end"
                  >
                    {isPercentage ? `${yVal.toFixed(0)}%` : yVal.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* Line Trend */}
            <polyline
              fill="none"
              stroke={seriesColor}
              strokeWidth="2.5"
              points={pointsString}
            />

            {/* Data Points */}
            {series.map((val, idx) => {
              const cx = scaleX(idx);
              const cy = scaleY(val);
              const layer = layer_points[idx];

              return (
                <g
                  key={`layer-pt-${idx}`}
                  className="cursor-pointer"
                  onClick={() => onSelectLayer && onSelectLayer(layer.layer_name)}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={5}
                    fill="#09090b"
                    stroke={seriesColor}
                    strokeWidth="2.5"
                  />
                  {/* Layer Label on X axis */}
                  <text
                    x={cx}
                    y={height - padding + 18}
                    fill="#a1a1aa"
                    fontSize="9"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {layer.layer_name}
                  </text>
                  {/* Value tag */}
                  <text
                    x={cx}
                    y={cy - 10}
                    fill={seriesColor}
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {isPercentage ? `${val.toFixed(0)}%` : val.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Layer Points Table */}
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-4 space-y-3">
        <h3 className="text-xs font-mono font-bold text-zinc-200">
          Probed Layer Geometry Progression
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-zinc-800 text-[10px] text-zinc-400 uppercase">
                <th className="pb-2">Depth</th>
                <th className="pb-2">Layer Name</th>
                <th className="pb-2">Dim (D)</th>
                <th className="pb-2">Intra Compactness (d̄)</th>
                <th className="pb-2">Inter Separation (Δ)</th>
                <th className="pb-2">Sep / Comp Ratio</th>
                <th className="pb-2">k-NN Consistency</th>
                <th className="pb-2">PCA Var (2D)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {layer_points.map((pt) => (
                <tr
                  key={pt.layer_name}
                  onClick={() => onSelectLayer && onSelectLayer(pt.layer_name)}
                  className="hover:bg-zinc-800/30 transition-colors cursor-pointer"
                >
                  <td className="py-2 text-zinc-500 font-bold">L{pt.depth_index}</td>
                  <td className="py-2 text-cyan-300 font-medium">{pt.layer_name}</td>
                  <td className="py-2 text-zinc-400">{pt.feature_dim}</td>
                  <td className="py-2 text-cyan-200">{pt.mean_intra_class_distance.toFixed(3)}</td>
                  <td className="py-2 text-indigo-200">
                    {pt.mean_inter_class_centroid_distance.toFixed(3)}
                  </td>
                  <td className="py-2 text-purple-300 font-bold">
                    {pt.separation_to_compactness_ratio.toFixed(2)}x
                  </td>
                  <td className="py-2 text-emerald-300">
                    {(pt.mean_label_consistency * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 text-amber-300">
                    {(pt.pca_first_two_variance_ratio * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
