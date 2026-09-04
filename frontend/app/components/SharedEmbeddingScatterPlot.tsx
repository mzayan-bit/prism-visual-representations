"use client";

import React from "react";
import { MultimodalSamplePayload } from "../types";

interface SharedEmbeddingScatterPlotProps {
  samples: MultimodalSamplePayload[];
  selectedSampleId: string;
  onSelectSampleId: (id: string) => void;
  explainedVarianceRatio: number[];
}

export const SharedEmbeddingScatterPlot: React.FC<SharedEmbeddingScatterPlotProps> = ({
  samples,
  selectedSampleId,
  onSelectSampleId,
  explainedVarianceRatio,
}) => {
  // Determine coordinate bounding box
  const allX = samples.flatMap((s) => [s.image_pca[0], s.text_pca[0]]);
  const allY = samples.flatMap((s) => [s.image_pca[1], s.text_pca[1]]);

  const minX = Math.min(...allX, -1);
  const maxX = Math.max(...allX, 1);
  const minY = Math.min(...allY, -1);
  const maxY = Math.max(...allY, 1);

  const rangeX = Math.max(0.1, maxX - minX);
  const rangeY = Math.max(0.1, maxY - minY);

  const width = 450;
  const height = 300;
  const padding = 35;

  const toSvgX = (val: number) =>
    padding + ((val - minX) / rangeX) * (width - 2 * padding);
  const toSvgY = (val: number) =>
    height - padding - ((val - minY) / rangeY) * (height - 2 * padding);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-3">
      {/* Title & Explained Variance Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-bold text-sm">🌐 Shared PCA Embedding Geometry</span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          Joint PCA: PC1 ({(explainedVarianceRatio[0] * 100).toFixed(1)}%), PC2 (
          {(explainedVarianceRatio[1] * 100).toFixed(1)}%)
        </span>
      </div>

      {/* SVG Canvas */}
      <div className="relative w-full aspect-[3/2] bg-slate-950/80 rounded-lg border border-slate-800/80 overflow-hidden flex items-center justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full select-none"
        >
          {/* Coordinate Grid lines */}
          <line
            x1={padding}
            y1={toSvgY(0)}
            x2={width - padding}
            y2={toSvgY(0)}
            stroke="#334155"
            strokeDasharray="3 3"
            strokeWidth="1"
          />
          <line
            x1={toSvgX(0)}
            y1={padding}
            x2={toSvgX(0)}
            y2={height - padding}
            stroke="#334155"
            strokeDasharray="3 3"
            strokeWidth="1"
          />

          {/* Connection lines between paired image and text */}
          {samples.map((s) => {
            const isSelected = s.sample_id === selectedSampleId;
            const x1 = toSvgX(s.image_pca[0]);
            const y1 = toSvgY(s.image_pca[1]);
            const x2 = toSvgX(s.text_pca[0]);
            const y2 = toSvgY(s.text_pca[1]);

            return (
              <line
                key={`line-${s.sample_id}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={isSelected ? "#38bdf8" : "#475569"}
                strokeWidth={isSelected ? "2" : "1"}
                strokeDasharray={isSelected ? "none" : "2 2"}
                opacity={isSelected ? 1.0 : 0.4}
              />
            );
          })}

          {/* Points */}
          {samples.map((s) => {
            const isSelected = s.sample_id === selectedSampleId;
            const imgX = toSvgX(s.image_pca[0]);
            const imgY = toSvgY(s.image_pca[1]);
            const txtX = toSvgX(s.text_pca[0]);
            const txtY = toSvgY(s.text_pca[1]);

            return (
              <g key={`pair-g-${s.sample_id}`} className="cursor-pointer">
                {/* Image circle */}
                <circle
                  cx={imgX}
                  cy={imgY}
                  r={isSelected ? 6 : 4}
                  fill="#06b6d4"
                  stroke={isSelected ? "#ffffff" : "#0891b2"}
                  strokeWidth={isSelected ? 2 : 1}
                  onClick={() => onSelectSampleId(s.sample_id)}
                >
                  <title>{`Image: ${s.sample_id} (${s.class_name})`}</title>
                </circle>

                {/* Text diamond */}
                <polygon
                  points={`${txtX},${txtY - (isSelected ? 6 : 4)} ${txtX + (isSelected ? 6 : 4)},${txtY} ${txtX},${txtY + (isSelected ? 6 : 4)} ${txtX - (isSelected ? 6 : 4)},${txtY}`}
                  fill="#a855f7"
                  stroke={isSelected ? "#ffffff" : "#9333ea"}
                  strokeWidth={isSelected ? 2 : 1}
                  onClick={() => onSelectSampleId(s.sample_id)}
                >
                  <title>{`Text: "${s.text}"`}</title>
                </polygon>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between text-xs text-slate-400 font-mono px-2">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-500 inline-block" />
            <span>Image Embedding</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 bg-purple-500 transform rotate-45 inline-block" />
            <span>Text Embedding</span>
          </div>
        </div>
        <span className="text-[10px] text-slate-500">Dashed line: Cross-modal pair</span>
      </div>
    </div>
  );
};
