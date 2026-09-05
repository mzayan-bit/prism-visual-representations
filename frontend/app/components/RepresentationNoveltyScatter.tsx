"use client";

import React from "react";
import { RepresentationConfidenceRelationshipPayload, UncertaintySampleItemPayload } from "../types";

interface RepresentationNoveltyScatterProps {
  samples: UncertaintySampleItemPayload[];
  selectedSampleId?: string | null;
  onSelectSampleId?: (id: string) => void;
  onSelectSample?: (sample: UncertaintySampleItemPayload) => void;
  relationships?: RepresentationConfidenceRelationshipPayload | null;
  relationship?: RepresentationConfidenceRelationshipPayload | null;
}

export const RepresentationNoveltyScatter: React.FC<
  RepresentationNoveltyScatterProps
> = ({
  samples,
  selectedSampleId,
  onSelectSampleId,
  onSelectSample,
  relationships,
  relationship,
}) => {
  const activeRel = relationship || relationships || null;
  const maxDist = Math.max(...samples.map((s) => s.centroid_distance), 3.5);

  const handlePointClick = (s: UncertaintySampleItemPayload) => {
    if (onSelectSampleId) onSelectSampleId(s.sample_id);
    if (onSelectSample) onSelectSample(s);
  };

  const svgWidth = 560;
  const svgHeight = 340;
  const padding = { top: 25, right: 30, bottom: 45, left: 50 };
  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              Representation Novelty vs Predictive Confidence
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/40">
              GEOMETRY vs PROBABILITY
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Distance to Nearest Class Centroid (X) vs Softmax Max Probability (Y)
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-cyan-400"></div>
            <span className="text-slate-300">Correct ID</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-rose-400"></div>
            <span className="text-slate-300">Incorrect ID</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-amber-400"></div>
            <span className="text-slate-300">OOD</span>
          </div>
        </div>
      </div>

      {/* Scatter Plot */}
      <div className="my-4 flex flex-col items-center bg-slate-950/60 p-3 rounded-xl border border-slate-800/70">
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full max-w-[580px] h-auto select-none overflow-visible">
          {/* Grid Lines */}
          {[0.0, 0.25, 0.5, 0.75, 1.0].map((v) => {
            const y = padding.top + plotHeight * (1 - v);
            return (
              <g key={`y-${v}`}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={padding.left + plotWidth}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="2,2"
                />
                <text
                  x={padding.left - 8}
                  y={y + 3}
                  textAnchor="end"
                  fontSize="9"
                  fill="#64748b"
                  className="font-mono"
                >
                  {(v * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}

          {[0, 1, 2, 3, 4].map((distVal) => {
            if (distVal > maxDist) return null;
            const x = padding.left + (distVal / maxDist) * plotWidth;
            return (
              <g key={`x-${distVal}`}>
                <line
                  x1={x}
                  y1={padding.top}
                  x2={x}
                  y2={padding.top + plotHeight}
                  stroke="#1e293b"
                  strokeDasharray="2,2"
                />
                <text
                  x={x}
                  y={padding.top + plotHeight + 14}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#64748b"
                  className="font-mono"
                >
                  {distVal.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Points */}
          {samples.map((s) => {
            const cx = padding.left + (Math.min(s.centroid_distance, maxDist) / maxDist) * plotWidth;
            const cy = padding.top + (1.0 - s.confidence) * plotHeight;
            const isSelected = selectedSampleId === s.sample_id;

            let fillColor = "#f59e0b"; // OOD
            if (
              s.category === "IN_DISTRIBUTION" ||
              s.category === "in_distribution"
            ) {
              fillColor = s.is_correct ? "#06b6d4" : "#f43f5e";
            }

            return (
              <g
                key={s.sample_id}
                onClick={() => handlePointClick(s)}
                className="cursor-pointer group"
              >
                {isSelected && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r="8"
                    fill="none"
                    stroke="#ffffff"
                    strokeWidth="2"
                    className="animate-ping opacity-60"
                  />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? "6" : "4.5"}
                  fill={fillColor}
                  stroke={isSelected ? "#fff" : "#0f172a"}
                  strokeWidth={isSelected ? "2" : "1"}
                  className="transition-transform group-hover:scale-125"
                />
              </g>
            );
          })}

          {/* Axis Titles */}
          <text
            x={padding.left + plotWidth / 2}
            y={svgHeight - 4}
            textAnchor="middle"
            fontSize="10"
            fontWeight="600"
            fill="#94a3b8"
          >
            Distance to Nearest Class Centroid ||h - μ_c||
          </text>
          <text
            transform={`rotate(-90 ${padding.left - 30} ${padding.top + plotHeight / 2})`}
            x={padding.left - 30}
            y={padding.top + plotHeight / 2}
            textAnchor="middle"
            fontSize="10"
            fontWeight="600"
            fill="#94a3b8"
          >
            Predictive Confidence p_max
          </text>
        </svg>
      </div>

      {/* Geometry Relationship Readout */}
      {activeRel && (
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400">Centroid r(dist, conf)</div>
            <div className="text-sm font-bold text-cyan-400 mt-0.5">
              {activeRel.centroid_distance_pearson_correlation !== null
                ? activeRel.centroid_distance_pearson_correlation.toFixed(3)
                : "N/A"}
            </div>
          </div>
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400">kNN r(dist, conf)</div>
            <div className="text-sm font-bold text-cyan-400 mt-0.5">
              {activeRel.knn_distance_pearson_correlation !== null
                ? activeRel.knn_distance_pearson_correlation.toFixed(3)
                : "N/A"}
            </div>
          </div>
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400">Correct Mean Dist</div>
            <div className="text-sm font-bold text-emerald-400 mt-0.5">
              {activeRel.correct_mean_centroid_distance.toFixed(3)}
            </div>
          </div>
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] text-slate-400">Error Mean Dist</div>
            <div className="text-sm font-bold text-rose-400 mt-0.5">
              {activeRel.incorrect_mean_centroid_distance.toFixed(3)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
