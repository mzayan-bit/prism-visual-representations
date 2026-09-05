"use client";

import React, { useState } from "react";
import { CalibrationReportPayload } from "../types";

interface ReliabilityDiagramCardProps {
  report: CalibrationReportPayload;
  calibrationMode?: string;
  mode?: string;
}

export const ReliabilityDiagramCard: React.FC<ReliabilityDiagramCardProps> = ({
  report,
  calibrationMode,
  mode,
}) => {
  const activeMode = calibrationMode || mode || "uncalibrated";
  const [selectedBinIndex, setSelectedBinIndex] = useState<number | null>(null);

  const bins = report.reliability_bins;
  const maxCount = Math.max(...bins.map((b) => b.sample_count), 1);

  // SVG dimensions
  const svgWidth = 460;
  const svgHeight = 320;
  const padding = { top: 20, right: 20, bottom: 40, left: 45 };
  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  const binWidth = plotWidth / bins.length;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              Empirical Reliability Diagram
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/40 uppercase">
              {activeMode.replace(/_/g, " ")} • {bins.length} BINS
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Mean Softmax Confidence vs Empirical Accuracy • Ideal Calibration: y = x
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-cyan-500 rounded-sm"></div>
            <span className="text-slate-300">Empirical Acc</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 bg-rose-500/40 border border-rose-500/80 rounded-sm"></div>
            <span className="text-slate-300">Calibration Gap</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-slate-400 border-t border-dashed"></div>
            <span className="text-slate-400">Ideal (y=x)</span>
          </div>
        </div>
      </div>

      {/* Main Diagram Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 my-4">
        {/* SVG Chart */}
        <div className="lg:col-span-7 flex flex-col items-center justify-center bg-slate-950/60 p-3 rounded-xl border border-slate-800/70">
          <svg
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="w-full max-w-[480px] h-auto overflow-visible select-none"
          >
            {/* Grid lines */}
            {[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v) => {
              const y = padding.top + plotHeight * (1 - v);
              const x = padding.left + plotWidth * v;
              return (
                <g key={v}>
                  {/* Horizontal grid */}
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={padding.left + plotWidth}
                    y2={y}
                    stroke="#1e293b"
                    strokeDasharray="2,2"
                  />
                  {/* Vertical grid */}
                  <line
                    x1={x}
                    y1={padding.top}
                    x2={x}
                    y2={padding.top + plotHeight}
                    stroke="#1e293b"
                    strokeDasharray="2,2"
                  />
                  {/* Y Axis Label */}
                  <text
                    x={padding.left - 8}
                    y={y + 3}
                    textAnchor="end"
                    fontSize="10"
                    fill="#64748b"
                    className="font-mono"
                  >
                    {(v * 100).toFixed(0)}%
                  </text>
                  {/* X Axis Label */}
                  <text
                    x={x}
                    y={padding.top + plotHeight + 16}
                    textAnchor="middle"
                    fontSize="10"
                    fill="#64748b"
                    className="font-mono"
                  >
                    {(v * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}

            {/* Ideal diagonal line y = x */}
            <line
              x1={padding.left}
              y1={padding.top + plotHeight}
              x2={padding.left + plotWidth}
              y2={padding.top}
              stroke="#94a3b8"
              strokeWidth="2"
              strokeDasharray="4,4"
            />

            {/* Bars for Each Bin */}
            {bins.map((bin, i) => {
              const x = padding.left + i * binWidth;
              const barAccHeight = plotHeight * bin.empirical_accuracy;
              const barConfHeight = plotHeight * bin.mean_confidence;
              const yAcc = padding.top + plotHeight - barAccHeight;
              const yConf = padding.top + plotHeight - barConfHeight;
              const isSelected = selectedBinIndex === i;

              if (bin.sample_count === 0) {
                return (
                  <g key={i}>
                    {/* Empty bin indicator */}
                    <rect
                      x={x + 2}
                      y={padding.top}
                      width={binWidth - 4}
                      height={plotHeight}
                      fill="transparent"
                      stroke="#334155"
                      strokeDasharray="2,2"
                      className="cursor-pointer opacity-30 hover:opacity-60"
                      onClick={() => setSelectedBinIndex(i)}
                    />
                  </g>
                );
              }

              return (
                <g
                  key={i}
                  className="cursor-pointer group"
                  onClick={() => setSelectedBinIndex(i)}
                >
                  {/* Calibration Gap Box */}
                  <rect
                    x={x + 3}
                    y={Math.min(yAcc, yConf)}
                    width={binWidth - 6}
                    height={Math.max(Math.abs(barAccHeight - barConfHeight), 2)}
                    fill="rgba(244, 63, 94, 0.35)"
                    stroke="rgba(244, 63, 94, 0.8)"
                    strokeWidth="1"
                    className="transition-all group-hover:brightness-125"
                  />

                  {/* Empirical Accuracy Bar */}
                  <rect
                    x={x + 3}
                    y={yAcc}
                    width={binWidth - 6}
                    height={barAccHeight}
                    fill={isSelected ? "#06b6d4" : "#0891b2"}
                    stroke={isSelected ? "#67e8f9" : "#0e7490"}
                    strokeWidth={isSelected ? 2 : 1}
                    rx="2"
                    className="transition-all group-hover:brightness-110"
                  />

                  {/* Mean Confidence Marker Line */}
                  <line
                    x1={x + 1}
                    y1={yConf}
                    x2={x + binWidth - 1}
                    y2={yConf}
                    stroke="#f59e0b"
                    strokeWidth="2.5"
                  />
                </g>
              );
            })}

            {/* Axis Titles */}
            <text
              x={padding.left + plotWidth / 2}
              y={svgHeight - 4}
              textAnchor="middle"
              fontSize="11"
              fontWeight="600"
              fill="#94a3b8"
            >
              Mean Confidence per Bin
            </text>
            <text
              transform={`rotate(-90 ${padding.left - 28} ${padding.top + plotHeight / 2})`}
              x={padding.left - 28}
              y={padding.top + plotHeight / 2}
              textAnchor="middle"
              fontSize="11"
              fontWeight="600"
              fill="#94a3b8"
            >
              Empirical Accuracy
            </text>
          </svg>

          {/* Sub-strip: Sample Count per Bin */}
          <div className="w-full mt-3 pt-2 border-t border-slate-800">
            <div className="flex justify-between items-center text-[10px] text-slate-400 mb-1">
              <span>Sample Density per Bin:</span>
              <span className="font-mono">Total N = {report.sample_count}</span>
            </div>
            <div className="grid grid-cols-10 gap-1 h-8 items-end bg-slate-900/60 p-1 rounded-lg border border-slate-800">
              {bins.map((b, idx) => {
                const heightPct = (b.sample_count / maxCount) * 100;
                return (
                  <div
                    key={idx}
                    onClick={() => setSelectedBinIndex(idx)}
                    title={`Bin ${idx}: N=${b.sample_count}, Conf=${(b.mean_confidence * 100).toFixed(1)}%, Acc=${(b.empirical_accuracy * 100).toFixed(1)}%`}
                    className={`rounded-t transition-all cursor-pointer ${
                      selectedBinIndex === idx
                        ? "bg-cyan-400"
                        : b.sample_count > 0
                        ? "bg-slate-600 hover:bg-slate-400"
                        : "bg-slate-800/40"
                    }`}
                    style={{ height: `${Math.max(heightPct, 8)}%` }}
                  />
                );
              })}
            </div>
          </div>
        </div>

        {/* Reliability Bins Detail Table */}
        <div className="lg:col-span-5 flex flex-col justify-between">
          <div className="bg-slate-950/40 rounded-xl border border-slate-800 overflow-hidden">
            <div className="p-2.5 bg-slate-900/80 border-b border-slate-800 flex items-center justify-between text-xs font-bold text-slate-200">
              <span>Bin Partition Diagnostics</span>
              <span className="text-[10px] text-amber-400 font-mono">
                ECE = {(report.ece * 100).toFixed(2)}%
              </span>
            </div>
            <div className="max-h-[220px] overflow-y-auto text-xs font-mono">
              <table className="w-full text-left">
                <thead className="text-[10px] uppercase text-slate-400 bg-slate-900/60 sticky top-0">
                  <tr>
                    <th className="p-2">Bin</th>
                    <th className="p-2">Range</th>
                    <th className="p-2">Count</th>
                    <th className="p-2">Acc</th>
                    <th className="p-2">Conf</th>
                    <th className="p-2">Gap</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {bins.map((b, idx) => {
                    const isSelected = selectedBinIndex === idx;
                    const gapColor =
                      b.calibration_gap > 0.15
                        ? "text-rose-400"
                        : b.calibration_gap > 0.05
                        ? "text-amber-400"
                        : "text-emerald-400";

                    return (
                      <tr
                        key={idx}
                        onClick={() => setSelectedBinIndex(idx)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-cyan-950/60 text-cyan-200"
                            : "hover:bg-slate-800/40 text-slate-300"
                        }`}
                      >
                        <td className="p-2 font-bold">{idx + 1}</td>
                        <td className="p-2 text-slate-400 text-[11px]">
                          [{b.lower_bound.toFixed(2)}, {b.upper_bound.toFixed(2)}]
                        </td>
                        <td className="p-2">{b.sample_count}</td>
                        <td className="p-2">
                          {b.sample_count > 0
                            ? `${(b.empirical_accuracy * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="p-2">
                          {b.sample_count > 0
                            ? `${(b.mean_confidence * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className={`p-2 font-bold ${gapColor}`}>
                          {b.sample_count > 0
                            ? `${(b.calibration_gap * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Overconfidence Summary Card */}
          {(() => {
            const correctSummary = report.correct_subset_summary || report.correct_predictions_summary || {
              sample_count: 0,
              mean_max_probability: 0,
              median_max_probability: 0,
              mean_entropy: 0,
              mean_normalized_entropy: 0,
            };
            const errorSummary = report.error_subset_summary || report.incorrect_predictions_summary || {
              sample_count: 0,
              mean_max_probability: 0,
              median_max_probability: 0,
              mean_entropy: 0,
              mean_normalized_entropy: 0,
            };
            return (
              <div className="mt-3 p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs">
                <div className="text-[11px] font-bold text-slate-300 mb-1 flex items-center justify-between">
                  <span>Overconfidence Diagnostic:</span>
                  <span className="text-rose-400 font-mono">
                    Error Conf = {(errorSummary.mean_max_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  When the model makes an incorrect prediction, its mean confidence is{" "}
                  <strong className="text-rose-400">
                    {(errorSummary.mean_max_probability * 100).toFixed(1)}%
                  </strong>{" "}
                  (vs <strong className="text-emerald-400">
                    {(correctSummary.mean_max_probability * 100).toFixed(1)}%
                  </strong>{" "}
                  on correct inputs), demonstrating significant overconfidence on mistakes.
                </p>
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
};
