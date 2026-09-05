"use client";

import React from "react";
import { OODBinaryEvaluationSummaryPayload, UncertaintySampleItemPayload } from "../types";

interface OODROCCardProps {
  evaluation?: OODBinaryEvaluationSummaryPayload;
  activeOODEval?: OODBinaryEvaluationSummaryPayload;
  selectedScoreMethod?: string;
  selectedMethod?: string;
  samples?: UncertaintySampleItemPayload[];
}

export const OODROCCard: React.FC<OODROCCardProps> = ({
  evaluation,
  activeOODEval,
  selectedScoreMethod,
  selectedMethod,
}) => {
  const currentEval = evaluation || activeOODEval;
  const currentMethod = selectedScoreMethod || selectedMethod || "msp";

  if (!currentEval) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h2 className="text-sm font-bold text-slate-100">OOD Discrimination ROC Curve</h2>
        <p className="text-xs text-slate-400 mt-2">Evaluation not available.</p>
      </div>
    );
  }

  const aurocVal = currentEval.auroc;
  const aurocPercent = (aurocVal * 100).toFixed(1);

  // Approximate parametric ROC curve points for visualization based on AUROC
  const svgWidth = 360;
  const svgHeight = 280;
  const padding = { top: 20, right: 20, bottom: 40, left: 45 };
  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  // Generate smooth convex ROC points
  // Shape parameterized by AUROC: TPR = FPR^( (1-A)/A )
  const power = Math.max(0.05, (1.0 - aurocVal) / Math.max(aurocVal, 0.01));
  const numPts = 40;
  const rocPoints: Array<[number, number]> = [];
  for (let step = 0; step <= numPts; step++) {
    const fpr = step / numPts;
    const tpr = Math.min(Math.pow(fpr, power), 1.0);
    rocPoints.push([fpr, tpr]);
  }

  const pathD = rocPoints
    .map(([fpr, tpr], i) => {
      const x = padding.left + fpr * plotWidth;
      const y = padding.top + (1 - tpr) * plotHeight;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  const opX = padding.left + currentEval.fpr_at_threshold * plotWidth;
  const opY = padding.top + (1 - currentEval.tpr_at_threshold) * plotHeight;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              OOD Binary Discrimination ROC
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800/40">
              AUROC = {aurocPercent}%
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Trade-off for {currentMethod.replace(/_/g, " ").toUpperCase()} score • TPR vs False Alarm Rate (FPR)
          </p>
        </div>
      </div>

      {/* ROC Chart */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 my-4 items-center">
        <div className="sm:col-span-7 flex justify-center bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/70">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full max-w-[340px] h-auto select-none">
            {/* Grid */}
            {[0.0, 0.25, 0.5, 0.75, 1.0].map((v) => {
              const y = padding.top + plotHeight * (1 - v);
              const x = padding.left + plotWidth * v;
              return (
                <g key={v}>
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={padding.left + plotWidth}
                    y2={y}
                    stroke="#1e293b"
                    strokeDasharray="2,2"
                  />
                  <line
                    x1={x}
                    y1={padding.top}
                    x2={x}
                    y2={padding.top + plotHeight}
                    stroke="#1e293b"
                    strokeDasharray="2,2"
                  />
                  <text
                    x={padding.left - 6}
                    y={y + 3}
                    textAnchor="end"
                    fontSize="9"
                    fill="#64748b"
                    className="font-mono"
                  >
                    {(v * 100).toFixed(0)}%
                  </text>
                  <text
                    x={x}
                    y={padding.top + plotHeight + 14}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#64748b"
                    className="font-mono"
                  >
                    {(v * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}

            {/* Random chance line */}
            <line
              x1={padding.left}
              y1={padding.top + plotHeight}
              x2={padding.left + plotWidth}
              y2={padding.top}
              stroke="#475569"
              strokeDasharray="3,3"
            />

            {/* Area under curve fill */}
            <path
              d={`${pathD} L ${padding.left + plotWidth} ${padding.top + plotHeight} L ${padding.left} ${padding.top + plotHeight} Z`}
              fill="rgba(245, 158, 11, 0.12)"
            />

            {/* ROC Curve line */}
            <path d={pathD} fill="none" stroke="#f59e0b" strokeWidth="2.5" />

            {/* Operating Point */}
            <circle cx={opX} cy={opY} r="5" fill="#f43f5e" stroke="#fff" strokeWidth="1.5" />

            {/* Axis titles */}
            <text
              x={padding.left + plotWidth / 2}
              y={svgHeight - 4}
              textAnchor="middle"
              fontSize="10"
              fontWeight="600"
              fill="#94a3b8"
            >
              False Positive Rate (FPR)
            </text>
            <text
              transform={`rotate(-90 ${padding.left - 28} ${padding.top + plotHeight / 2})`}
              x={padding.left - 28}
              y={padding.top + plotHeight / 2}
              textAnchor="middle"
              fontSize="10"
              fontWeight="600"
              fill="#94a3b8"
            >
              True Positive Rate (TPR)
            </text>
          </svg>
        </div>

        {/* Threshold Metrics */}
        <div className="sm:col-span-5 flex flex-col gap-2.5">
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] uppercase font-mono text-slate-400">Target ID TPR</div>
            <div className="text-base font-bold font-mono text-cyan-400">95.0% Acceptance</div>
            <div className="text-[10px] text-slate-500 font-mono">Policy: TARGET_ID_TPR</div>
          </div>

          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <div className="text-[10px] uppercase font-mono text-slate-400">Threshold Performance</div>
            <div className="text-xs text-slate-300 font-mono mt-1 space-y-0.5">
              <div className="flex justify-between">
                <span className="text-slate-400">Decision Threshold:</span>
                <span className="font-bold text-rose-400">θ = {currentEval.threshold.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">OOD Recall (TPR):</span>
                <span className="font-bold text-emerald-400">{(currentEval.tpr_at_threshold * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">False Alarm (FPR):</span>
                <span className="font-bold text-amber-400">{(currentEval.fpr_at_threshold * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Detection Accuracy:</span>
                <span className="font-bold text-indigo-400">{(currentEval.detection_accuracy_at_threshold * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
