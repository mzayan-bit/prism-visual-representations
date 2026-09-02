"use client";

import React, { useState } from "react";
import { SelfSupervisedLearningReportPayload } from "../types";

interface TrainingDynamicsCardProps {
  report: SelfSupervisedLearningReportPayload;
}

export function TrainingDynamicsCard({ report }: TrainingDynamicsCardProps) {
  const [activeTab, setActiveTab] = useState<"similarities" | "loss">("similarities");

  const epochs = report.epochs;
  const posSims = report.positive_similarity_trajectory;
  const negSims = report.negative_similarity_trajectory;
  const losses = report.loss_trajectory;

  // SVG Coordinates calculation
  const width = 500;
  const height = 180;
  const padX = 40;
  const padY = 25;

  const getPoints = (values: number[], minVal: number, maxVal: number) => {
    return values
      .map((val, idx) => {
        const x = padX + (idx / Math.max(1, epochs - 1)) * (width - 2 * padX);
        const y = height - padY - ((val - minVal) / Math.max(1e-5, maxVal - minVal)) * (height - 2 * padY);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const simPointsPos = getPoints(posSims, 0.0, 1.0);
  const simPointsNeg = getPoints(negSims, 0.0, 1.0);
  const minLoss = Math.min(...losses) * 0.9;
  const maxLoss = Math.max(...losses) * 1.05;
  const lossPoints = getPoints(losses, minLoss, maxLoss);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Self-Supervised Pretraining Dynamics
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Trajectories of instance similarity alignment and NT-Xent loss across epochs.
          </p>
        </div>
        <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1">
          <button
            onClick={() => setActiveTab("similarities")}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
              activeTab === "similarities"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Cosine Similarities
          </button>
          <button
            onClick={() => setActiveTab("loss")}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
              activeTab === "loss"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            NT-Xent Loss
          </button>
        </div>
      </div>

      {/* Telemetry metrics bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-2.5">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Final Pos Similarity</div>
          <div className="text-base font-bold text-emerald-400 font-mono mt-0.5">
            {posSims[posSims.length - 1]?.toFixed(3) ?? "N/A"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-2.5">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Final Neg Similarity</div>
          <div className="text-base font-bold text-rose-400 font-mono mt-0.5">
            {negSims[negSims.length - 1]?.toFixed(3) ?? "N/A"}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-2.5">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Similarity Gap (&Delta;)</div>
          <div className="text-base font-bold text-indigo-400 font-mono mt-0.5">
            {(
              (posSims[posSims.length - 1] ?? 0) - (negSims[negSims.length - 1] ?? 0)
            ).toFixed(3)}
          </div>
        </div>
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-2.5">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Final NT-Xent Loss</div>
          <div className="text-base font-bold text-amber-400 font-mono mt-0.5">
            {losses[losses.length - 1]?.toFixed(3) ?? "N/A"}
          </div>
        </div>
      </div>

      {/* Chart SVG */}
      <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-44">
          {/* Grid lines */}
          <line x1={padX} y1={padY} x2={width - padX} y2={padY} stroke="#334155" strokeDasharray="3,3" strokeWidth="0.8" />
          <line x1={padX} y1={height / 2} x2={width - padX} y2={height / 2} stroke="#334155" strokeDasharray="3,3" strokeWidth="0.8" />
          <line x1={padX} y1={height - padY} x2={width - padX} y2={height - padY} stroke="#475569" strokeWidth="1" />

          {activeTab === "similarities" ? (
            <>
              {/* Positive Similarity Series */}
              <polyline fill="none" stroke="#10b981" strokeWidth="2.5" points={simPointsPos} />
              {/* Negative Similarity Series */}
              <polyline fill="none" stroke="#f43f5e" strokeWidth="2.5" points={simPointsNeg} />
            </>
          ) : (
            <>
              {/* NT-Xent Loss Series */}
              <polyline fill="none" stroke="#f59e0b" strokeWidth="2.5" points={lossPoints} />
            </>
          )}

          {/* Labels */}
          <text x={padX} y={height - 8} fill="#64748b" fontSize="9" fontFamily="monospace">Ep 1</text>
          <text x={width - padX - 25} y={height - 8} fill="#64748b" fontSize="9" fontFamily="monospace">Ep {epochs}</text>
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-2 pt-2 border-t border-slate-800/60 text-xs">
          {activeTab === "similarities" ? (
            <>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-emerald-500 rounded-full inline-block" />
                <span className="text-slate-300">Positive Pairs (Views of Same Image)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-rose-500 rounded-full inline-block" />
                <span className="text-slate-300">Negative Pairs (Distinct Samples)</span>
              </div>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-amber-500 rounded-full inline-block" />
              <span className="text-slate-300">NT-Xent Loss (&tau; = {report.temperature})</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
