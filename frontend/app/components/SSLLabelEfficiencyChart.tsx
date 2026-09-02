"use client";

import React from "react";
import { SSLLabelEfficiencyPointPayload } from "../types";

interface SSLLabelEfficiencyChartProps {
  points: SSLLabelEfficiencyPointPayload[];
}

export function SSLLabelEfficiencyChart({ points }: SSLLabelEfficiencyChartProps) {
  const width = 500;
  const height = 180;
  const padX = 45;
  const padY = 25;

  const minAcc = 0.2;
  const maxAcc = 1.0;

  const getPoints = (accessor: (pt: SSLLabelEfficiencyPointPayload) => number) => {
    return points
      .map((pt, idx) => {
        const x = padX + (idx / Math.max(1, points.length - 1)) * (width - 2 * padX);
        const y = height - padY - ((accessor(pt) - minAcc) / (maxAcc - minAcc)) * (height - 2 * padY);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const sslPoints = getPoints((p) => p.ssl_accuracy);
  const supPoints = getPoints((p) => p.supervised_accuracy);
  const scratchPoints = getPoints((p) => p.scratch_accuracy);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Target Label-Efficiency Scaling
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Downstream linear probe accuracy across restricted target label fractions (10% to 100%).
          </p>
        </div>
      </div>

      <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-44">
          {/* Grid lines */}
          <line x1={padX} y1={padY} x2={width - padX} y2={padY} stroke="#334155" strokeDasharray="3,3" strokeWidth="0.8" />
          <line x1={padX} y1={height / 2} x2={width - padX} y2={height / 2} stroke="#334155" strokeDasharray="3,3" strokeWidth="0.8" />
          <line x1={padX} y1={height - padY} x2={width - padX} y2={height - padY} stroke="#475569" strokeWidth="1" />

          {/* Lines */}
          <polyline fill="none" stroke="#10b981" strokeWidth="2.5" points={supPoints} />
          <polyline fill="none" stroke="#6366f1" strokeWidth="2.5" points={sslPoints} />
          <polyline fill="none" stroke="#94a3b8" strokeWidth="2.0" strokeDasharray="4,4" points={scratchPoints} />

          {/* Dots */}
          {points.map((pt, idx) => {
            const x = padX + (idx / Math.max(1, points.length - 1)) * (width - 2 * padX);
            const ySSL = height - padY - ((pt.ssl_accuracy - minAcc) / (maxAcc - minAcc)) * (height - 2 * padY);
            return (
              <circle key={idx} cx={x} cy={ySSL} r="3.5" fill="#6366f1" stroke="#ffffff" strokeWidth="1" />
            );
          })}

          {/* X Axis Labels */}
          {points.map((pt, idx) => {
            const x = padX + (idx / Math.max(1, points.length - 1)) * (width - 2 * padX);
            return (
              <text key={idx} x={x - 10} y={height - 8} fill="#64748b" fontSize="9" fontFamily="monospace">
                {pt.budget_percent_label}
              </text>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-center gap-5 mt-2 pt-2 border-t border-slate-800/60 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-emerald-500 rounded-full inline-block" />
            <span className="text-slate-300">Supervised Encoder</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-indigo-500 rounded-full inline-block" />
            <span className="text-slate-300">SimCLR SSL Encoder</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-slate-400 rounded-full inline-block border-b border-dashed" />
            <span className="text-slate-300">Scratch Baseline</span>
          </div>
        </div>
      </div>
    </div>
  );
}
