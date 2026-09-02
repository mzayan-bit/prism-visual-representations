"use client";

import React, { useState } from "react";
import { ReconstructionDynamicsPayload } from "../types";

interface ReconstructionDynamicsCardProps {
  dynamics: ReconstructionDynamicsPayload;
}

export function ReconstructionDynamicsCard({
  dynamics,
}: ReconstructionDynamicsCardProps) {
  const [activeMetric, setActiveMetric] = useState<
    "loss" | "masked_mse" | "latent_std"
  >("masked_mse");

  const epochs = dynamics.epochs;
  const values =
    activeMetric === "loss"
      ? dynamics.total_loss
      : activeMetric === "masked_mse"
      ? dynamics.masked_mse
      : dynamics.latent_std;

  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal > 0 ? maxVal - minVal : 1;

  // SVG Chart Dimensions
  const width = 500;
  const height = 180;
  const padding = 30;

  const points = values
    .map((v, i) => {
      const x = padding + (i / (values.length - 1)) * (width - 2 * padding);
      const y =
        height - padding - ((v - minVal) / range) * (height - 2 * padding);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4 pb-3 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Training Dynamics & Convergence
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Epoch trajectory of reconstruction error and latent feature variance.
            </p>
          </div>

          {/* Metric Selector */}
          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 self-start">
            <button
              onClick={() => setActiveMetric("masked_mse")}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                activeMetric === "masked_mse"
                  ? "bg-violet-600 text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Masked MSE
            </button>
            <button
              onClick={() => setActiveMetric("loss")}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                activeMetric === "loss"
                  ? "bg-cyan-600 text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Total Loss
            </button>
            <button
              onClick={() => setActiveMetric("latent_std")}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                activeMetric === "latent_std"
                  ? "bg-emerald-600 text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Latent Std
            </button>
          </div>
        </div>

        {/* SVG Curve Display */}
        <div className="relative bg-slate-950 rounded-lg p-2 border border-slate-800/80 mb-4">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-44 overflow-visible"
          >
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((t) => {
              const y = height - padding - t * (height - 2 * padding);
              const labelVal = minVal + t * range;
              return (
                <g key={t}>
                  <line
                    x1={padding}
                    y1={y}
                    x2={width - padding}
                    y2={y}
                    stroke="#334155"
                    strokeDasharray="2,2"
                    strokeWidth="0.8"
                  />
                  <text
                    x={padding - 5}
                    y={y + 3}
                    fill="#64748b"
                    fontSize="9"
                    textAnchor="end"
                    fontFamily="monospace"
                  >
                    {labelVal.toFixed(3)}
                  </text>
                </g>
              );
            })}

            {/* Sparkline curve */}
            <polyline
              fill="none"
              stroke={
                activeMetric === "masked_mse"
                  ? "#8b5cf6"
                  : activeMetric === "loss"
                  ? "#06b6d4"
                  : "#10b981"
              }
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={points}
            />

            {/* Data points */}
            {values.map((v, i) => {
              const x =
                padding + (i / (values.length - 1)) * (width - 2 * padding);
              const y =
                height -
                padding -
                ((v - minVal) / range) * (height - 2 * padding);
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="3"
                  className={`${
                    activeMetric === "masked_mse"
                      ? "fill-violet-400"
                      : activeMetric === "loss"
                      ? "fill-cyan-400"
                      : "fill-emerald-400"
                  } hover:r-4 transition-all`}
                >
                  <title>{`Epoch ${epochs[i]}: ${v.toFixed(4)}`}</title>
                </circle>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Trajectory KPIs */}
      <div className="grid grid-cols-3 gap-2 text-xs bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
        <div>
          <span className="text-slate-500 block text-[10px] uppercase">Initial Loss</span>
          <span className="text-slate-300 font-mono font-semibold">
            {dynamics.total_loss[0].toFixed(4)}
          </span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px] uppercase">Final Masked MSE</span>
          <span className="text-violet-400 font-mono font-bold">
            {dynamics.masked_mse[dynamics.masked_mse.length - 1].toFixed(4)}
          </span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px] uppercase">Latent Std (&sigma;)</span>
          <span className="text-emerald-400 font-mono font-bold">
            {dynamics.latent_std[dynamics.latent_std.length - 1].toFixed(4)}
          </span>
        </div>
      </div>
    </div>
  );
}
