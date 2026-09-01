"use client";

import React from "react";
import { ParameterFreezePlanPayload, TransferStrategyType } from "../types";

interface FreezeMapCardProps {
  freezePlan: ParameterFreezePlanPayload;
  strategy: TransferStrategyType;
  architecture: string;
}

export function FreezeMapCard({
  freezePlan,
  strategy,
}: FreezeMapCardProps) {
  const trainablePct = (freezePlan.trainable_fraction * 100).toFixed(1);
  const frozenPct = ((1 - freezePlan.trainable_fraction) * 100).toFixed(1);

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            Parameter Freeze Architecture Map
          </h3>
          <p className="text-xs text-slate-400">
            Explicit gradient routing and weight decay boundary across layers
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`px-2.5 py-0.5 rounded text-xs font-mono font-medium uppercase ${
              strategy === "linear_probe"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                : strategy === "full_fine_tune"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                : strategy === "partial_fine_tune"
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                : "bg-slate-700/40 text-slate-300 border border-slate-600/40"
            }`}
          >
            {strategy.replace("_", " ")}
          </span>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">
            Total Parameters
          </div>
          <div className="text-sm font-bold text-white font-mono mt-0.5">
            {freezePlan.total_scalar_elements.toLocaleString()}
          </div>
          <div className="text-[10px] text-slate-500">
            {freezePlan.total_tensors} tensor blocks
          </div>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-amber-400">
            Frozen Elements
          </div>
          <div className="text-sm font-bold text-amber-300 font-mono mt-0.5">
            {freezePlan.frozen_scalar_elements.toLocaleString()} ({frozenPct}%)
          </div>
          <div className="text-[10px] text-slate-500">
            {freezePlan.frozen_tensors} tensors frozen
          </div>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-emerald-400">
            Trainable Elements
          </div>
          <div className="text-sm font-bold text-emerald-300 font-mono mt-0.5">
            {freezePlan.trainable_scalar_elements.toLocaleString()} ({trainablePct}%)
          </div>
          <div className="text-[10px] text-slate-500">
            {freezePlan.trainable_tensors} tensors active
          </div>
        </div>
      </div>

      {/* Visual Proportion Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-[11px] font-mono mb-1 text-slate-400">
          <span className="text-amber-400">Frozen: {frozenPct}%</span>
          <span className="text-emerald-400">Trainable: {trainablePct}%</span>
        </div>
        <div className="w-full h-3 rounded-full bg-slate-950 overflow-hidden flex border border-slate-800">
          <div
            style={{ width: `${frozenPct}%` }}
            className="bg-amber-500/80 transition-all duration-500"
            title={`Frozen: ${freezePlan.frozen_scalar_elements} params`}
          />
          <div
            style={{ width: `${trainablePct}%` }}
            className="bg-emerald-500 transition-all duration-500"
            title={`Trainable: ${freezePlan.trainable_scalar_elements} params`}
          />
        </div>
      </div>

      {/* Logical Stage Breakdown List */}
      <div className="flex-1 overflow-y-auto max-h-[260px] pr-1 space-y-2">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-1">
          Stage-by-Stage Gradient Policy
        </div>

        {Object.entries(freezePlan.logical_stages).map(([stageName, paramKeys]) => {
          const isClassifier = stageName === "classifier";
          const allFrozen = paramKeys.every((k) =>
            freezePlan.frozen_parameters.includes(k)
          );
          const allTrainable = paramKeys.every((k) =>
            freezePlan.trainable_parameters.includes(k)
          );

          let statusBadge = (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              MIXED
            </span>
          );

          if (isClassifier) {
            statusBadge = (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                NEW HEAD
              </span>
            );
          } else if (allFrozen) {
            statusBadge = (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-amber-500/20 text-amber-300 border border-amber-500/30">
                FROZEN
              </span>
            );
          } else if (allTrainable) {
            statusBadge = (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                TRAINABLE
              </span>
            );
          }

          return (
            <div
              key={stageName}
              className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80 flex items-center justify-between"
            >
              <div>
                <div className="text-xs font-semibold text-white font-mono capitalize">
                  {stageName.replace("_", " ")}
                </div>
                <div className="text-[10px] text-slate-400 font-mono truncate max-w-[280px]">
                  {paramKeys.join(", ")}
                </div>
              </div>
              <div>{statusBadge}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
