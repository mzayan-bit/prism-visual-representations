"use client";

import React from "react";
import { TransferExperimentMetaPayload, TransferStrategyType } from "../types";

interface TransferHeaderProps {
  metadata: TransferExperimentMetaPayload;
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  selectedStrategy: TransferStrategyType;
  onSelectStrategy: (strat: TransferStrategyType) => void;
  selectedBudget: number;
  onSelectBudget: (budget: number) => void;
  trainableFraction: number;
}

export function TransferHeader({
  metadata,
  selectedArch,
  onSelectArch,
  selectedStrategy,
  onSelectStrategy,
  selectedBudget,
  onSelectBudget,
  trainableFraction,
}: TransferHeaderProps) {
  const strategyLabels: Record<TransferStrategyType, string> = {
    scratch_baseline: "Scratch Baseline",
    linear_probe: "Linear Probe (Frozen)",
    partial_fine_tune: "Partial Fine-Tune",
    full_fine_tune: "Full Fine-Tune",
  };

  const archLabels: Record<string, string> = {
    cnn: "Convolutional Net (CNN)",
    resnet: "Residual Net (ResNet)",
    vit: "Vision Transformer (ViT)",
  };

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-5 mb-6 shadow-xl">
      {/* Title & Research Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-4 mb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              PHASE 17
            </span>
            <span className="text-xs font-mono text-slate-400">
              EXP: {metadata.experiment_id}
            </span>
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Transfer Learning & Representation Reuse Laboratory
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Empirical investigation of learned representation transferability, parameter freezing policies, and target sample-efficiency.
          </p>
        </div>

        {/* Global Experiment Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5 text-right">
            <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Source Classes (5)
            </div>
            <div className="text-xs font-semibold text-cyan-300 truncate max-w-[200px]">
              {metadata.source_classes.join(", ")}
            </div>
          </div>
          <div className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5 text-right">
            <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Target Task (5 Classes)
            </div>
            <div className="text-xs font-semibold text-emerald-300 truncate max-w-[200px]">
              {metadata.target_classes.join(", ")}
            </div>
          </div>
          <div className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5 text-right">
            <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Trainable Params
            </div>
            <div className="text-xs font-bold text-amber-400 font-mono">
              {(trainableFraction * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Selectors Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* 1. Architecture Selector */}
        <div>
          <label className="block text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
            Model Architecture
          </label>
          <div className="grid grid-cols-3 gap-1.5 bg-slate-950/60 p-1 rounded-lg border border-slate-800">
            {metadata.architectures.map((arch) => {
              const active = selectedArch.toLowerCase() === arch.toLowerCase();
              return (
                <button
                  key={arch}
                  onClick={() => onSelectArch(arch)}
                  title={archLabels[arch] || arch.toUpperCase()}
                  className={`py-1.5 px-2 rounded text-xs font-medium transition-all ${
                    active
                      ? "bg-cyan-600 text-white shadow-md font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  }`}
                >
                  {arch.toUpperCase()}
                </button>
              );
            })}
          </div>
        </div>

        {/* 2. Transfer Strategy Selector */}
        <div>
          <label className="block text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
            Transfer Strategy
          </label>
          <div className="grid grid-cols-2 gap-1.5 bg-slate-950/60 p-1 rounded-lg border border-slate-800">
            {(["scratch_baseline", "linear_probe", "partial_fine_tune", "full_fine_tune"] as TransferStrategyType[]).map(
              (strat) => {
                const active = selectedStrategy === strat;
                return (
                  <button
                    key={strat}
                    onClick={() => onSelectStrategy(strat)}
                    className={`py-1 px-2 rounded text-[11px] transition-all truncate ${
                      active
                        ? "bg-emerald-600 text-white shadow-md font-semibold"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                    }`}
                    title={strategyLabels[strat]}
                  >
                    {strategyLabels[strat]}
                  </button>
                );
              }
            )}
          </div>
        </div>

        {/* 3. Target Data Budget */}
        <div>
          <label className="block text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
            Target Data Budget
          </label>
          <div className="grid grid-cols-4 gap-1.5 bg-slate-950/60 p-1 rounded-lg border border-slate-800">
            {[0.1, 0.25, 0.5, 1.0].map((b) => {
              const active = selectedBudget === b;
              return (
                <button
                  key={b}
                  onClick={() => onSelectBudget(b)}
                  className={`py-1.5 px-2 rounded text-xs font-mono transition-all ${
                    active
                      ? "bg-amber-600 text-white shadow-md font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  }`}
                >
                  {Math.round(b * 100)}%
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
