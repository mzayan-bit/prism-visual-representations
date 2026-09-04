"use client";

import React from "react";
import {
  TemporalAggregationType,
  TemporalTransferStrategyType,
} from "../types";

interface TemporalHeaderProps {
  architectures: string[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  pretrainingObjectives: string[];
  selectedObjective: string;
  onSelectObjective: (obj: string) => void;
  aggregators: TemporalAggregationType[];
  selectedAggregator: TemporalAggregationType;
  onSelectAggregator: (agg: TemporalAggregationType) => void;
  transferStrategies: TemporalTransferStrategyType[];
  selectedStrategy: TemporalTransferStrategyType;
  onSelectStrategy: (strat: TemporalTransferStrategyType) => void;
  sampleIds: string[];
  selectedSampleId: string;
  onSelectSampleId: (id: string) => void;
}

export const TemporalHeader: React.FC<TemporalHeaderProps> = ({
  architectures,
  selectedArch,
  onSelectArch,
  pretrainingObjectives,
  selectedObjective,
  onSelectObjective,
  aggregators,
  selectedAggregator,
  onSelectAggregator,
  transferStrategies,
  selectedStrategy,
  onSelectStrategy,
  sampleIds,
  selectedSampleId,
  onSelectSampleId,
}) => {
  return (
    <div className="bg-slate-900 border-b border-slate-800 p-4 sticky top-12 z-30 shadow-md">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Title & Badge */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400 font-bold text-lg shadow-inner">
            🎬
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-100">
                Temporal Representation & Aggregation Laboratory
              </h1>
              <span className="bg-amber-950/80 text-amber-400 text-[10px] font-mono font-semibold px-2 py-0.5 rounded border border-amber-500/30">
                PHASE 21
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Frame embeddings, temporal pooling, SimpleRNN BPTT, consistency & sequence transfer
            </p>
          </div>
        </div>

        {/* Control Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Architecture */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Architecture
            </label>
            <select
              value={selectedArch}
              onChange={(e) => onSelectArch(e.target.value)}
              className="bg-slate-950 border border-slate-700 hover:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-medium focus:outline-none focus:border-amber-500 transition-colors"
            >
              {architectures.map((arch) => (
                <option key={arch} value={arch}>
                  {arch.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Pretraining Objective */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Source Objective
            </label>
            <select
              value={selectedObjective}
              onChange={(e) => onSelectObjective(e.target.value)}
              className="bg-slate-950 border border-slate-700 hover:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-medium focus:outline-none focus:border-amber-500 transition-colors"
            >
              {pretrainingObjectives.map((obj) => (
                <option key={obj} value={obj}>
                  {obj.replace("_", " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Temporal Aggregator */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Aggregator
            </label>
            <select
              value={selectedAggregator}
              onChange={(e) => onSelectAggregator(e.target.value as TemporalAggregationType)}
              className="bg-slate-950 border border-slate-700 hover:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-amber-300 font-medium focus:outline-none focus:border-amber-500 transition-colors"
            >
              {aggregators.map((agg) => (
                <option key={agg} value={agg}>
                  {agg.replace(/_/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Transfer Strategy */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Strategy
            </label>
            <select
              value={selectedStrategy}
              onChange={(e) => onSelectStrategy(e.target.value as TemporalTransferStrategyType)}
              className="bg-slate-950 border border-slate-700 hover:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-medium focus:outline-none focus:border-amber-500 transition-colors"
            >
              {transferStrategies.map((strat) => (
                <option key={strat} value={strat}>
                  {strat.replace(/_/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Active Video Sample */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Video Sample
            </label>
            <select
              value={selectedSampleId}
              onChange={(e) => onSelectSampleId(e.target.value)}
              className="bg-slate-950 border border-slate-700 hover:border-slate-600 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-mono focus:outline-none focus:border-cyan-500 transition-colors"
            >
              {sampleIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};
