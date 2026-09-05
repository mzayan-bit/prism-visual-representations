"use client";

import React from "react";
import { TradeoffPointPayload } from "../../benchmarkData";

interface TradeoffExplorerCardProps {
  tradeoffs: TradeoffPointPayload[];
}

export const TradeoffExplorerCard: React.FC<TradeoffExplorerCardProps> = ({
  tradeoffs,
}) => {
  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div>
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>⚖️</span> Tradeoff & Robustness Scatter Explorer
        </h3>
        <p className="text-xs text-slate-400">
          In-Distribution Accuracy vs Perturbation Accuracy Drop (lower drop = more robust)
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {tradeoffs.map((pt, idx) => {
          const arch = String(pt.factors?.architecture || "model").toUpperCase();
          const obj = String(pt.factors?.pretraining_objective || "supervised");

          return (
            <div
              key={idx}
              className="p-3.5 bg-slate-950 rounded-xl border border-slate-800/80 flex flex-col justify-between space-y-2 hover:border-slate-700 transition-all font-mono"
            >
              <div className="flex items-center justify-between border-b border-slate-800/60 pb-1.5">
                <span className="text-xs font-bold text-cyan-300">
                  {arch} • <span className="capitalize text-slate-300">{obj}</span>
                </span>
                <span className="text-[10px] text-slate-500">{pt.experiment_id}</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-[10px] text-slate-400 block">Clean Acc (X)</span>
                  <span className="text-emerald-400 font-bold font-mono text-sm">
                    {(pt.x_value * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block">Robustness Drop (Y)</span>
                  <span className="text-amber-400 font-bold font-mono text-sm">
                    {(pt.y_value * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="text-[9px] text-slate-500 pt-1 border-t border-slate-900">
                {pt.note}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
