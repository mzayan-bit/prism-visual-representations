"use client";

import React from "react";
import { SpatialDataEfficiencyRecord } from "../types";

interface SpatialDataEfficiencyCardProps {
  efficiencyRecords: SpatialDataEfficiencyRecord[];
}

export const SpatialDataEfficiencyCard: React.FC<SpatialDataEfficiencyCardProps> = ({
  efficiencyRecords,
}) => {
  if (!efficiencyRecords || efficiencyRecords.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500 font-mono text-xs">
        No spatial data efficiency records available.
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl shadow-slate-950/40">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-base">📈</span>
          <h3 className="text-sm font-bold text-slate-100 font-mono tracking-tight">
            SPATIAL ANNOTATION DATA EFFICIENCY SCALING
          </h3>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Budget: 10% → 100% Target Labels
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-[11px] font-mono text-slate-400">
              <th className="pb-2.5 font-bold">BUDGET FRACTION</th>
              <th className="pb-2.5 text-cyan-400 font-bold">SUPERVISED</th>
              <th className="pb-2.5 text-indigo-400 font-bold">SIMCLR</th>
              <th className="pb-2.5 text-violet-400 font-bold">RECONSTRUCTION</th>
              <th className="pb-2.5 text-slate-400 font-bold">SCRATCH</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {efficiencyRecords.map((rec) => {
              const budgetPct = `${Math.round(rec.budget_fraction * 100)}%`;
              return (
                <tr key={rec.budget_fraction} className="hover:bg-slate-950/60">
                  <td className="py-2.5 font-bold text-slate-200">
                    <span className="bg-slate-800 px-2 py-0.5 rounded text-[11px]">
                      {budgetPct}
                    </span>
                  </td>
                  <td className="py-2.5 text-cyan-300 font-bold">
                    {(rec.supervised_iou * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 text-indigo-300 font-bold">
                    {(rec.simclr_iou * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 text-violet-300 font-bold">
                    {(rec.reconstruction_iou * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 text-slate-400">
                    {(rec.scratch_iou * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
