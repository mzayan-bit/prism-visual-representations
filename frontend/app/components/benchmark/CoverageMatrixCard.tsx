"use client";

import React, { useState } from "react";
import { CoverageMatrixPayload } from "../../benchmarkData";

interface CoverageMatrixCardProps {
  coverageMatrix: CoverageMatrixPayload;
  onSelectCell?: (row: string, col: string) => void;
}

export const CoverageMatrixCard: React.FC<CoverageMatrixCardProps> = ({
  coverageMatrix,
  onSelectCell,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{ row: string; col: string } | null>(null);

  const getStatusColor = (counts: Record<string, number>) => {
    const observed = counts["observed"] || 0;
    const aggregated = counts["aggregated"] || 0;
    const missing = counts["missing"] || 0;
    const failed = counts["failed"] || 0;
    const na = counts["not_applicable"] || 0;

    if (failed > 0) return "bg-rose-950/80 border-rose-600 text-rose-300";
    if (observed > 0 || aggregated > 0) return "bg-emerald-950/70 border-emerald-500 text-emerald-300 hover:border-emerald-300";
    if (na > 0) return "bg-slate-900 border-slate-700 text-slate-500";
    if (missing > 0) return "bg-amber-950/60 border-amber-600 text-amber-300";
    return "bg-slate-900 border-slate-800 text-slate-400";
  };

  const getStatusLabel = (counts: Record<string, number>) => {
    const observed = counts["observed"] || 0;
    const aggregated = counts["aggregated"] || 0;
    const failed = counts["failed"] || 0;
    const na = counts["not_applicable"] || 0;

    if (failed > 0) return "FAILED";
    if (observed > 0 || aggregated > 0) return `${observed + aggregated} OBS`;
    if (na > 0) return "N/A";
    return "MISSING";
  };

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>🗺️</span> 2D Experimental Coverage Grid
          </h3>
          <p className="text-xs text-slate-400">
            Observation status across {coverageMatrix.row_factor} vs {coverageMatrix.column_factor}
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-emerald-500 inline-block"></span> Observed
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-amber-500 inline-block"></span> Missing
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-slate-700 inline-block"></span> N/A
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800">
              <th className="p-2.5 text-slate-400 font-mono uppercase text-[11px] bg-slate-950/60 rounded-tl-lg">
                Objective \ Arch
              </th>
              {coverageMatrix.column_values.map((col) => (
                <th
                  key={col}
                  className="p-2.5 text-center text-slate-300 font-mono uppercase text-[11px] bg-slate-950/60"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {coverageMatrix.row_values.map((row) => (
              <tr key={row} className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors">
                <td className="p-2.5 font-mono text-slate-200 capitalize font-medium">
                  {row}
                </td>
                {coverageMatrix.column_values.map((col) => {
                  const counts = coverageMatrix.grid[row]?.[col] || {};
                  const isHovered = hoveredCell?.row === row && hoveredCell?.col === col;

                  return (
                    <td key={col} className="p-1.5 text-center">
                      <button
                        onClick={() => onSelectCell && onSelectCell(row, col)}
                        onMouseEnter={() => setHoveredCell({ row, col })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className={`w-full py-2 px-3 rounded-lg border text-[11px] font-mono font-bold transition-all ${getStatusColor(
                          counts
                        )} ${isHovered ? "scale-105 shadow-md shadow-cyan-500/10" : ""}`}
                      >
                        {getStatusLabel(counts)}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {hoveredCell && (
        <div className="p-2.5 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 flex items-center justify-between animate-fadeIn">
          <span>
            Selected Combo: <strong className="text-cyan-400">{hoveredCell.row}</strong> ×{" "}
            <strong className="text-cyan-400">{hoveredCell.col}</strong>
          </span>
          <span>
            Observed:{" "}
            <strong className="text-emerald-400">
              {coverageMatrix.grid[hoveredCell.row]?.[hoveredCell.col]?.["observed"] || 0}
            </strong>{" "}
            cells
          </span>
        </div>
      )}
    </div>
  );
};
