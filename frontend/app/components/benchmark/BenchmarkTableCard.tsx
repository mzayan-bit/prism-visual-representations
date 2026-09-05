"use client";

import React, { useState } from "react";
import { BenchmarkTablePayload } from "../../benchmarkData";

interface BenchmarkTableCardProps {
  tables: BenchmarkTablePayload[];
  onSelectCell?: (info: Record<string, unknown>) => void;
}

export const BenchmarkTableCard: React.FC<BenchmarkTableCardProps> = ({
  tables,
  onSelectCell,
}) => {
  const [selectedTableIdx, setSelectedTableIdx] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>("");

  if (!tables || tables.length === 0) {
    return (
      <div className="p-6 bg-slate-900/90 rounded-2xl border border-slate-800 text-center text-slate-400">
        No benchmark tables available.
      </div>
    );
  }

  const activeTable = tables[selectedTableIdx] || tables[0];
  const rowFactor = activeTable.row_factor;
  const colKeys = activeTable.rows?.[0]
    ? Object.keys(activeTable.rows[0]).filter((k) => k !== rowFactor)
    : [];

  const filteredRows = activeTable.rows.filter((r) => {
    const rowVal = String(r[rowFactor] || "").toLowerCase();
    return rowVal.includes(searchQuery.toLowerCase());
  });

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      {/* Table Selector & Search */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2 overflow-x-auto">
          {tables.map((t, idx) => (
            <button
              key={t.table_id}
              onClick={() => setSelectedTableIdx(idx)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all whitespace-nowrap ${
                selectedTableIdx === idx
                  ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/20"
                  : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
              }`}
            >
              {t.metric_id.replace(/_/g, " ").toUpperCase()}
            </button>
          ))}
        </div>

        <input
          type="text"
          placeholder="Filter rows..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="px-3 py-1.5 bg-slate-950 rounded-lg border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
        />
      </div>

      {/* Header Info */}
      <div className="flex items-center justify-between pt-1">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>📋</span> {activeTable.title}
          </h3>
          <p className="text-xs text-slate-400">
            Unit: <span className="text-cyan-400 font-mono">{activeTable.unit || "scalar"}</span> • Direction:{" "}
            <span className="text-emerald-400 font-mono capitalize">
              {activeTable.metric_direction.replace(/_/g, " ")}
            </span>
          </p>
        </div>
        <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-400">
          Control: <strong className="text-emerald-400">STRICT</strong>
        </span>
      </div>

      {/* Tabular View */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/80">
              <th className="p-3 text-slate-400 font-mono uppercase text-[11px] rounded-tl-lg capitalize">
                {rowFactor.replace(/_/g, " ")}
              </th>
              {colKeys.map((col) => (
                <th
                  key={col}
                  className="p-3 text-center text-slate-300 font-mono uppercase text-[11px]"
                >
                  {col.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((rowDict, rIdx) => {
              const rowVal = String(rowDict[rowFactor]);

              return (
                <tr
                  key={rIdx}
                  className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors"
                >
                  <td className="p-3 font-mono font-bold text-slate-200 capitalize">
                    {rowVal}
                  </td>
                  {colKeys.map((col) => {
                    const cellData = rowDict[col];
                    const isObj =
                      typeof cellData === "object" && cellData !== null;
                    const disp = isObj
                      ? String(
                          (cellData as Record<string, unknown>).display ?? "—"
                        )
                      : String(cellData ?? "—");
                    const val = isObj
                      ? (cellData as Record<string, unknown>).value
                      : cellData;
                    const seedCount = isObj
                      ? Number(
                          (cellData as Record<string, unknown>).seed_count ?? 1
                        )
                      : 1;

                    return (
                      <td key={col} className="p-2 text-center">
                        <button
                          onClick={() =>
                            onSelectCell &&
                            onSelectCell({
                              rowFactor: rowVal,
                              colFactor: col,
                              metric: activeTable.metric_id,
                              value: val,
                              display: disp,
                              seedCount,
                            })
                          }
                          className="w-full py-1.5 px-2.5 rounded-lg bg-slate-950/80 hover:bg-cyan-950/60 border border-slate-800 hover:border-cyan-600 transition-all font-mono text-cyan-300 text-xs font-bold"
                        >
                          <div>{disp}</div>
                          {seedCount > 1 && (
                            <div className="text-[9px] text-slate-500 font-normal">
                              N={seedCount} seeds
                            </div>
                          )}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {activeTable.footnotes && activeTable.footnotes.length > 0 && (
        <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800/80 text-[11px] text-slate-400 font-mono">
          <strong>Methodological Notes:</strong> {activeTable.footnotes.join(" ")}
        </div>
      )}
    </div>
  );
};
