"use client";

import React from "react";

interface ProvenanceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  selectedInfo: Record<string, unknown> | null;
}

export const ProvenanceDrawer: React.FC<ProvenanceDrawerProps> = ({
  isOpen,
  onClose,
  selectedInfo,
}) => {
  if (!isOpen || !selectedInfo) return null;

  const metric = String(selectedInfo.metric ?? "N/A");
  const display = String(selectedInfo.display ?? selectedInfo.value ?? "N/A");
  const rowFactor = String(selectedInfo.rowFactor ?? "N/A");
  const colFactor = String(selectedInfo.colFactor ?? "N/A");
  const seedCount = Number(selectedInfo.seedCount ?? 1);

  return (
    <div className="fixed inset-y-0 right-0 w-full max-w-md bg-slate-900 border-l border-slate-800 shadow-2xl z-50 flex flex-col justify-between animate-slideLeft">
      <div className="p-5 overflow-y-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <span className="text-base">🧬</span>
            <h3 className="text-sm font-bold text-white">Cryptographic Provenance</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 text-xs font-mono"
          >
            ✕ Close
          </button>
        </div>

        {/* Selected Data Overview */}
        <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
          <div className="flex justify-between">
            <span className="text-slate-400">Metric:</span>
            <strong className="text-cyan-400">{metric}</strong>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Value:</span>
            <strong className="text-emerald-400">{display}</strong>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Row Factor:</span>
            <span className="text-slate-200 capitalize">{rowFactor}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Col Factor:</span>
            <span className="text-slate-200 uppercase">{colFactor}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Evaluated Seeds:</span>
            <span className="text-indigo-400 font-bold">{seedCount}</span>
          </div>
        </div>

        {/* Raw Metadata JSON */}
        <div className="space-y-1.5">
          <span className="text-xs font-mono text-slate-400 font-bold">Raw Cell Metadata:</span>
          <pre className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-[10px] font-mono text-slate-300 overflow-x-auto max-h-64">
            {JSON.stringify(selectedInfo, null, 2)}
          </pre>
        </div>
      </div>

      <div className="p-4 bg-slate-950 border-t border-slate-800">
        <button
          onClick={onClose}
          className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-bold font-mono transition-all"
        >
          Dismiss Inspector
        </button>
      </div>
    </div>
  );
};
