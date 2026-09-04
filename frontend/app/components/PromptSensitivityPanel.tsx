"use client";

import React from "react";
import { PromptSensitivityPayload } from "../types";

interface PromptSensitivityPanelProps {
  sensitivity: PromptSensitivityPayload;
  selectedTemplate: string;
  onSelectTemplate: (tpl: string) => void;
}

export const PromptSensitivityPanel: React.FC<PromptSensitivityPanelProps> = ({
  sensitivity,
  selectedTemplate,
  onSelectTemplate,
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-amber-400 font-bold text-sm">🧪 Prompt Sensitivity Analysis</span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          {sensitivity.templates.length} Evaluated Templates
        </span>
      </div>

      {/* Templates Comparison Table */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Template Accuracy Comparison
        </span>
        <div className="flex flex-col gap-1.5">
          {sensitivity.templates.map((tpl) => {
            const res = sensitivity.results[tpl];
            const isSelected = tpl === selectedTemplate;
            const acc = res ? res.accuracy : 0.0;

            return (
              <div
                key={tpl}
                onClick={() => onSelectTemplate(tpl)}
                className={`flex items-center justify-between p-2.5 rounded-lg border cursor-pointer transition-all ${
                  isSelected
                    ? "bg-amber-950/40 border-amber-500/50"
                    : "bg-slate-950/40 border-slate-800/80 hover:bg-slate-800/40"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-200">
                    &quot;{tpl}&quot;
                  </span>
                  {isSelected && (
                    <span className="bg-amber-500/20 text-amber-300 text-[9px] font-bold px-1.5 py-0.2 rounded border border-amber-500/30">
                      ACTIVE
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-amber-400 h-full rounded-full"
                      style={{ width: `${Math.max(0, Math.min(100, acc * 100))}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs font-bold text-amber-300 w-12 text-right">
                    {(acc * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Pairwise Agreement Badges */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Cross-Template Prediction Agreement
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {Object.entries(sensitivity.pairwise_agreements).map(([pairKey, agree]) => (
            <div
              key={pairKey}
              className="bg-slate-950/40 p-2 rounded-lg border border-slate-800/80 flex items-center justify-between"
            >
              <span className="text-[10px] font-mono text-slate-400 truncate max-w-[180px]">
                {pairKey}
              </span>
              <span className="text-xs font-bold font-mono text-emerald-400">
                {(agree * 100).toFixed(0)}% Agree
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
