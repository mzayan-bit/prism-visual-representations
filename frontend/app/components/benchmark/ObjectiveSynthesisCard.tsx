"use client";

import React from "react";

interface ObjectiveSynthesisCardProps {
  synthesis: Record<string, Record<string, { mean?: number; std?: number | null }>>;
}

export const ObjectiveSynthesisCard: React.FC<ObjectiveSynthesisCardProps> = ({
  synthesis,
}) => {
  const objectives = Object.keys(synthesis || {});

  const metricsToCompare = [
    { key: "linear_probe_accuracy", label: "Transfer Probe Acc" },
    { key: "transfer_gain", label: "Label-Efficiency Gain" },
    { key: "robustness_accuracy_drop", label: "Robustness Drop" },
    { key: "ood_auroc", label: "OOD AUROC" },
    { key: "retrieval_r1", label: "Multimodal R@1" },
  ];

  const objDescriptions: Record<string, string> = {
    supervised: "Optimizes category separability; high in-domain accuracy but vulnerable to distribution shift.",
    simclr: "Maximizes instance invariance; superior label-efficiency and robust manifold clustering.",
    reconstruction: "Preserves pixel-level and spatial details; highest spatial detection & segmentation probe transfer.",
    vision_language: "Aligns cross-modal embedding spaces; achieves zero-shot generalization and retrieval.",
    scratch: "Baseline trained without pretraining; exhibits steep data hungry decay curves.",
  };

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div>
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>🎯</span> Cross-Objective Pretraining Synthesis
        </h3>
        <p className="text-xs text-slate-400">
          Comparing pretraining representations across supervised, contrastive, reconstruction, and multimodal objectives
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {objectives.map((obj) => {
          const objData = synthesis[obj] || {};

          return (
            <div
              key={obj}
              className="p-4 bg-slate-950 rounded-xl border border-slate-800 flex flex-col justify-between space-y-3"
            >
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-mono font-bold text-cyan-400 capitalize text-sm">
                  {obj}
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 uppercase">
                  {obj === "supervised" ? "Supervised" : "Self-Supervised"}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                {metricsToCompare.map((m) => {
                  const agg = objData[m.key];
                  const val = agg?.mean;
                  const std = agg?.std;

                  return (
                    <div key={m.key} className="flex items-center justify-between font-mono">
                      <span className="text-slate-400 text-[11px]">{m.label}</span>
                      <span className="text-slate-200 font-bold">
                        {val !== undefined && val !== null ? (
                          <>
                            {val.toFixed(3)}
                            {std !== null && std !== undefined && (
                              <span className="text-[9px] text-slate-500 font-normal">
                                {" "}±{std.toFixed(3)}
                              </span>
                            )}
                          </>
                        ) : (
                          "—"
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="text-[10px] text-slate-500 pt-2 border-t border-slate-900 font-mono">
                {objDescriptions[obj] || "Controlled evaluation in PRISM benchmark framework."}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
