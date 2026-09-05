"use client";

import React from "react";

interface ArchitectureSynthesisCardProps {
  synthesis: Record<string, Record<string, { mean?: number; std?: number | null }>>;
}

export const ArchitectureSynthesisCard: React.FC<ArchitectureSynthesisCardProps> = ({
  synthesis,
}) => {
  const architectures = Object.keys(synthesis || {});

  const metricsToCompare = [
    { key: "accuracy", label: "Clean Accuracy", unit: "%" },
    { key: "robustness_accuracy_drop", label: "Corruption Drop", unit: "% (lower better)" },
    { key: "detection_mean_iou", label: "Spatial Detection mIoU", unit: "IoU" },
    { key: "segmentation_miou", label: "Dense Segmentation", unit: "mIoU" },
    { key: "temporal_consistency", label: "Temporal Cosine Sim", unit: "cos" },
    { key: "ece", label: "Calibration (ECE)", unit: "ECE (lower better)" },
  ];

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div>
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>🏛️</span> Cross-Architecture Synthesis
        </h3>
        <p className="text-xs text-slate-400">
          Comparing inductive biases and transfer retention across CNN, ResNet, and ViT
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {architectures.map((arch) => {
          const archData = synthesis[arch] || {};

          return (
            <div
              key={arch}
              className="p-4 bg-slate-950 rounded-xl border border-slate-800 flex flex-col justify-between space-y-3"
            >
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-mono font-bold text-cyan-400 uppercase text-sm">
                  {arch}
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                  {arch === "vit" ? "Global Attention" : arch === "resnet" ? "Residual Bias" : "Local Conv"}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                {metricsToCompare.map((m) => {
                  const agg = archData[m.key];
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
                {arch === "vit"
                  ? "Superior in raw capacity, weaker local spatial inductive bias without pretraining."
                  : arch === "resnet"
                  ? "Balanced inductive bias, consistent robustness and strong representation geometry."
                  : "Tight local translation invariance, lower peak capacity on multimodal alignment."}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
