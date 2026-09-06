"use client";

import React from "react";
import {
  PretrainingObjectiveType,
  SpatialTaskType,
  SpatialTransferStrategyType,
} from "../types";

interface SpatialTransferHeaderProps {
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  selectedTask: SpatialTaskType;
  onSelectTask: (task: SpatialTaskType) => void;
  selectedObjective: PretrainingObjectiveType;
  onSelectObjective: (obj: PretrainingObjectiveType) => void;
  selectedStrategy: SpatialTransferStrategyType;
  onSelectStrategy: (strat: SpatialTransferStrategyType) => void;
  selectedLayer: string;
  onSelectLayer: (layer: string) => void;
  availableLayers: string[];
  selectedBudget: number;
  onSelectBudget: (budget: number) => void;
}

export const SpatialTransferHeader: React.FC<SpatialTransferHeaderProps> = ({
  selectedArch,
  onSelectArch,
  selectedTask,
  onSelectTask,
  selectedObjective,
  onSelectObjective,
  selectedStrategy,
  onSelectStrategy,
  selectedLayer,
  onSelectLayer,
  availableLayers,
  selectedBudget,
  onSelectBudget,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 mb-6 shadow-xl shadow-slate-950/40">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 mb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="text-xl">🎯</span>
            <h1 className="text-lg font-bold text-slate-100 tracking-tight">
              Spatial Transfer Laboratory
            </h1>
            <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[10px] font-mono font-bold px-2 py-0.5 rounded-full uppercase">
              Spatial Transfer
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Evaluate whether representations learned through supervised classification,
            SimCLR contrastive learning, and reconstruction transfer to spatial localization
            and dense pixel segmentation.
          </p>
        </div>

        {/* Target Spatial Task Switcher */}
        <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800 self-start lg:self-auto">
          <button
            id="task-btn-detection"
            onClick={() => onSelectTask("object_detection")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${
              selectedTask === "object_detection"
                ? "bg-amber-600 text-white shadow-lg shadow-amber-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>📦</span>
            <span>Object Detection</span>
          </button>
          <button
            id="task-btn-segmentation"
            onClick={() => onSelectTask("semantic_segmentation")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${
              selectedTask === "semantic_segmentation"
                ? "bg-teal-600 text-white shadow-lg shadow-teal-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🎨</span>
            <span>Semantic Segmentation</span>
          </button>
        </div>
      </div>

      {/* Primary Research Controls Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
        {/* Architecture */}
        <div>
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5">
            ENCODER ARCHITECTURE
          </label>
          <select
            id="spatial-arch-select"
            value={selectedArch}
            onChange={(e) => onSelectArch(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-semibold text-slate-200 focus:outline-none focus:border-amber-500 transition-colors"
          >
            <option value="cnn">Tiny CNN Baseline</option>
            <option value="resnet">Residual Network (ResNet)</option>
            <option value="vit">Vision Transformer (ViT)</option>
          </select>
        </div>

        {/* Source Objective */}
        <div>
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5">
            SOURCE PRETRAINING
          </label>
          <select
            id="spatial-obj-select"
            value={selectedObjective}
            onChange={(e) =>
              onSelectObjective(e.target.value as PretrainingObjectiveType)
            }
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-semibold text-slate-200 focus:outline-none focus:border-amber-500 transition-colors"
          >
            <option value="supervised">Supervised Classification</option>
            <option value="simclr">SimCLR Contrastive SSL</option>
            <option value="reconstruction">Reconstruction (MIM)</option>
            <option value="scratch">Random Scratch Baseline</option>
          </select>
        </div>

        {/* Transfer Strategy */}
        <div>
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5">
            TRANSFER STRATEGY
          </label>
          <select
            id="spatial-strat-select"
            value={selectedStrategy}
            onChange={(e) =>
              onSelectStrategy(e.target.value as SpatialTransferStrategyType)
            }
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-semibold text-slate-200 focus:outline-none focus:border-amber-500 transition-colors"
          >
            <option value="frozen_spatial_probe">Frozen Spatial Probe</option>
            <option value="partial_fine_tune">Partial Fine-Tune (Late)</option>
            <option value="full_fine_tune">Full Encoder Fine-Tune</option>
          </select>
        </div>

        {/* Spatial Layer */}
        <div>
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5">
            SPATIAL PROBE LAYER
          </label>
          <select
            id="spatial-layer-select"
            value={selectedLayer}
            onChange={(e) => onSelectLayer(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-semibold text-slate-200 focus:outline-none focus:border-amber-500 transition-colors"
          >
            {availableLayers.map((lay) => (
              <option key={lay} value={lay}>
                {lay}
              </option>
            ))}
          </select>
        </div>

        {/* Target Data Budget */}
        <div>
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5">
            SPATIAL DATA BUDGET
          </label>
          <select
            id="spatial-budget-select"
            value={selectedBudget}
            onChange={(e) => onSelectBudget(parseFloat(e.target.value))}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-semibold text-slate-200 focus:outline-none focus:border-amber-500 transition-colors"
          >
            <option value={1.0}>100% Annotations</option>
            <option value={0.5}>50% Annotations</option>
            <option value={0.25}>25% Annotations</option>
            <option value={0.1}>10% Annotations</option>
          </select>
        </div>
      </div>
    </div>
  );
};
