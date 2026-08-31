"use client";

import React from "react";
import { ExplainabilityExperimentMeta, ExplainabilitySamplePayload, TargetClassMode } from "../types";

interface ExplainabilityHeaderProps {
  meta: ExplainabilityExperimentMeta;
  samples: ExplainabilitySamplePayload[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  selectedSampleId: string;
  onSelectSampleId: (sampleId: string) => void;
  targetMode: TargetClassMode;
  onSelectTargetMode: (mode: TargetClassMode) => void;
  explicitTargetClass: number;
  onSelectExplicitTargetClass: (c: number) => void;
  selectedLayer: string;
  onSelectLayer: (layer: string) => void;
}

export const ExplainabilityHeader: React.FC<ExplainabilityHeaderProps> = ({
  meta,
  samples,
  selectedArch,
  onSelectArch,
  selectedSampleId,
  onSelectSampleId,
  targetMode,
  onSelectTargetMode,
  explicitTargetClass,
  onSelectExplicitTargetClass,
  selectedLayer,
  onSelectLayer,
}) => {
  const currentSample = samples.find((s) => s.sample_id === selectedSampleId) || samples[0];
  const currentPred = currentSample?.predictions[selectedArch];
  const availableLayers = meta.layers[selectedArch] || ["final_hidden"];

  return (
    <header className="bg-slate-900/90 backdrop-blur border-b border-slate-800 p-6 rounded-2xl shadow-xl mb-6">
      {/* Top Title & Scientific Guardrail Notice */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-cyan-950 border border-cyan-700/60 text-cyan-400">
              PHASE 16 // EXPLAINABILITY LAB
            </span>
            <span className="text-xs text-slate-500 font-mono">|</span>
            <span className="text-xs text-slate-400 font-mono">Spatial Evidence & Stability Analysis</span>
          </div>
          <h1 className="text-xl lg:text-2xl font-black tracking-tight text-white flex items-center gap-2">
            Visual Attribution & Model Interpretability Laboratory
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Probing spatial attribution signals across CNN, ResNet, and Vision Transformers.
            Attribution maps indicate model sensitivity and gradient concentration, not causal truth.
          </p>
        </div>

        {/* Prediction Status Pill */}
        {currentSample && currentPred && (
          <div className="flex items-center gap-3 bg-slate-950 p-3 rounded-xl border border-slate-800/80 shadow-inner">
            <div className="text-right">
              <div className="text-[10px] font-mono text-slate-500 uppercase">True Label</div>
              <div className="text-xs font-bold text-emerald-400 font-mono">
                {currentSample.class_name} ({currentSample.true_class})
              </div>
            </div>
            <span className="text-slate-700">/</span>
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase">Prediction</div>
              <div className="text-xs font-bold text-cyan-400 font-mono flex items-center gap-1.5">
                <span>{currentPred.predicted_name}</span>
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                  {Math.round(currentPred.confidence * 100)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Control Selector Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3.5 mt-5">
        {/* 1. Architecture Selector */}
        <div className="flex flex-col gap-1.5">
          <label className="text-[11px] font-mono text-slate-400 font-bold uppercase">Architecture</label>
          <select
            id="arch-selector"
            value={selectedArch}
            onChange={(e) => onSelectArch(e.target.value)}
            className="bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            {meta.architectures.map((arch) => (
              <option key={arch} value={arch}>
                {arch.toUpperCase()} ({arch === "vit" ? "Vision Transformer" : arch === "resnet" ? "Residual CNN" : "ConvNet"})
              </option>
            ))}
          </select>
        </div>

        {/* 2. Sample Selector */}
        <div className="flex flex-col gap-1.5">
          <label className="text-[11px] font-mono text-slate-400 font-bold uppercase">Sample Case</label>
          <select
            id="sample-selector"
            value={selectedSampleId}
            onChange={(e) => onSelectSampleId(e.target.value)}
            className="bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            {samples.map((s) => (
              <option key={s.sample_id} value={s.sample_id}>
                {s.sample_id.replace("sample_", "Sample #")} ({s.class_name})
              </option>
            ))}
          </select>
        </div>

        {/* 3. Target Class Mode */}
        <div className="flex flex-col gap-1.5">
          <label className="text-[11px] font-mono text-slate-400 font-bold uppercase">Target Mode</label>
          <select
            id="target-mode-selector"
            value={targetMode}
            onChange={(e) => onSelectTargetMode(e.target.value as TargetClassMode)}
            className="bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="predicted_class">Predicted Class</option>
            <option value="true_class">True Class</option>
            <option value="explicit_class">Explicit Target Class</option>
          </select>
        </div>

        {/* 4. Target Class Picker (conditional) or Layer Selection */}
        {targetMode === "explicit_class" ? (
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-slate-400 font-bold uppercase">Target Class</label>
            <select
              id="explicit-class-selector"
              value={explicitTargetClass}
              onChange={(e) => onSelectExplicitTargetClass(parseInt(e.target.value, 10))}
              className="bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              {meta.class_names.map((cName, idx) => (
                <option key={idx} value={idx}>
                  Class {idx}: {cName}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-slate-400 font-bold uppercase">Spatial Layer</label>
            <select
              id="layer-selector"
              value={selectedLayer}
              onChange={(e) => onSelectLayer(e.target.value)}
              className="bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              {availableLayers.map((layer) => (
                <option key={layer} value={layer}>
                  {layer}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* 5. Quick Sample Navigation Buttons */}
        <div className="flex flex-col gap-1.5 justify-end">
          <div className="flex items-center gap-1.5">
            <button
              id="btn-prev-sample"
              onClick={() => {
                const curIdx = samples.findIndex((s) => s.sample_id === selectedSampleId);
                if (curIdx > 0) onSelectSampleId(samples[curIdx - 1].sample_id);
              }}
              className="flex-1 py-2 px-3 bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg text-xs font-mono text-slate-300 font-bold hover:text-white"
            >
              ← Prev
            </button>
            <button
              id="btn-next-sample"
              onClick={() => {
                const curIdx = samples.findIndex((s) => s.sample_id === selectedSampleId);
                if (curIdx < samples.length - 1) onSelectSampleId(samples[curIdx + 1].sample_id);
              }}
              className="flex-1 py-2 px-3 bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg text-xs font-mono text-slate-300 font-bold hover:text-white"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
