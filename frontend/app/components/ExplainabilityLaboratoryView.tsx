"use client";

import React, { useState } from "react";
import { AttributionMethodGrid } from "./AttributionMethodGrid";
import { CorruptionAttributionPanel } from "./CorruptionAttributionPanel";
import { CrossArchitectureAttributionPanel } from "./CrossArchitectureAttributionPanel";
import { ExplainabilityFailureExplorer } from "./ExplainabilityFailureExplorer";
import { ExplainabilityHeader } from "./ExplainabilityHeader";
import { MethodAgreementMatrix } from "./MethodAgreementMatrix";
import {
  getAllExplainabilitySamples,
  getExplainabilityMetadata,
} from "../explainabilityData";
import { TargetClassMode } from "../types";

export const ExplainabilityLaboratoryView: React.FC = () => {
  const meta = getExplainabilityMetadata();
  const samples = getAllExplainabilitySamples();

  const [selectedArch, setSelectedArch] = useState<string>("resnet");
  const [selectedSampleId, setSelectedSampleId] = useState<string>(
    samples[0]?.sample_id || "sample_001_airplane"
  );
  const [targetMode, setTargetMode] =
    useState<TargetClassMode>("predicted_class");
  const [explicitTargetClass, setExplicitTargetClass] = useState<number>(0);
  const [selectedLayer, setSelectedLayer] = useState<string>("final_stage");
  const [activeTab, setActiveTab] = useState<
    "attribution_grid" | "corruption_drift" | "cross_architecture" | "failure_taxonomy"
  >("attribution_grid");

  const currentSample =
    samples.find((s) => s.sample_id === selectedSampleId) || samples[0];

  const handleSelectArch = (arch: string) => {
    setSelectedArch(arch);
    const layers = meta.layers[arch] || ["final_hidden"];
    setSelectedLayer(layers[0]);
  };

  if (!currentSample) {
    return (
      <div className="p-12 text-center text-slate-500 font-mono text-sm">
        Loading Explainability Laboratory data...
      </div>
    );
  }

  const comparisonReport = currentSample.comparison_reports[selectedArch];

  return (
    <div className="space-y-6">
      {/* 1. Header & Configuration Selectors */}
      <ExplainabilityHeader
        meta={meta}
        samples={samples}
        selectedArch={selectedArch}
        onSelectArch={handleSelectArch}
        selectedSampleId={selectedSampleId}
        onSelectSampleId={setSelectedSampleId}
        targetMode={targetMode}
        onSelectTargetMode={setTargetMode}
        explicitTargetClass={explicitTargetClass}
        onSelectExplicitTargetClass={setExplicitTargetClass}
        selectedLayer={selectedLayer}
        onSelectLayer={setSelectedLayer}
      />

      {/* 2. Research View Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
        <button
          id="tab-attribution-grid"
          onClick={() => setActiveTab("attribution_grid")}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center gap-2 ${
            activeTab === "attribution_grid"
              ? "bg-cyan-600 text-white shadow-lg shadow-cyan-600/20"
              : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
          }`}
        >
          <span>🔬</span> Attribution Methods & Agreement
        </button>

        <button
          id="tab-corruption-drift"
          onClick={() => setActiveTab("corruption_drift")}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center gap-2 ${
            activeTab === "corruption_drift"
              ? "bg-cyan-600 text-white shadow-lg shadow-cyan-600/20"
              : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
          }`}
        >
          <span>🌪️</span> Corruption Stability & Drift
        </button>

        <button
          id="tab-cross-arch"
          onClick={() => setActiveTab("cross_architecture")}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center gap-2 ${
            activeTab === "cross_architecture"
              ? "bg-cyan-600 text-white shadow-lg shadow-cyan-600/20"
              : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
          }`}
        >
          <span>🌐</span> Cross-Architecture Comparison
        </button>

        <button
          id="tab-failure-taxonomy"
          onClick={() => setActiveTab("failure_taxonomy")}
          className={`px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all flex items-center gap-2 ${
            activeTab === "failure_taxonomy"
              ? "bg-cyan-600 text-white shadow-lg shadow-cyan-600/20"
              : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
          }`}
        >
          <span>⚠️</span> Failure Diagnostics & Explorer
        </button>
      </div>

      {/* 3. Tab Contents */}
      {activeTab === "attribution_grid" && (
        <div className="space-y-6 animate-fadeIn">
          <AttributionMethodGrid
            sample={currentSample}
            selectedArch={selectedArch}
            selectedLayer={selectedLayer}
          />
          <MethodAgreementMatrix
            report={comparisonReport}
            selectedArch={selectedArch}
          />
        </div>
      )}

      {activeTab === "corruption_drift" && (
        <div className="animate-fadeIn">
          <CorruptionAttributionPanel
            sample={currentSample}
            selectedArch={selectedArch}
          />
        </div>
      )}

      {activeTab === "cross_architecture" && (
        <div className="animate-fadeIn">
          <CrossArchitectureAttributionPanel sample={currentSample} />
        </div>
      )}

      {activeTab === "failure_taxonomy" && (
        <div className="animate-fadeIn">
          <ExplainabilityFailureExplorer
            samples={samples}
            selectedArch={selectedArch}
            onSelectSample={setSelectedSampleId}
          />
        </div>
      )}
    </div>
  );
};
