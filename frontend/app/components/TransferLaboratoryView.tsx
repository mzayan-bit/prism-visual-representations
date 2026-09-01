"use client";

import React, { useState } from "react";
import {
  getTransferDataEfficiency,
  getTransferLayerProbes,
  getTransferMetadata,
  getTransferReport,
  getTransferSharedPCA,
} from "../transferData";
import { TransferStrategyType } from "../types";
import { DataEfficiencyChart } from "./DataEfficiencyChart";
import { FreezeMapCard } from "./FreezeMapCard";
import { LayerTransferabilityPanel } from "./LayerTransferabilityPanel";
import { RepresentationRetentionPanel } from "./RepresentationRetentionPanel";
import { TransferHeader } from "./TransferHeader";
import { TransferStrategyComparisonCard } from "./TransferStrategyComparisonCard";

export function TransferLaboratoryView() {
  const metadata = getTransferMetadata();

  const [selectedArch, setSelectedArch] = useState<string>("cnn");
  const [selectedStrategy, setSelectedStrategy] =
    useState<TransferStrategyType>("linear_probe");
  const [selectedBudget, setSelectedBudget] = useState<number>(1.0);

  const report = getTransferReport(selectedArch, selectedStrategy, selectedBudget);
  const layerProbes = getTransferLayerProbes(selectedArch);
  const dataEfficiency = getTransferDataEfficiency(selectedArch);
  const sharedPCA = getTransferSharedPCA(selectedArch);

  if (!report || !metadata) {
    return (
      <div className="p-8 text-center bg-slate-900/60 rounded-xl border border-slate-800">
        <p className="text-slate-400 font-mono text-sm">
          Loading transfer learning experimental dataset...
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 1. Header & Experiment Selector */}
      <TransferHeader
        metadata={metadata}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        selectedStrategy={selectedStrategy}
        onSelectStrategy={setSelectedStrategy}
        selectedBudget={selectedBudget}
        onSelectBudget={setSelectedBudget}
        trainableFraction={report.freeze_plan.trainable_fraction}
      />

      {/* 2. Side-by-Side Strategy Comparison Strip */}
      <TransferStrategyComparisonCard
        currentReport={report}
        comparison={report.scratch_comparison}
      />

      {/* 3. Main Analytical Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Parameter Freeze Architecture & Label Efficiency */}
        <div className="space-y-6">
          <FreezeMapCard
            freezePlan={report.freeze_plan}
            strategy={selectedStrategy}
            architecture={selectedArch}
          />

          <DataEfficiencyChart
            dataEfficiency={dataEfficiency}
            architecture={selectedArch}
          />
        </div>

        {/* Right Column: Layer Transfer Probes & Shared PCA Representation Drift */}
        <div className="space-y-6">
          <LayerTransferabilityPanel
            probes={layerProbes}
            architecture={selectedArch}
          />

          <RepresentationRetentionPanel
            driftSummary={report.representation_drift}
            sharedPCA={sharedPCA}
            architecture={selectedArch}
          />
        </div>
      </div>

      {/* 4. Scientific Disclaimers & Experimental Warnings */}
      {report.warnings && report.warnings.length > 0 && (
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-4">
          <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400 mb-2 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            Methodological Disclaimers & Transfer Constraints
          </div>
          <ul className="space-y-1 text-xs text-slate-400 list-disc list-inside font-mono">
            {report.warnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
