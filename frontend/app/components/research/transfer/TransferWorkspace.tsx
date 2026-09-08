"use client";

import React, { useState } from "react";
import {
  getTransferDataEfficiency,
  getTransferLayerProbes,
  getTransferMetadata,
  getTransferReport,
  getTransferSharedPCA,
} from "../../../transferData";
import { TransferStrategyType } from "../../../types";
import { DataEfficiencyChart } from "../../DataEfficiencyChart";
import { FreezeMapCard } from "../../FreezeMapCard";
import { LayerTransferabilityPanel } from "../../LayerTransferabilityPanel";
import { RepresentationRetentionPanel } from "../../RepresentationRetentionPanel";
import { TransferHeader } from "../../TransferHeader";
import { TransferStrategyComparisonCard } from "../../TransferStrategyComparisonCard";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";

export const TransferWorkspace: React.FC = () => {
  const metadata = getTransferMetadata();

  const [selectedArch, setSelectedArch] = useState<string>("cnn");
  const [selectedStrategy, setSelectedStrategy] =
    useState<TransferStrategyType>("linear_probe");
  const [selectedBudget, setSelectedBudget] = useState<number>(1.0);
  const [activeTab, setActiveTab] = useState<"strategy" | "layers" | "freeze">("strategy");

  const report = getTransferReport(selectedArch, selectedStrategy, selectedBudget);
  const layerProbes = getTransferLayerProbes(selectedArch);
  const dataEfficiency = getTransferDataEfficiency(selectedArch);
  const sharedPCA = getTransferSharedPCA(selectedArch);

  const tabs = [
    { id: "strategy" as const, label: "Strategy & Sample Efficiency" },
    { id: "layers" as const, label: "Layer Transferability & Drift" },
    { id: "freeze" as const, label: "Parameter Freeze Architecture" },
  ];

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
      <div className="flex flex-col gap-4 pb-2 border-b border-slate-800/80">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100">Transfer Learning & Representation Reuse</h1>
              <Badge variant="success" size="sm">
                Downstream Fine-Tuning
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Probe linear separability, parameter freeze policies, and sample efficiency curves.
            </p>
          </div>
        </div>

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
      </div>

      {/* Progressive Disclosure Tabs */}
      <Tabs
        variant="segmented"
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Tab Content */}
      <div className="min-h-[500px]">
        {activeTab === "strategy" && (
          <div className="space-y-6">
            <TransferStrategyComparisonCard
              currentReport={report}
              comparison={report.scratch_comparison}
            />
            <DataEfficiencyChart
              dataEfficiency={dataEfficiency}
              architecture={selectedArch}
            />
          </div>
        )}

        {activeTab === "layers" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
        )}

        {activeTab === "freeze" && (
          <div className="max-w-3xl">
            <FreezeMapCard
              freezePlan={report.freeze_plan}
              strategy={selectedStrategy}
              architecture={selectedArch}
            />
          </div>
        )}
      </div>

      {/* Scientific Warnings */}
      {report.warnings && report.warnings.length > 0 && (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-lg p-4">
          <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400 mb-1.5 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            Methodological Notes & Transfer Constraints
          </div>
          <ul className="space-y-1 text-xs text-slate-400 list-disc list-inside">
            {report.warnings.map((w: string, i: number) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
