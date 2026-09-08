"use client";

import React, { useState } from "react";
import { SSLHeader } from "../../SSLHeader";
import { AugmentationPairViewer } from "../../AugmentationPairViewer";
import { TrainingDynamicsCard } from "../../TrainingDynamicsCard";
import { CollapseDiagnosticsCard } from "../../CollapseDiagnosticsCard";
import { SupervisedVsSSLComparisonCard } from "../../SupervisedVsSSLComparisonCard";
import { SSLLabelEfficiencyChart } from "../../SSLLabelEfficiencyChart";
import { SSLGeometryPanel } from "../../SSLGeometryPanel";
import { SSLLayerProbePanel } from "../../SSLLayerProbePanel";
import {
  sslDemoData,
  getSSLReport,
  getSSLComparison,
  getSSLLabelEfficiency,
  getSSLGeometryPoints,
  getSSLLayerProbes,
} from "../../../data/sslData";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";

export const SSLWorkspace: React.FC = () => {
  const [selectedArch, setSelectedArch] = useState<string>("resnet");
  const [selectedTemp, setSelectedTemp] = useState<number>(0.5);
  const [activeTab, setActiveTab] = useState<"pairs" | "training" | "probing">("pairs");

  const report = getSSLReport(selectedArch);
  const comparison = getSSLComparison(selectedArch);
  const labelEfficiency = getSSLLabelEfficiency(selectedArch);
  const geometryPoints = getSSLGeometryPoints(selectedArch);
  const layerProbes = getSSLLayerProbes(selectedArch);

  const tabs = [
    { id: "pairs" as const, label: "Augmentation & Representation" },
    { id: "training" as const, label: "Training & Collapse Diagnostics" },
    { id: "probing" as const, label: "Supervised Comparison & Probes" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 pb-2 border-b border-slate-800/80">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100">Self-Supervised Learning (SimCLR)</h1>
              <Badge variant="accent" size="sm">
                NT-Xent Contrastive
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Probe contrastive view invariance, projection head geometry, and dimensional collapse diagnostics.
            </p>
          </div>
        </div>

        <SSLHeader
          metadata={sslDemoData.metadata}
          selectedArch={selectedArch}
          onSelectArch={setSelectedArch}
          selectedTemp={selectedTemp}
          onSelectTemp={setSelectedTemp}
        />
      </div>

      {/* Progressive Disclosure Tabs */}
      <Tabs
        variant="segmented"
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Main Content Area */}
      <div className="min-h-[500px]">
        {activeTab === "pairs" && (
          <div className="space-y-6">
            <AugmentationPairViewer />
            <SSLGeometryPanel points={geometryPoints} />
          </div>
        )}

        {activeTab === "training" && report && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TrainingDynamicsCard report={report} />
            <CollapseDiagnosticsCard collapse={report.collapse_summary} />
          </div>
        )}

        {activeTab === "probing" && (
          <div className="space-y-6">
            {comparison && (
              <SupervisedVsSSLComparisonCard comparison={comparison} />
            )}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SSLLabelEfficiencyChart points={labelEfficiency} />
              <SSLLayerProbePanel probes={layerProbes} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
