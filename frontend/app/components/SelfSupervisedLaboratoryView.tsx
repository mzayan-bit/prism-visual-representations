"use client";

import React, { useState } from "react";
import { SSLHeader } from "./SSLHeader";
import { AugmentationPairViewer } from "./AugmentationPairViewer";
import { TrainingDynamicsCard } from "./TrainingDynamicsCard";
import { CollapseDiagnosticsCard } from "./CollapseDiagnosticsCard";
import { SupervisedVsSSLComparisonCard } from "./SupervisedVsSSLComparisonCard";
import { SSLLabelEfficiencyChart } from "./SSLLabelEfficiencyChart";
import { SSLGeometryPanel } from "./SSLGeometryPanel";
import { SSLLayerProbePanel } from "./SSLLayerProbePanel";
import {
  sslDemoData,
  getSSLReport,
  getSSLComparison,
  getSSLLabelEfficiency,
  getSSLGeometryPoints,
  getSSLLayerProbes,
} from "../data/sslData";

export function SelfSupervisedLaboratoryView() {
  const [selectedArch, setSelectedArch] = useState<string>("resnet");
  const [selectedTemp, setSelectedTemp] = useState<number>(0.5);

  const report = getSSLReport(selectedArch);
  const comparison = getSSLComparison(selectedArch);
  const labelEfficiency = getSSLLabelEfficiency(selectedArch);
  const geometryPoints = getSSLGeometryPoints(selectedArch);
  const layerProbes = getSSLLayerProbes(selectedArch);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <SSLHeader
        metadata={sslDemoData.metadata}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        selectedTemp={selectedTemp}
        onSelectTemp={setSelectedTemp}
      />

      {/* Main Grid */}
      <div className="space-y-6">
        {/* Top: Augmentation Pair Viewer */}
        <AugmentationPairViewer />

        {/* Training Dynamics & Collapse Diagnostics */}
        {report && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TrainingDynamicsCard report={report} />
            <CollapseDiagnosticsCard collapse={report.collapse_summary} />
          </div>
        )}

        {/* Supervised vs SSL Comparison */}
        {comparison && (
          <SupervisedVsSSLComparisonCard comparison={comparison} />
        )}

        {/* Label Efficiency & Layer Probes */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SSLLabelEfficiencyChart points={labelEfficiency} />
          <SSLLayerProbePanel probes={layerProbes} />
        </div>

        {/* Post-Hoc Representation Geometry */}
        <SSLGeometryPanel points={geometryPoints} />
      </div>
    </div>
  );
}
