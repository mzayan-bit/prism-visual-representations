"use client";

import React, { useState } from "react";
import { ReconstructionHeader } from "./ReconstructionHeader";
import { VisualTripletViewer } from "./VisualTripletViewer";
import { ReconstructionDynamicsCard } from "./ReconstructionDynamicsCard";
import { ObjectiveComparisonCard } from "./ObjectiveComparisonCard";
import { MaskingRatioStudyCard } from "./MaskingRatioStudyCard";
import { ReconstructionLayerProbePanel } from "./ReconstructionLayerProbePanel";
import { ReconstructionFailureExplorer } from "./ReconstructionFailureExplorer";
import {
  getReconstructionMetadata,
  getVisualTriplets,
  getReconstructionDynamics,
  getMaskingRatioStudy,
  getThreeWayComparison,
  getReconstructionLayerProbes,
  getReconstructionFailureCases,
} from "../data/reconstructionData";

export function ReconstructionLaboratoryView() {
  const [selectedArch, setSelectedArch] = useState<string>("vit");
  const [selectedMethod, setSelectedMethod] = useState<string>(
    "masked_patch_reconstruction"
  );
  const [selectedMaskRatio, setSelectedMaskRatio] = useState<number>(0.5);
  const [selectedCorruption, setSelectedCorruption] =
    useState<string>("gaussian_noise");

  const metadata = getReconstructionMetadata();
  const triplets = getVisualTriplets(selectedMethod);
  const dynamics = getReconstructionDynamics();
  const ratioStudy = getMaskingRatioStudy();
  const comparisons = getThreeWayComparison();
  const layerProbes = getReconstructionLayerProbes();
  const failureCases = getReconstructionFailureCases();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header Controls */}
      <ReconstructionHeader
        metadata={metadata}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        selectedMethod={selectedMethod}
        onSelectMethod={setSelectedMethod}
        selectedMaskRatio={selectedMaskRatio}
        onSelectMaskRatio={setSelectedMaskRatio}
        selectedCorruption={selectedCorruption}
        onSelectCorruption={setSelectedCorruption}
      />

      {/* Main Grid Workspace */}
      <div className="space-y-6">
        {/* Visual Triplet & Spatial Error Map */}
        <VisualTripletViewer
          triplets={triplets}
          selectedMethod={selectedMethod}
        />

        {/* Training Dynamics & 3-Way Paradigm Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ReconstructionDynamicsCard dynamics={dynamics} />
          <ObjectiveComparisonCard comparisons={comparisons} />
        </div>

        {/* Masking Ratio Study & Layer-Wise Representation Probes */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MaskingRatioStudyCard points={ratioStudy} />
          <ReconstructionLayerProbePanel probes={layerProbes} />
        </div>

        {/* Failure Case Diagnostics Explorer */}
        <ReconstructionFailureExplorer failureCases={failureCases} />
      </div>
    </div>
  );
}
