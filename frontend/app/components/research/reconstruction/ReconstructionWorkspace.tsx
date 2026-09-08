"use client";

import React, { useState } from "react";
import { ReconstructionHeader } from "../../ReconstructionHeader";
import { VisualTripletViewer } from "../../VisualTripletViewer";
import { ReconstructionDynamicsCard } from "../../ReconstructionDynamicsCard";
import { ObjectiveComparisonCard } from "../../ObjectiveComparisonCard";
import { MaskingRatioStudyCard } from "../../MaskingRatioStudyCard";
import { ReconstructionLayerProbePanel } from "../../ReconstructionLayerProbePanel";
import { ReconstructionFailureExplorer } from "../../ReconstructionFailureExplorer";
import {
  getReconstructionMetadata,
  getVisualTriplets,
  getReconstructionDynamics,
  getMaskingRatioStudy,
  getThreeWayComparison,
  getReconstructionLayerProbes,
  getReconstructionFailureCases,
} from "../../../data/reconstructionData";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";

export const ReconstructionWorkspace: React.FC = () => {
  const [selectedArch, setSelectedArch] = useState<string>("vit");
  const [selectedMethod, setSelectedMethod] = useState<string>(
    "masked_patch_reconstruction"
  );
  const [selectedMaskRatio, setSelectedMaskRatio] = useState<number>(0.5);
  const [selectedCorruption, setSelectedCorruption] =
    useState<string>("gaussian_noise");
  const [activeTab, setActiveTab] = useState<"triptych" | "dynamics" | "probes" | "failures">("triptych");

  const metadata = getReconstructionMetadata();
  const triplets = getVisualTriplets(selectedMethod);
  const dynamics = getReconstructionDynamics();
  const ratioStudy = getMaskingRatioStudy();
  const comparisons = getThreeWayComparison();
  const layerProbes = getReconstructionLayerProbes();
  const failureCases = getReconstructionFailureCases();

  const tabs = [
    { id: "triptych" as const, label: "Visual Triptych & Errors" },
    { id: "dynamics" as const, label: "Training Dynamics & 3-Way Comparison" },
    { id: "probes" as const, label: "Mask Ratio Study & Layer Probes" },
    { id: "failures" as const, label: "Forensic Failure Cases" },
  ];

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex flex-col gap-4 pb-2 border-b border-slate-800/80">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100">Reconstruction & Masked Autoencoding</h1>
              <Badge variant="warning" size="sm">
                Generative Pretraining
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Probe patch masking dynamics, denoising autoencoders, and spatial pixel-level reconstruction errors.
            </p>
          </div>
        </div>

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
        {activeTab === "triptych" && (
          <VisualTripletViewer
            triplets={triplets}
            selectedMethod={selectedMethod}
          />
        )}

        {activeTab === "dynamics" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ReconstructionDynamicsCard dynamics={dynamics} />
            <ObjectiveComparisonCard comparisons={comparisons} />
          </div>
        )}

        {activeTab === "probes" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <MaskingRatioStudyCard points={ratioStudy} />
            <ReconstructionLayerProbePanel probes={layerProbes} />
          </div>
        )}

        {activeTab === "failures" && (
          <ReconstructionFailureExplorer failureCases={failureCases} />
        )}
      </div>
    </div>
  );
};
