"use client";

import React, { useMemo, useState } from "react";
import { AppShell } from "./components/shell/AppShell";
import { OverviewHub } from "./components/research/overview/OverviewHub";
import { GeometryWorkspace } from "./components/research/geometry/GeometryWorkspace";
import { BenchmarkObservatoryView } from "./components/benchmark/BenchmarkObservatoryView";
import RobustnessLaboratoryView from "./components/RobustnessLaboratoryView";
import { ExplainabilityLaboratoryView } from "./components/ExplainabilityLaboratoryView";
import { TransferWorkspace } from "./components/research/transfer/TransferWorkspace";
import { SSLWorkspace } from "./components/research/ssl/SSLWorkspace";
import { ReconstructionWorkspace } from "./components/research/reconstruction/ReconstructionWorkspace";
import { SpatialTransferLaboratoryView } from "./components/SpatialTransferLaboratoryView";
import { TemporalLaboratoryView } from "./components/TemporalLaboratoryView";
import { MultimodalLaboratoryView } from "./components/MultimodalLaboratoryView";
import { UncertaintyLaboratoryView } from "./components/UncertaintyLaboratoryView";
import { AppMode } from "./components/ResearchPlatformNavigation";
import {
  getCrossArchitectureComparison,
  getLayerGeometryProfile,
  getObservatoryMetadata,
  getRepresentationGeometryReport,
} from "./observatoryData";
import { DistanceMetric } from "./types";

export default function PRISMDashboardPage() {
  const [appMode, setAppMode] = useState<AppMode | "overview">("overview");

  // Observatory state
  const metadata = useMemo(() => getObservatoryMetadata(), []);
  const comparison = useMemo(() => getCrossArchitectureComparison(), []);

  const [selectedArch, setSelectedArch] = useState<string>("resnet");
  const [selectedLayer, setSelectedLayer] = useState<string>("final_hidden");
  const [selectedBudget, setSelectedBudget] = useState<number>(1.0);
  const [distanceMetric, setDistanceMetric] =
    useState<DistanceMetric>("euclidean");
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  // Available layers for currently selected architecture
  const availableLayers = useMemo(() => {
    return metadata.layers[selectedArch] || ["final_hidden"];
  }, [metadata, selectedArch]);

  // Ensure selectedLayer is valid when architecture changes
  const handleSelectArch = (arch: string) => {
    setSelectedArch(arch);
    const layers = metadata.layers[arch] || ["final_hidden"];
    if (!layers.includes(selectedLayer)) {
      setSelectedLayer(layers[layers.length - 1]);
    }
  };

  // Get active report and profile
  const activeProfile = useMemo(() => {
    return getLayerGeometryProfile(selectedArch);
  }, [selectedArch]);

  const activeReport = useMemo(() => {
    return getRepresentationGeometryReport(selectedArch, selectedLayer);
  }, [selectedArch, selectedLayer]);

  // Dynamic breadcrumbs based on mode
  const breadcrumb = useMemo(() => {
    switch (appMode) {
      case "overview":
        return ["PRISM", "Overview"];
      case "benchmark":
        return ["PRISM", "Synthesis", "Benchmark Observatory"];
      case "observatory":
        return ["PRISM", "Representation", "Geometry Observatory"];
      case "robustness":
        return ["PRISM", "Representation", "Robustness & Invariance"];
      case "explainability":
        return ["PRISM", "Representation", "Layer Attribution"];
      case "transfer":
        return ["PRISM", "Learning", "Transfer Dynamics"];
      case "ssl":
        return ["PRISM", "Learning", "Self-Supervised (SimCLR)"];
      case "reconstruction":
        return ["PRISM", "Learning", "Reconstruction & Latents"];
      case "spatial":
        return ["PRISM", "Downstream", "Spatial Dense Transfer"];
      case "temporal":
        return ["PRISM", "Downstream", "Temporal Video Sequences"];
      case "multimodal":
        return ["PRISM", "Downstream", "Vision-Language Alignment"];
      case "uncertainty":
        return ["PRISM", "Reliability", "Uncertainty & Calibration"];
      default:
        return ["PRISM", "Research"];
    }
  }, [appMode]);

  return (
    <AppShell
      currentMode={appMode}
      onSelectMode={setAppMode}
      breadcrumb={breadcrumb}
      activeDataset="cifar10"
      activeSeed={42}
      isSynthetic={true}
      inspectorTitle={appMode === "observatory" ? "Geometry Inspector" : "Experiment Details"}
      inspectorSubtitle={`Architecture: ${selectedArch.toUpperCase()} | Layer: ${selectedLayer}`}
      inspectorMeta={[
        { label: "Architecture", value: selectedArch.toUpperCase() },
        { label: "Active Layer", value: selectedLayer.replace(/_/g, " ") },
        { label: "Data Budget", value: `${Math.round(selectedBudget * 100)}%` },
        { label: "Distance Metric", value: distanceMetric },
      ]}
      inspectorProvenance={{
        experimentId: `demo_exp_${selectedArch}_supervised_s42`,
        runId: `demo_run_${selectedArch}_supervised_42`,
        seed: 42,
        hardware: "Apple Silicon / CPU",
        fingerprint: `fp_${selectedArch}_42_canonical`,
      }}
    >
      {/* 0. Overview / Research Question Hub */}
      {appMode === "overview" && (
        <OverviewHub onNavigate={(mode) => setAppMode(mode)} />
      )}

      {/* 1. Synthesis: Cross-Paradigm Benchmark & Evidence Observatory */}
      {appMode === "benchmark" && <BenchmarkObservatoryView />}

      {/* 2. Representation: Geometry Observatory */}
      {appMode === "observatory" && (
        <GeometryWorkspace
          architectures={metadata.architectures}
          selectedArch={selectedArch}
          onSelectArch={handleSelectArch}
          availableLayers={availableLayers}
          selectedLayer={selectedLayer}
          onSelectLayer={setSelectedLayer}
          dataBudgets={metadata.data_budgets}
          selectedBudget={selectedBudget}
          onSelectBudget={setSelectedBudget}
          distanceMetric={distanceMetric}
          onSelectDistanceMetric={setDistanceMetric}
          activeReport={activeReport}
          activeProfile={activeProfile}
          comparison={comparison}
          selectedSampleId={selectedSampleId}
          onSelectSampleId={setSelectedSampleId}
        />
      )}

      {/* 3. Representation: Robustness & Distribution Shift Laboratory */}
      {appMode === "robustness" && <RobustnessLaboratoryView />}

      {/* 4. Representation: Explainability & Visual Attribution Laboratory */}
      {appMode === "explainability" && <ExplainabilityLaboratoryView />}

      {/* 5. Learning Paradigms: Transfer Dynamics */}
      {appMode === "transfer" && <TransferWorkspace />}

      {/* 6. Learning Paradigms: Self-Supervised Learning (SimCLR) */}
      {appMode === "ssl" && <SSLWorkspace />}

      {/* 7. Learning Paradigms: Reconstruction & Latents */}
      {appMode === "reconstruction" && <ReconstructionWorkspace />}

      {/* 8. Downstream: Spatial Dense Transfer */}
      {appMode === "spatial" && <SpatialTransferLaboratoryView />}

      {/* 9. Downstream: Video & Temporal Representation Learning */}
      {appMode === "temporal" && <TemporalLaboratoryView />}

      {/* 10. Downstream: Vision-Language Multimodal Alignment */}
      {appMode === "multimodal" && <MultimodalLaboratoryView />}

      {/* 11. Reliability: Uncertainty & Calibration Laboratory */}
      {appMode === "uncertainty" && <UncertaintyLaboratoryView />}
    </AppShell>
  );
}
