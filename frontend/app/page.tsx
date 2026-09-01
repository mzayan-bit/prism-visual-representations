"use client";

import React, { useMemo, useState } from "react";
import { ArchitectureComparisonPanel } from "./components/ArchitectureComparisonPanel";
import { ClassCentroidsTable } from "./components/ClassCentroidsTable";
import { ExplainabilityLaboratoryView } from "./components/ExplainabilityLaboratoryView";
import { FailureExplorerPanel } from "./components/FailureExplorerPanel";
import { LayerEvolutionPanel } from "./components/LayerEvolutionPanel";
import { MetricOverviewStrip } from "./components/MetricOverviewStrip";
import { NeighborhoodPanel } from "./components/NeighborhoodPanel";
import { ObservatoryHeader } from "./components/ObservatoryHeader";
import { PCAScatterPlot } from "./components/PCAScatterPlot";
import RobustnessLaboratoryView from "./components/RobustnessLaboratoryView";
import { TransferLaboratoryView } from "./components/TransferLaboratoryView";
import {
  getCrossArchitectureComparison,
  getLayerGeometryProfile,
  getObservatoryMetadata,
  getRepresentationGeometryReport,
} from "./observatoryData";
import {
  DistanceMetric,
  NormalizationPolicy,
  SpatialTransformation,
} from "./types";

export default function PRISMDashboardPage() {
  const [appMode, setAppMode] = useState<
    "transfer" | "explainability" | "robustness" | "observatory"
  >("transfer");

  // Observatory state
  const metadata = useMemo(() => getObservatoryMetadata(), []);
  const comparison = useMemo(() => getCrossArchitectureComparison(), []);

  const [selectedArch, setSelectedArch] = useState<string>("resnet");
  const [selectedLayer, setSelectedLayer] = useState<string>("final_hidden");
  const [selectedBudget, setSelectedBudget] = useState<number>(1.0);
  const [spatialPolicy, setSpatialPolicy] = useState<SpatialTransformation>(
    "global_average_pool"
  );
  const [normPolicy, setNormPolicy] =
    useState<NormalizationPolicy>("none");
  const [distanceMetric, setDistanceMetric] =
    useState<DistanceMetric>("euclidean");
  const [activeTab, setActiveTab] = useState<
    "geometry" | "evolution" | "comparison"
  >("geometry");
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

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-900 selection:text-cyan-100">
      {/* Top Application Mode Navigation Switcher */}
      <nav className="bg-slate-900 border-b border-slate-800 px-6 py-2.5 flex flex-col md:flex-row md:items-center justify-between gap-3 z-40 sticky top-0">
        <div className="flex items-center gap-3">
          <div className="text-sm font-black tracking-wider text-cyan-400 font-mono">
            PRISM // RESEARCH SUITE
          </div>
          <span className="text-xs text-slate-500">|</span>
          <span className="text-xs text-slate-400">Visual Representation & Attribution Platform</span>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800 overflow-x-auto">
          <button
            id="nav-mode-transfer"
            onClick={() => setAppMode("transfer")}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
              appMode === "transfer"
                ? "bg-emerald-600 text-white shadow-md shadow-emerald-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🔄</span> Transfer Lab (Phase 17)
          </button>
          <button
            id="nav-mode-explainability"
            onClick={() => setAppMode("explainability")}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
              appMode === "explainability"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🔍</span> Explainability (Phase 16)
          </button>
          <button
            id="nav-mode-robustness"
            onClick={() => setAppMode("robustness")}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
              appMode === "robustness"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🛡️</span> Robustness (Phase 15)
          </button>
          <button
            id="nav-mode-observatory"
            onClick={() => setAppMode("observatory")}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
              appMode === "observatory"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🔬</span> Observatory (Phase 14)
          </button>
        </div>
      </nav>

      {/* Phase 17: Transfer Learning & Representation Reuse Laboratory */}
      {appMode === "transfer" && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <TransferLaboratoryView />
        </div>
      )}

      {/* Phase 16: Explainability & Visual Attribution Laboratory */}
      {appMode === "explainability" && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <ExplainabilityLaboratoryView />
        </div>
      )}

      {/* Phase 15: Robustness & Distribution Shift Laboratory */}
      {appMode === "robustness" && <RobustnessLaboratoryView />}

      {/* Phase 14: Representation Geometry Observatory */}
      {appMode === "observatory" && (
        <div>
          {/* Top Header Controls */}
          <ObservatoryHeader
            architectures={metadata.architectures}
            selectedArch={selectedArch}
            onSelectArch={handleSelectArch}
            availableLayers={availableLayers}
            selectedLayer={selectedLayer}
            onSelectLayer={setSelectedLayer}
            dataBudgets={metadata.data_budgets}
            selectedBudget={selectedBudget}
            onSelectBudget={setSelectedBudget}
            spatialPolicy={spatialPolicy}
            onSelectSpatialPolicy={setSpatialPolicy}
            normPolicy={normPolicy}
            onSelectNormPolicy={setNormPolicy}
            distanceMetric={distanceMetric}
            onSelectDistanceMetric={setDistanceMetric}
            activeTab={activeTab}
            onSelectTab={setActiveTab}
          />

          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
            {/* KPI Metric Strip */}
            <MetricOverviewStrip report={activeReport} />

            {/* Tab 1: Geometry & Neighborhood Inspector */}
            {activeTab === "geometry" && activeReport && (
              <div className="space-y-6">
                {/* Top Grid: Scatter Plot + Inspector Panels */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  {/* Central Left: Interactive 2D PCA Scatter Plot (7 cols) */}
                  <div className="lg:col-span-7">
                    <PCAScatterPlot
                      projection={activeReport.pca_projection}
                      centroidGeometry={activeReport.centroid_geometry}
                      sampleNeighborhoods={
                        activeReport.neighborhood_geometry.sample_neighborhoods
                      }
                      selectedSampleId={selectedSampleId}
                      onSelectSample={setSelectedSampleId}
                    />
                  </div>

                  {/* Central Right: Neighborhood & Failure Inspector (5 cols) */}
                  <div className="lg:col-span-5 space-y-4">
                    <NeighborhoodPanel
                      neighborhood={
                        selectedSampleId
                          ? activeReport.neighborhood_geometry.sample_neighborhoods[
                              selectedSampleId
                            ] || null
                          : null
                      }
                      selectedSampleId={selectedSampleId}
                      onSelectNeighbor={setSelectedSampleId}
                    />

                    <FailureExplorerPanel
                      failures={activeReport.candidate_failures}
                      selectedSampleId={selectedSampleId}
                      onSelectSample={setSelectedSampleId}
                    />
                  </div>
                </div>

                {/* Bottom: Class Centroids Table */}
                <ClassCentroidsTable
                  centroidGeometry={activeReport.centroid_geometry}
                />
              </div>
            )}

            {/* Tab 2: Layer-Wise Geometry Evolution */}
            {activeTab === "evolution" && (
              <LayerEvolutionPanel
                profile={activeProfile}
                onSelectLayer={(layer) => {
                  setSelectedLayer(layer);
                  setActiveTab("geometry");
                }}
              />
            )}

            {/* Tab 3: Cross-Architecture Benchmarks */}
            {activeTab === "comparison" && (
              <ArchitectureComparisonPanel comparison={comparison} />
            )}
          </div>
        </div>
      )}
    </main>
  );
}
