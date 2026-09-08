import React, { useState } from "react";
import { PCAScatterPlot } from "../../PCAScatterPlot";
import { MetricOverviewStrip } from "../../MetricOverviewStrip";
import { LayerEvolutionPanel } from "../../LayerEvolutionPanel";
import { ClassCentroidsTable } from "../../ClassCentroidsTable";
import { NeighborhoodPanel } from "../../NeighborhoodPanel";
import { ArchitectureComparisonPanel } from "../../ArchitectureComparisonPanel";
import { FailureExplorerPanel } from "../../FailureExplorerPanel";
import { Select } from "../../ui/Select";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";
import {
  CrossArchitectureGeometryReport,
  DistanceMetric,
  LayerGeometryProfile,
  RepresentationGeometryReport,
} from "../../../types";

export interface GeometryWorkspaceProps {
  architectures: string[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  availableLayers: string[];
  selectedLayer: string;
  onSelectLayer: (layer: string) => void;
  dataBudgets: number[];
  selectedBudget: number;
  onSelectBudget: (budget: number) => void;
  distanceMetric: DistanceMetric;
  onSelectDistanceMetric: (metric: DistanceMetric) => void;
  activeReport: RepresentationGeometryReport | null;
  activeProfile: LayerGeometryProfile | null;
  comparison: CrossArchitectureGeometryReport | null;
  selectedSampleId: string | null;
  onSelectSampleId: (id: string | null) => void;
}

export const GeometryWorkspace: React.FC<GeometryWorkspaceProps> = ({
  architectures,
  selectedArch,
  onSelectArch,
  availableLayers,
  selectedLayer,
  onSelectLayer,
  dataBudgets,
  selectedBudget,
  onSelectBudget,
  distanceMetric,
  onSelectDistanceMetric,
  activeReport,
  activeProfile,
  comparison,
  selectedSampleId,
  onSelectSampleId,
}) => {
  const [activeTab, setActiveTab] = useState<"geometry" | "evolution" | "comparison">("geometry");

  const tabs = [
    { id: "geometry" as const, label: "PCA Manifold & Neighborhoods" },
    { id: "evolution" as const, label: "Layer-Wise Evolution" },
    { id: "comparison" as const, label: "Cross-Architecture Benchmark" },
  ];

  if (!activeReport) {
    return (
      <div className="p-8 text-center text-slate-400">
        No geometry report available for the selected parameters.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Workspace Header & Primary Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-slate-100">Representation Geometry Observatory</h1>
            <Badge variant="accent" size="sm">
              Manifold Forensics
            </Badge>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Probe linear separability, compactness, intrinsic dimension, and layer-by-layer manifold progression.
          </p>
        </div>

        {/* Compact Selector Strip */}
        <div className="flex flex-wrap items-center gap-2.5">
          <Select
            label="Architecture"
            value={selectedArch}
            onChange={onSelectArch}
            options={architectures.map((a) => ({ value: a, label: a.toUpperCase() }))}
          />
          <Select
            label="Layer"
            value={selectedLayer}
            onChange={onSelectLayer}
            options={availableLayers.map((l) => ({ value: l, label: l.replace(/_/g, " ") }))}
          />
          <Select
            label="Data Budget"
            value={selectedBudget}
            onChange={(v) => onSelectBudget(parseFloat(v))}
            options={dataBudgets.map((b) => ({ value: b, label: `${Math.round(b * 100)}%` }))}
          />
          <Select
            label="Distance"
            value={distanceMetric}
            onChange={(v) => onSelectDistanceMetric(v as DistanceMetric)}
            options={[
              { value: "euclidean", label: "Euclidean" },
              { value: "cosine", label: "Cosine" },
            ]}
          />
        </div>
      </div>

      {/* Metrics Summary Strip */}
      <MetricOverviewStrip report={activeReport} />

      {/* Progressive Disclosure Tabs */}
      <Tabs
        variant="segmented"
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Tab Content */}
      <div className="min-h-[500px]">
        {activeTab === "geometry" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Large Interactive PCA Scatter Plot (7 cols) */}
              <div className="lg:col-span-7">
                <PCAScatterPlot
                  projection={activeReport.pca_projection}
                  centroidGeometry={activeReport.centroid_geometry}
                  sampleNeighborhoods={
                    activeReport.neighborhood_geometry.sample_neighborhoods
                  }
                  selectedSampleId={selectedSampleId}
                  onSelectSample={onSelectSampleId}
                />
              </div>

              {/* Neighborhood and Failure Inspector (5 cols) */}
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
                  onSelectNeighbor={onSelectSampleId}
                />

                <FailureExplorerPanel
                  failures={activeReport.candidate_failures}
                  selectedSampleId={selectedSampleId}
                  onSelectSample={onSelectSampleId}
                />
              </div>
            </div>

            {/* Bottom Centroids Table */}
            <ClassCentroidsTable
              centroidGeometry={activeReport.centroid_geometry}
            />
          </div>
        )}

        {activeTab === "evolution" && activeProfile && (
          <LayerEvolutionPanel
            profile={activeProfile}
            onSelectLayer={(layer) => {
              onSelectLayer(layer);
              setActiveTab("geometry");
            }}
          />
        )}

        {activeTab === "comparison" && comparison && (
          <ArchitectureComparisonPanel comparison={comparison} />
        )}
      </div>
    </div>
  );
};
