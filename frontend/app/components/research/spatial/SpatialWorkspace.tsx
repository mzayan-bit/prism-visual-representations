"use client";

import React, { useMemo, useState } from "react";
import {
  getSpatialDataEfficiency,
  getSpatialDetectionSamples,
  getSpatialLayerTransferability,
  getSpatialObjectiveComparison,
  getSpatialSegmentationSamples,
} from "../../../data/spatialData";
import {
  PretrainingObjectiveType,
  SpatialTaskType,
  SpatialTransferStrategyType,
} from "../../../types";
import { DetectionVisualizer } from "../../DetectionVisualizer";
import { SegmentationVisualizer } from "../../SegmentationVisualizer";
import { SpatialDataEfficiencyCard } from "../../SpatialDataEfficiencyCard";
import { SpatialLayerTransferabilityPanel } from "../../SpatialLayerTransferabilityPanel";
import { SpatialObjectiveComparisonCard } from "../../SpatialObjectiveComparisonCard";
import { SpatialTransferHeader } from "../../SpatialTransferHeader";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";
import { StatStrip, StatItem } from "../../ui/StatStrip";

const ARCH_AVAILABLE_LAYERS: Record<string, string[]> = {
  cnn: ["conv_0", "conv_1", "final_spatial"],
  resnet: ["stem", "stage_0", "stage_1", "final_spatial"],
  vit: ["patch_embeddings", "encoder_0", "encoder_1", "final_spatial"],
};

export const SpatialWorkspace: React.FC = () => {
  const [selectedArch, setSelectedArch] = useState<string>("cnn");
  const [selectedTask, setSelectedTask] = useState<SpatialTaskType>(
    "object_detection"
  );
  const [selectedObjective, setSelectedObjective] =
    useState<PretrainingObjectiveType>("supervised");
  const [selectedStrategy, setSelectedStrategy] =
    useState<SpatialTransferStrategyType>("frozen_spatial_probe");
  const [selectedLayer, setSelectedLayer] = useState<string>("final_spatial");
  const [selectedBudget, setSelectedBudget] = useState<number>(1.0);
  const [activeTab, setActiveTab] = useState<"visualizer" | "objectives" | "layers">("visualizer");

  const availableLayers = useMemo(() => {
    return ARCH_AVAILABLE_LAYERS[selectedArch.toLowerCase()] || ["final_spatial"];
  }, [selectedArch]);

  const handleSelectArch = (arch: string) => {
    setSelectedArch(arch);
    const layers = ARCH_AVAILABLE_LAYERS[arch.toLowerCase()] || ["final_spatial"];
    if (!layers.includes(selectedLayer)) {
      setSelectedLayer(layers[layers.length - 1]);
    }
  };

  const comparison = useMemo(() => {
    return getSpatialObjectiveComparison(selectedArch, selectedTask);
  }, [selectedArch, selectedTask]);

  const layerRecords = useMemo(() => {
    return getSpatialLayerTransferability(selectedArch);
  }, [selectedArch]);

  const efficiencyRecords = useMemo(() => {
    return getSpatialDataEfficiency(selectedArch);
  }, [selectedArch]);

  const detectionSamples = useMemo(() => {
    return getSpatialDetectionSamples();
  }, []);

  const segmentationSamples = useMemo(() => {
    return getSpatialSegmentationSamples();
  }, []);

  const activeReport = comparison[selectedObjective];

  const meanIoU =
    selectedTask === "object_detection"
      ? activeReport?.detection_metrics?.mean_iou ?? 0
      : activeReport?.segmentation_metrics?.mean_iou ?? 0;

  const precisionOrAcc =
    selectedTask === "object_detection"
      ? `${Math.round((activeReport?.detection_metrics?.precision ?? 0) * 100)}%`
      : `${Math.round(
          (activeReport?.segmentation_metrics?.pixel_accuracy ?? 0) * 100
        )}%`;

  const precisionOrAccLabel =
    selectedTask === "object_detection" ? "Precision @ 0.5" : "Pixel Accuracy";

  const tabs = [
    { id: "visualizer" as const, label: selectedTask === "object_detection" ? "Detection Visualizer" : "Segmentation Visualizer" },
    { id: "objectives" as const, label: "Objective Comparison" },
    { id: "layers" as const, label: "Layer Transferability & Efficiency" },
  ];

  const stats: StatItem[] = [
    {
      label: "Mean IoU",
      value: `${(meanIoU * 100).toFixed(1)}%`,
      change: { value: "Localization", trend: "up" },
    },
    {
      label: precisionOrAccLabel,
      value: precisionOrAcc,
      change: { value: "Target Task", trend: "neutral" },
    },
    {
      label: "Feature Resolution",
      value: activeReport?.feature_resolution ?? "16x16",
    },
    {
      label: "Spatial Drift (Cos)",
      value: (activeReport?.spatial_representation_drift_cosine ?? 0).toFixed(3),
    },
    {
      label: "Trainable Fraction",
      value: `${Math.round((activeReport?.trainable_fraction ?? 0) * 100)}%`,
      change: { value: "Frozen Backbone", trend: "neutral" },
    },
    {
      label: "Epochs Trained",
      value: activeReport?.epochs_completed ?? 2,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col gap-4 pb-2 border-b border-slate-800/80">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100">Spatial Dense Transfer</h1>
              <Badge variant="accent" size="sm">
                Localization & Segmentation
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Probe spatial coordinate preservation, bounding box localization, and dense pixel semantic segmentation.
            </p>
          </div>
        </div>

        <SpatialTransferHeader
          selectedArch={selectedArch}
          onSelectArch={handleSelectArch}
          selectedTask={selectedTask}
          onSelectTask={setSelectedTask}
          selectedObjective={selectedObjective}
          onSelectObjective={setSelectedObjective}
          selectedStrategy={selectedStrategy}
          onSelectStrategy={setSelectedStrategy}
          selectedLayer={selectedLayer}
          onSelectLayer={setSelectedLayer}
          availableLayers={availableLayers}
          selectedBudget={selectedBudget}
          onSelectBudget={setSelectedBudget}
        />
      </div>

      {/* Metric Strip */}
      <StatStrip items={stats} />

      {/* Progressive Disclosure Tabs */}
      <Tabs
        variant="segmented"
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Main Content */}
      <div className="min-h-[500px]">
        {activeTab === "visualizer" && (
          <div>
            {selectedTask === "object_detection" ? (
              <DetectionVisualizer samples={detectionSamples} />
            ) : (
              <SegmentationVisualizer samples={segmentationSamples} />
            )}
          </div>
        )}

        {activeTab === "objectives" && (
          <SpatialObjectiveComparisonCard
            comparison={comparison}
            taskType={selectedTask}
          />
        )}

        {activeTab === "layers" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SpatialLayerTransferabilityPanel
              layerRecords={layerRecords}
              activeTask={selectedTask}
            />
            <SpatialDataEfficiencyCard
              efficiencyRecords={efficiencyRecords}
            />
          </div>
        )}
      </div>
    </div>
  );
};
