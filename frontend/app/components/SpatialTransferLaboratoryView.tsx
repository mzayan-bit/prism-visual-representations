"use client";

import React, { useMemo, useState } from "react";
import {
  getSpatialDataEfficiency,
  getSpatialDetectionSamples,
  getSpatialLayerTransferability,
  getSpatialObjectiveComparison,
  getSpatialSegmentationSamples,
} from "../data/spatialData";
import {
  PretrainingObjectiveType,
  SpatialTaskType,
  SpatialTransferStrategyType,
} from "../types";
import { DetectionVisualizer } from "./DetectionVisualizer";
import { SegmentationVisualizer } from "./SegmentationVisualizer";
import { SpatialDataEfficiencyCard } from "./SpatialDataEfficiencyCard";
import { SpatialLayerTransferabilityPanel } from "./SpatialLayerTransferabilityPanel";
import { SpatialObjectiveComparisonCard } from "./SpatialObjectiveComparisonCard";
import { SpatialTransferHeader } from "./SpatialTransferHeader";

const ARCH_AVAILABLE_LAYERS: Record<string, string[]> = {
  cnn: ["conv_0", "conv_1", "final_spatial"],
  resnet: ["stem", "stage_0", "stage_1", "final_spatial"],
  vit: ["patch_embeddings", "encoder_0", "encoder_1", "final_spatial"],
};

export const SpatialTransferLaboratoryView: React.FC = () => {
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

  // Retrieve dataset slices
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

  // Active selected report
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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* 1. Header with Controls */}
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

      {/* 2. Top Summary Metric Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-md">
          <span className="block text-[10px] font-mono text-slate-400">
            MEAN IOU
          </span>
          <span className="text-lg font-bold font-mono text-amber-400">
            {(meanIoU * 100).toFixed(1)}%
          </span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-md">
          <span className="block text-[10px] font-mono text-slate-400">
            {precisionOrAccLabel.toUpperCase()}
          </span>
          <span className="text-lg font-bold font-mono text-cyan-400">
            {precisionOrAcc}
          </span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-md">
          <span className="block text-[10px] font-mono text-slate-400">
            FEATURE RESOLUTION
          </span>
          <span className="text-lg font-bold font-mono text-slate-200">
            {activeReport?.feature_resolution ?? "16x16"}
          </span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-md">
          <span className="block text-[10px] font-mono text-slate-400">
            SPATIAL DRIFT (COS)
          </span>
          <span className="text-lg font-bold font-mono text-slate-200">
            {(activeReport?.spatial_representation_drift_cosine ?? 0).toFixed(3)}
          </span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-md">
          <span className="block text-[10px] font-mono text-slate-400">
            TRAINABLE FRACTION
          </span>
          <span className="text-lg font-bold font-mono text-emerald-400">
            {Math.round((activeReport?.trainable_fraction ?? 0) * 100)}%
          </span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-md">
          <span className="block text-[10px] font-mono text-slate-400">
            EPOCHS TRAINED
          </span>
          <span className="text-lg font-bold font-mono text-slate-200">
            {activeReport?.epochs_completed ?? 2}
          </span>
        </div>
      </div>

      {/* 3. Pretraining Objective Comparison Matrix */}
      <SpatialObjectiveComparisonCard
        comparison={comparison}
        taskType={selectedTask}
      />

      {/* 4. Interactive Downstream Task Visualizer */}
      {selectedTask === "object_detection" ? (
        <DetectionVisualizer samples={detectionSamples} />
      ) : (
        <SegmentationVisualizer samples={segmentationSamples} />
      )}

      {/* 5. Bottom Dual Panels: Layer Transferability & Data Efficiency */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SpatialLayerTransferabilityPanel
          layerRecords={layerRecords}
          activeTask={selectedTask}
        />
        <SpatialDataEfficiencyCard efficiencyRecords={efficiencyRecords} />
      </div>
    </div>
  );
};
