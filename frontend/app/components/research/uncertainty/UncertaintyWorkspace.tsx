"use client";

import React, { useState } from "react";
import {
  getCalibratedReport,
  getCalibrationReport,
  getCorruptionUncertaintyCurves,
  getOODBinaryEvaluations,
  getRepresentationRelationship,
  getTemperatureScalingResult,
  getUncertaintyArchitectureComparisons,
  getUncertaintyMetadata,
  getUncertaintyObjectiveComparisons,
  getUncertaintyReport,
  getUncertaintySamples,
} from "../../../uncertaintyData";
import { UncertaintySampleItemPayload } from "../../../types";
import { UncertaintyHeader } from "../../UncertaintyHeader";
import { ReliabilityDiagramCard } from "../../ReliabilityDiagramCard";
import { ConfidenceHistogramCard } from "../../ConfidenceHistogramCard";
import { TemperatureScalingCard } from "../../TemperatureScalingCard";
import { OODDistributionCard } from "../../OODDistributionCard";
import { OODROCCard } from "../../OODROCCard";
import { OODSampleExplorer } from "../../OODSampleExplorer";
import { RepresentationNoveltyScatter } from "../../RepresentationNoveltyScatter";
import { CorruptionUncertaintyCard } from "../../CorruptionUncertaintyCard";
import { UncertaintyObjectiveComparisonCard } from "../../UncertaintyObjectiveComparisonCard";
import { UncertaintyFailureExplorer } from "../../UncertaintyFailureExplorer";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";
import { StatStrip, StatItem } from "../../ui/StatStrip";

export const UncertaintyWorkspace: React.FC = () => {
  const meta = getUncertaintyMetadata();
  const rawReport = getUncertaintyReport();
  const uncalibratedReport = getCalibrationReport();
  const calibratedReport = getCalibratedReport();
  const tempScaling = getTemperatureScalingResult();
  const oodEvaluations = getOODBinaryEvaluations();
  const corruptionCurves = getCorruptionUncertaintyCurves();
  const representationRel = getRepresentationRelationship();
  const samples = getUncertaintySamples();
  const objectiveComparisons = getUncertaintyObjectiveComparisons();
  const architectureComparisons = getUncertaintyArchitectureComparisons();

  // State
  const [selectedArch, setSelectedArch] = useState<string>(meta.architecture || "ResNet-18");
  const [selectedObjective, setSelectedObjective] = useState<string>(
    meta.source_objective || "supervised"
  );
  const [selectedCalibrationMode, setSelectedCalibrationMode] = useState<string>(
    "uncalibrated"
  );
  const [selectedOODMethod, setSelectedOODMethod] = useState<string>("msp");
  const [selectedCorruption, setSelectedCorruption] = useState<string>(
    meta.corruptions?.[0] || "gaussian_noise"
  );
  const [selectedBinCount, setSelectedBinCount] = useState<number>(10);
  const [activeTab, setActiveTab] = useState<
    "calibration" | "ood" | "novelty" | "corruption" | "comparisons" | "failures"
  >("calibration");
  const [selectedSample, setSelectedSample] =
    useState<UncertaintySampleItemPayload | null>(samples[0] || null);

  const activeCalibrationReport =
    selectedCalibrationMode === "temperature_scaled" && calibratedReport
      ? calibratedReport
      : uncalibratedReport;

  const activeOODEval =
    oodEvaluations[selectedOODMethod] ||
    oodEvaluations["msp"] ||
    Object.values(oodEvaluations)[0];

  const ece = activeCalibrationReport?.ece ?? 0;
  const mce = activeCalibrationReport?.mce ?? 0;
  const brier = activeCalibrationReport?.brier_score ?? 0;
  const nll = activeCalibrationReport?.nll ?? 0;
  const auroc = activeOODEval?.auroc ?? 0;
  const fittedTemp = tempScaling?.fitted_temperature ?? 1.0;

  const tabs = [
    { id: "calibration" as const, label: "Calibration & Temperature Scaling" },
    { id: "ood" as const, label: "OOD Detection & ROC" },
    { id: "novelty" as const, label: "Feature Space Novelty" },
    { id: "corruption" as const, label: "Corruption Uncertainty" },
    { id: "comparisons" as const, label: "Cross-Model Comparisons" },
    { id: "failures" as const, label: "Failure Explorer" },
  ];

  const stats: StatItem[] = [
    {
      label: "Expected Calibration Error (ECE)",
      value: `${(ece * 100).toFixed(2)}%`,
      change: {
        value: selectedCalibrationMode === "temperature_scaled" ? "Scaled" : "Raw",
        trend: ece < 0.05 ? "up" : "down",
      },
    },
    {
      label: "Max Calibration Error (MCE)",
      value: `${(mce * 100).toFixed(2)}%`,
      change: { value: "Worst-case bin", trend: "neutral" },
    },
    {
      label: "Brier Score",
      value: brier.toFixed(4),
      change: { value: "Quadratic penalty", trend: "neutral" },
    },
    {
      label: "Negative Log Likelihood (NLL)",
      value: nll.toFixed(3),
      change: { value: "Proper score", trend: "neutral" },
    },
    {
      label: "OOD AUROC",
      value: `${(auroc * 100).toFixed(1)}%`,
      change: { value: selectedOODMethod.toUpperCase(), trend: auroc > 0.8 ? "up" : "neutral" },
    },
    {
      label: "Fitted Temperature (T*)",
      value: fittedTemp.toFixed(2),
      change: { value: "Post-hoc", trend: "neutral" },
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col gap-4 pb-2 border-b border-slate-800/80">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100">Uncertainty & OOD Diagnostics</h1>
              <Badge variant="accent" size="sm">
                Reliability & Calibration
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Probe predictive calibration, confidence over-confidence, out-of-distribution detection, and temperature scaling.
            </p>
          </div>
        </div>

        <UncertaintyHeader
          architectures={meta.architectures || ["ResNet-18", "Standard CNN", "ViT-Tiny"]}
          selectedArch={selectedArch}
          onSelectArch={setSelectedArch}
          objectives={meta.pretraining_objectives || ["supervised", "simclr", "reconstruction"]}
          selectedObjective={selectedObjective}
          onSelectObjective={setSelectedObjective}
          calibrationModes={meta.calibration_modes || ["uncalibrated", "temperature_scaled"]}
          selectedCalibrationMode={selectedCalibrationMode}
          onSelectCalibrationMode={setSelectedCalibrationMode}
          oodScoreMethods={meta.ood_score_methods || ["msp", "entropy", "class_centroid_distance", "knn_distance", "energy"]}
          selectedOODMethod={selectedOODMethod}
          onSelectOODMethod={setSelectedOODMethod}
          corruptions={meta.corruptions || ["gaussian_noise", "motion_blur"]}
          selectedCorruption={selectedCorruption}
          onSelectCorruption={setSelectedCorruption}
          binCounts={[5, 10, 15, 20]}
          selectedBinCount={selectedBinCount}
          onSelectBinCount={setSelectedBinCount}
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

      {/* Tab Panels */}
      <div className="min-h-[500px]">
        {activeTab === "calibration" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ReliabilityDiagramCard
                report={activeCalibrationReport}
                calibrationMode={selectedCalibrationMode}
              />
              <ConfidenceHistogramCard
                report={activeCalibrationReport}
                samples={samples}
              />
            </div>
            {tempScaling && (
              <TemperatureScalingCard
                temperatureScaling={tempScaling}
                uncalibratedReport={uncalibratedReport}
                calibratedReport={calibratedReport}
              />
            )}
          </div>
        )}

        {activeTab === "ood" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <OODDistributionCard
                evaluation={activeOODEval}
                selectedScoreMethod={selectedOODMethod}
                samples={samples}
              />
              <OODROCCard
                evaluation={activeOODEval}
                selectedScoreMethod={selectedOODMethod}
                samples={samples}
              />
            </div>
            <OODSampleExplorer
              samples={samples}
              selectedSampleId={selectedSample?.sample_id || null}
              onSelectSampleId={(id) => {
                const s = samples.find((item) => item.sample_id === id);
                if (s) setSelectedSample(s);
              }}
              selectedScoreMethod={selectedOODMethod}
              threshold={activeOODEval?.threshold ?? 0.5}
            />
          </div>
        )}

        {activeTab === "novelty" && (
          <RepresentationNoveltyScatter
            samples={samples}
            relationship={representationRel}
            onSelectSample={setSelectedSample}
          />
        )}

        {activeTab === "corruption" && (
          <CorruptionUncertaintyCard
            curves={corruptionCurves}
            predictionFlips={rawReport.prediction_flips || []}
            selectedCorruption={selectedCorruption}
            onSelectCorruption={setSelectedCorruption}
          />
        )}

        {activeTab === "comparisons" && (
          <UncertaintyObjectiveComparisonCard
            objectiveComparisons={objectiveComparisons}
            architectureComparisons={architectureComparisons}
            selectedObjective={selectedObjective}
            onSelectObjective={setSelectedObjective}
            selectedArch={selectedArch}
            onSelectArch={setSelectedArch}
          />
        )}

        {activeTab === "failures" && (
          <UncertaintyFailureExplorer
            samples={samples}
            failureCounts={rawReport.failure_counts || {}}
            onSelectSample={setSelectedSample}
          />
        )}
      </div>
    </div>
  );
};
