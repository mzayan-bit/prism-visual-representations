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
} from "../uncertaintyData";
import { UncertaintySampleItemPayload } from "../types";
import { UncertaintyHeader } from "./UncertaintyHeader";
import { UncertaintyOverviewStrip } from "./UncertaintyOverviewStrip";
import { ReliabilityDiagramCard } from "./ReliabilityDiagramCard";
import { ConfidenceHistogramCard } from "./ConfidenceHistogramCard";
import { TemperatureScalingCard } from "./TemperatureScalingCard";
import { OODDistributionCard } from "./OODDistributionCard";
import { OODROCCard } from "./OODROCCard";
import { OODSampleExplorer } from "./OODSampleExplorer";
import { RepresentationNoveltyScatter } from "./RepresentationNoveltyScatter";
import { CorruptionUncertaintyCard } from "./CorruptionUncertaintyCard";
import { UncertaintyObjectiveComparisonCard } from "./UncertaintyObjectiveComparisonCard";
import { UncertaintyFailureExplorer } from "./UncertaintyFailureExplorer";

export const UncertaintyLaboratoryView: React.FC = () => {
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

  // Global State
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
    "all" | "calibration" | "ood" | "novelty" | "corruption" | "comparisons" | "failures"
  >("all");
  const [selectedSample, setSelectedSample] =
    useState<UncertaintySampleItemPayload | null>(samples[0] || null);

  // Active calibration report based on mode
  const activeCalibrationReport =
    selectedCalibrationMode === "temperature_scaled" && calibratedReport
      ? calibratedReport
      : uncalibratedReport;

  const activeOODEval =
    oodEvaluations[selectedOODMethod] ||
    oodEvaluations["msp"] ||
    Object.values(oodEvaluations)[0];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Sticky Top Header */}
      <UncertaintyHeader
        architectures={meta.architectures || ["ResNet-18", "Standard CNN", "ViT-Tiny"]}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        objectives={
          meta.pretraining_objectives || [
            "supervised",
            "simclr",
            "reconstruction",
            "vision_language",
            "scratch",
          ]
        }
        selectedObjective={selectedObjective}
        onSelectObjective={setSelectedObjective}
        calibrationModes={meta.calibration_modes || ["uncalibrated", "temperature_scaled"]}
        selectedCalibrationMode={selectedCalibrationMode}
        onSelectCalibrationMode={setSelectedCalibrationMode}
        oodScoreMethods={
          meta.ood_score_methods || [
            "msp",
            "entropy",
            "class_centroid_distance",
            "knn_distance",
            "energy",
          ]
        }
        selectedOODMethod={selectedOODMethod}
        onSelectOODMethod={setSelectedOODMethod}
        corruptions={
          meta.corruptions || [
            "gaussian_noise",
            "motion_blur",
            "contrast_fade",
            "occlusion_patch",
            "pixel_shuffle",
          ]
        }
        selectedCorruption={selectedCorruption}
        onSelectCorruption={setSelectedCorruption}
        binCounts={[5, 10, 15, 20]}
        selectedBinCount={selectedBinCount}
        onSelectBinCount={setSelectedBinCount}
      />

      <main className="max-w-7xl w-full mx-auto p-4 sm:p-6 flex flex-col gap-6">
        {/* Metric Overview Strip */}
        <UncertaintyOverviewStrip
          calibrationReport={activeCalibrationReport}
          calibratedReport={calibratedReport}
          temperatureScaling={tempScaling}
          oodEvaluations={oodEvaluations}
          selectedOODMethod={selectedOODMethod}
          calibrationMode={selectedCalibrationMode}
        />

        {/* View Navigation Sub-tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 border-b border-slate-800 text-xs font-semibold">
          {[
            { id: "all", label: "📊 All Laboratory Views", icon: "🌐" },
            { id: "calibration", label: "🎯 Calibration & Scaling", icon: "🎯" },
            { id: "ood", label: "🛸 OOD Detection", icon: "🛸" },
            { id: "novelty", label: "🌌 Rep Novelty vs Confidence", icon: "🌌" },
            { id: "corruption", label: "🌪️ Corruption Curves", icon: "🌪️" },
            { id: "comparisons", label: "⚖️ Objective Benchmarks", icon: "⚖️" },
            { id: "failures", label: "🔍 Failure Auditing", icon: "🔍" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-slate-800 text-cyan-400 border border-slate-700 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
              }`}
            >
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* SECTION 1: Calibration & Temperature Scaling */}
        {(activeTab === "all" || activeTab === "calibration") && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <span>Section 1</span>
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
              <span className="text-slate-200">Probability Calibration & Reliability Diagnostics</span>
            </div>

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

            <TemperatureScalingCard
              temperatureScaling={tempScaling}
              uncalibratedReport={uncalibratedReport}
              calibratedReport={calibratedReport}
            />
          </section>
        )}

        {/* SECTION 2: Out-of-Distribution Representation Novelty */}
        {(activeTab === "all" || activeTab === "ood") && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <span>Section 2</span>
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
              <span className="text-slate-200">Out-of-Distribution Binary Separation & ROC Analysis</span>
            </div>

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
          </section>
        )}

        {/* SECTION 3: Representation Novelty Scatter */}
        {(activeTab === "all" || activeTab === "novelty") && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <span>Section 3</span>
              <span className="w-1.5 h-1.5 rounded-full bg-purple-500" />
              <span className="text-slate-200">Feature Manifold Geometry vs Predictive Confidence</span>
            </div>

            <RepresentationNoveltyScatter
              samples={samples}
              relationship={representationRel}
              onSelectSample={setSelectedSample}
            />
          </section>
        )}

        {/* SECTION 4: Corruption Uncertainty Dynamics */}
        {(activeTab === "all" || activeTab === "corruption") && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <span>Section 4</span>
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              <span className="text-slate-200">Perturbation Robustness & Prediction Flips</span>
            </div>

            <CorruptionUncertaintyCard
              curves={corruptionCurves}
              predictionFlips={rawReport.prediction_flips || []}
              selectedCorruption={selectedCorruption}
              onSelectCorruption={setSelectedCorruption}
            />
          </section>
        )}

        {/* SECTION 5: Objective & Architecture Comparative Benchmarks */}
        {(activeTab === "all" || activeTab === "comparisons") && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <span>Section 5</span>
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
              <span className="text-slate-200">Pretraining Objective & Architecture Benchmarks</span>
            </div>

            <UncertaintyObjectiveComparisonCard
              objectiveComparisons={objectiveComparisons}
              architectureComparisons={architectureComparisons}
              selectedObjective={selectedObjective}
              onSelectObjective={setSelectedObjective}
              selectedArch={selectedArch}
              onSelectArch={setSelectedArch}
            />
          </section>
        )}

        {/* SECTION 6: Failure Taxonomy Auditing */}
        {(activeTab === "all" || activeTab === "failures") && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <span>Section 6</span>
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span className="text-slate-200">Failure Mode Taxonomy & Edge-Case Explorer</span>
            </div>

            <UncertaintyFailureExplorer
              samples={samples}
              failureCounts={rawReport.failure_counts || {}}
              onSelectSample={setSelectedSample}
            />
          </section>
        )}
      </main>
    </div>
  );
};
