"use client";

import React, { useMemo, useState } from "react";
import {
  getCorruptionEvaluation,
  getCorruptionSeverityCurve,
  getCrossArchitectureRobustness,
  getRobustnessMetadata,
  getRobustnessReport,
  getSampleDrifts,
} from "../robustnessData";
import { CorruptionType } from "../types";
import ClassRobustnessPanel from "./ClassRobustnessPanel";
import CrossArchitectureRobustnessPanel from "./CrossArchitectureRobustnessPanel";
import PairedPCADriftPlot from "./PairedPCADriftPlot";
import RobustnessFailureExplorer from "./RobustnessFailureExplorer";
import RobustnessHeader from "./RobustnessHeader";
import RobustnessOverviewStrip from "./RobustnessOverviewStrip";
import SampleDriftInspector from "./SampleDriftInspector";
import SeverityCurvesPanel from "./SeverityCurvesPanel";
import ViTAttentionDriftPanel from "./ViTAttentionDriftPanel";

export default function RobustnessLaboratoryView() {
  const metadata = useMemo(() => getRobustnessMetadata(), []);
  const crossArchComparison = useMemo(
    () => getCrossArchitectureRobustness(),
    []
  );

  const [selectedArch, setSelectedArch] = useState<string>("resnet");
  const [selectedCorruption, setSelectedCorruption] =
    useState<CorruptionType>("gaussian_noise");
  const [selectedSeverity, setSelectedSeverity] = useState<number>(3);
  const [activeTab, setActiveTab] = useState<
    "overview" | "pca" | "severity_curves" | "failures" | "attention" | "cross_arch"
  >("overview");
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  const report = useMemo(() => {
    return getRobustnessReport(selectedArch);
  }, [selectedArch]);

  const evaluation = useMemo(() => {
    return getCorruptionEvaluation(
      selectedArch,
      selectedCorruption,
      selectedSeverity
    );
  }, [selectedArch, selectedCorruption, selectedSeverity]);

  const severityCurve = useMemo(() => {
    return getCorruptionSeverityCurve(selectedArch, selectedCorruption);
  }, [selectedArch, selectedCorruption]);

  const sampleDrifts = useMemo(() => {
    return getSampleDrifts(selectedArch, selectedCorruption, selectedSeverity);
  }, [selectedArch, selectedCorruption, selectedSeverity]);

  // Find active sample drift object
  const activeSampleDrift = useMemo(() => {
    if (!sampleDrifts.length) return null;
    if (selectedSampleId) {
      const found = sampleDrifts.find((s) => s.sample_id === selectedSampleId);
      if (found) return found;
    }
    return sampleDrifts[0];
  }, [sampleDrifts, selectedSampleId]);

  const isViT = selectedArch === "vit";

  return (
    <div className="space-y-6 pb-12">
      {/* Sticky Header with Arch, Corruption, Severity & Subtab Controls */}
      <RobustnessHeader
        architectures={metadata.architectures}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        corruptionTypes={metadata.corruption_types}
        selectedCorruption={selectedCorruption}
        onSelectCorruption={setSelectedCorruption}
        selectedSeverity={selectedSeverity}
        onSelectSeverity={setSelectedSeverity}
        activeTab={activeTab}
        onChangeTab={setActiveTab}
        isViT={isViT}
      />

      <div className="px-6 space-y-6">
        {/* KPI Metric Overview Strip */}
        <RobustnessOverviewStrip evaluation={evaluation} />

        {/* Tab 1: Overview & Metrics */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PairedPCADriftPlot
                pcaResult={evaluation?.geometry_drift.shared_pca || null}
                selectedSampleId={activeSampleDrift?.sample_id || null}
                onSelectSample={setSelectedSampleId}
                classNames={metadata.class_names}
              />
              <SeverityCurvesPanel
                curve={severityCurve}
                corruptionName={selectedCorruption}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SampleDriftInspector
                sampleDrift={activeSampleDrift}
                cleanNeighborhood={null}
                corruptedNeighborhood={null}
                classNames={metadata.class_names}
              />
              <ClassRobustnessPanel
                classDrifts={
                  evaluation?.geometry_drift.class_centroid_drifts || {}
                }
                classNames={metadata.class_names}
              />
            </div>
          </div>
        )}

        {/* Tab 2: Shared PCA Drift Plot */}
        {activeTab === "pca" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PairedPCADriftPlot
                pcaResult={evaluation?.geometry_drift.shared_pca || null}
                selectedSampleId={activeSampleDrift?.sample_id || null}
                onSelectSample={setSelectedSampleId}
                classNames={metadata.class_names}
              />
            </div>
            <div>
              <SampleDriftInspector
                sampleDrift={activeSampleDrift}
                cleanNeighborhood={null}
                corruptedNeighborhood={null}
                classNames={metadata.class_names}
              />
            </div>
          </div>
        )}

        {/* Tab 3: Severity Degradation Curves */}
        {activeTab === "severity_curves" && (
          <div className="space-y-6">
            <SeverityCurvesPanel
              curve={severityCurve}
              corruptionName={selectedCorruption}
            />
            <ClassRobustnessPanel
              classDrifts={
                evaluation?.geometry_drift.class_centroid_drifts || {}
              }
              classNames={metadata.class_names}
            />
          </div>
        )}

        {/* Tab 4: Failure Taxonomy Explorer */}
        {activeTab === "failures" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <RobustnessFailureExplorer
                failures={report?.flagged_failures || []}
                onSelectSample={setSelectedSampleId}
                selectedSampleId={activeSampleDrift?.sample_id || null}
              />
            </div>
            <div>
              <SampleDriftInspector
                sampleDrift={activeSampleDrift}
                cleanNeighborhood={null}
                corruptedNeighborhood={null}
                classNames={metadata.class_names}
              />
            </div>
          </div>
        )}

        {/* Tab 5: ViT Attention Drift */}
        {activeTab === "attention" && isViT && (
          <ViTAttentionDriftPanel
            attentionDrift={evaluation?.attention_drift || null}
            isViT={isViT}
          />
        )}

        {/* Tab 6: Cross-Architecture Benchmark */}
        {activeTab === "cross_arch" && (
          <CrossArchitectureRobustnessPanel comparison={crossArchComparison} />
        )}
      </div>
    </div>
  );
}
