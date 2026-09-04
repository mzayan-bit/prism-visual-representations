"use client";

import React, { useMemo, useState } from "react";
import {
  getTemporalCandidateFailures,
  getTemporalDataset,
  getTemporalMetadata,
  getTemporalObjectiveComparisons,
  getTemporalRobustnessBenchmarks,
  getTemporalSamples,
} from "../data/temporalData";
import {
  TemporalAggregationType,
  TemporalTransferStrategyType,
} from "../types";
import { TemporalAggregationCard } from "./TemporalAggregationCard";
import { TemporalFailureExplorer } from "./TemporalFailureExplorer";
import { TemporalHeader } from "./TemporalHeader";
import { TemporalLayerTransferabilityPanel } from "./TemporalLayerTransferabilityPanel";
import { TemporalObjectiveComparisonCard } from "./TemporalObjectiveComparisonCard";
import { TemporalPCATrajectoryPlot } from "./TemporalPCATrajectoryPlot";
import { TemporalRepresentationTimeline } from "./TemporalRepresentationTimeline";
import { TemporalRobustnessCard } from "./TemporalRobustnessCard";
import { VideoFrameStrip } from "./VideoFrameStrip";

export const TemporalLaboratoryView: React.FC = () => {
  const dataset = useMemo(() => getTemporalDataset(), []);
  const metadata = useMemo(() => getTemporalMetadata(), []);
  const samples = useMemo(() => getTemporalSamples(), []);
  const objectiveComparisons = useMemo(() => getTemporalObjectiveComparisons(), []);
  const robustnessBenchmarks = useMemo(() => getTemporalRobustnessBenchmarks(), []);
  const candidateFailures = useMemo(() => getTemporalCandidateFailures(), []);

  const [selectedArch, setSelectedArch] = useState<string>(
    metadata?.architectures?.[0] || "resnet"
  );
  const [selectedObjective, setSelectedObjective] = useState<string>(
    metadata?.pretraining_objectives?.[0] || "reconstruction"
  );
  const [selectedAggregator, setSelectedAggregator] = useState<TemporalAggregationType>(
    (metadata?.aggregators?.[0] as TemporalAggregationType) || "simple_rnn"
  );
  const [selectedStrategy, setSelectedStrategy] = useState<TemporalTransferStrategyType>(
    (metadata?.transfer_strategies?.[0] as TemporalTransferStrategyType) ||
      "frozen_frame_encoder"
  );
  const [selectedSampleId, setSelectedSampleId] = useState<string>(
    samples[0]?.video_id || "vid_val_0000"
  );
  const [activeFrameIndex, setActiveFrameIndex] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<
    "inspector" | "objectives_and_layers" | "robustness" | "failures"
  >("inspector");

  const activeSample = useMemo(() => {
    return samples.find((s) => s.video_id === selectedSampleId) || samples[0];
  }, [samples, selectedSampleId]);

  const layerProfiles = useMemo(() => {
    return dataset?.layer_profiles || {};
  }, [dataset]);

  return (
    <div>
      {/* Top Header Control Bar */}
      <TemporalHeader
        architectures={metadata?.architectures || ["resnet", "cnn", "vision_transformer"]}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        pretrainingObjectives={
          metadata?.pretraining_objectives || [
            "reconstruction",
            "supervised",
            "simclr",
            "scratch",
          ]
        }
        selectedObjective={selectedObjective}
        onSelectObjective={setSelectedObjective}
        aggregators={
          metadata?.aggregators || [
            "simple_rnn",
            "learned_temporal_pooling",
            "mean_pool",
            "max_pool",
            "last_frame",
          ]
        }
        selectedAggregator={selectedAggregator}
        onSelectAggregator={setSelectedAggregator}
        transferStrategies={
          metadata?.transfer_strategies || [
            "frozen_frame_encoder",
            "partial_fine_tune",
            "full_fine_tune",
            "frame_independent",
          ]
        }
        selectedStrategy={selectedStrategy}
        onSelectStrategy={setSelectedStrategy}
        sampleIds={samples.map((s) => s.video_id)}
        selectedSampleId={selectedSampleId}
        onSelectSampleId={setSelectedSampleId}
      />

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab("inspector")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeTab === "inspector"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🎞️</span> Sequence & Representation Inspector
          </button>
          <button
            onClick={() => setActiveTab("objectives_and_layers")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeTab === "objectives_and_layers"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>📊</span> Objectives & Layer Transferability
          </button>
          <button
            onClick={() => setActiveTab("robustness")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeTab === "robustness"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🛡️</span> Temporal Robustness
          </button>
          <button
            onClick={() => setActiveTab("failures")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeTab === "failures"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>🔍</span> Failure Explorer
          </button>
        </div>

        {/* Tab 1: Sequence & Representation Inspector */}
        {activeTab === "inspector" && (
          <div className="space-y-6">
            {/* Top Row: Video Frame Strip */}
            <VideoFrameStrip
              sample={activeSample}
              activeFrameIndex={activeFrameIndex}
              onSelectFrame={setActiveFrameIndex}
            />

            {/* Middle Row: Timeline Chart + Aggregation State Card */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-7">
                <TemporalRepresentationTimeline
                  metrics={activeSample?.timeline_metrics || []}
                  activeFrameIndex={activeFrameIndex}
                  onSelectFrame={setActiveFrameIndex}
                />
              </div>
              <div className="lg:col-span-5">
                <TemporalAggregationCard
                  aggregator={selectedAggregator}
                  sample={activeSample}
                  activeFrameIndex={activeFrameIndex}
                />
              </div>
            </div>

            {/* Bottom Row: 2D PCA Trajectory */}
            <TemporalPCATrajectoryPlot
              trajectory={activeSample?.pca_trajectory || []}
              activeFrameIndex={activeFrameIndex}
              onSelectFrame={setActiveFrameIndex}
            />
          </div>
        )}

        {/* Tab 2: Objectives & Layer Transferability */}
        {activeTab === "objectives_and_layers" && (
          <div className="space-y-6">
            <TemporalObjectiveComparisonCard comparisons={objectiveComparisons} />
            <TemporalLayerTransferabilityPanel layerProfiles={layerProfiles} />
          </div>
        )}

        {/* Tab 3: Temporal Robustness */}
        {activeTab === "robustness" && (
          <div className="space-y-6">
            <TemporalRobustnessCard benchmarks={robustnessBenchmarks} />
          </div>
        )}

        {/* Tab 4: Failure Explorer */}
        {activeTab === "failures" && (
          <div className="space-y-6">
            <TemporalFailureExplorer
              failures={candidateFailures}
              onSelectSample={(id) => {
                setSelectedSampleId(id);
                setActiveTab("inspector");
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};
