"use client";

import React, { useState } from "react";
import {
  getMultimodalDataset,
  getMultimodalSamples,
  getCrossModalRetrievalSummary,
  getZeroShotClassificationSummary,
  getPromptSensitivity,
  getSharedGeometry,
  getMultimodalObjectiveComparisons,
  getMultimodalRobustnessBenchmarks,
  getMultimodalCandidateFailures,
} from "../../../data/multimodalData";
import { MultimodalHeader } from "../../MultimodalHeader";
import { PairedSampleViewer } from "../../PairedSampleViewer";
import { TokenInspector } from "../../TokenInspector";
import { RetrievalExplorer } from "../../RetrievalExplorer";
import { SharedEmbeddingScatterPlot } from "../../SharedEmbeddingScatterPlot";
import { ZeroShotClassificationCard } from "../../ZeroShotClassificationCard";
import { PromptSensitivityPanel } from "../../PromptSensitivityPanel";
import { MultimodalObjectiveComparisonCard } from "../../MultimodalObjectiveComparisonCard";
import { MultimodalRobustnessCard } from "../../MultimodalRobustnessCard";
import { MultimodalFailureExplorer } from "../../MultimodalFailureExplorer";
import { Tabs } from "../../ui/Tabs";
import { Badge } from "../../ui/Badge";
import { StatStrip, StatItem } from "../../ui/StatStrip";

export const MultimodalWorkspace: React.FC = () => {
  const dataset = getMultimodalDataset();
  const samples = getMultimodalSamples();
  const retrievalSummary = getCrossModalRetrievalSummary();
  const zeroShotSummary = getZeroShotClassificationSummary();
  const promptSensitivity = getPromptSensitivity();
  const sharedGeometry = getSharedGeometry();
  const objectiveComparisons = getMultimodalObjectiveComparisons();
  const robustness = getMultimodalRobustnessBenchmarks();
  const failures = getMultimodalCandidateFailures();

  const [selectedArch, setSelectedArch] = useState<string>(
    dataset.metadata.architectures[0] || "resnet"
  );
  const [selectedSampleId, setSelectedSampleId] = useState<string>(
    samples[0]?.sample_id || ""
  );
  const [selectedTemplate, setSelectedTemplate] = useState<string>(
    dataset.metadata.prompt_templates[0] || "a {color} {shape} on the {position}"
  );
  const [retrievalDirection, setRetrievalDirection] = useState<
    "image_to_text" | "text_to_image"
  >("image_to_text");
  const [activeTab, setActiveTab] = useState<
    "alignment" | "retrieval" | "robustness" | "failures"
  >("alignment");

  const selectedSample =
    samples.find((s) => s.sample_id === selectedSampleId) || samples[0];

  if (!selectedSample) {
    return (
      <div className="p-8 text-center text-slate-400">
        No multimodal samples loaded.
      </div>
    );
  }

  const meanMatchedCosine = sharedGeometry.mean_paired_cosine || 0.84;
  const similarityGap = dataset.collapse_summary.similarity_gap || 0.65;

  const tabs = [
    { id: "alignment" as const, label: "Paired Alignment & Space" },
    { id: "retrieval" as const, label: "Cross-Modal Retrieval" },
    { id: "robustness" as const, label: "Prompt Sensitivity & Robustness" },
    { id: "failures" as const, label: "Failure Explorer" },
  ];

  const stats: StatItem[] = [
    {
      label: "Matched Cosine",
      value: meanMatchedCosine.toFixed(3),
      change: { value: "Paired Alignment", trend: "neutral" },
    },
    {
      label: "Similarity Gap",
      value: `+${similarityGap.toFixed(3)}`,
      change: { value: "Margin vs Unpaired", trend: "up" },
    },
    {
      label: "Image → Text R@1",
      value: `${(retrievalSummary.image_to_text_r1 * 100).toFixed(1)}%`,
      change: { value: "Top-1 Retrieval", trend: "up" },
    },
    {
      label: "Text → Image R@1",
      value: `${(retrievalSummary.text_to_image_r1 * 100).toFixed(1)}%`,
      change: { value: "Top-1 Retrieval", trend: "up" },
    },
    {
      label: "Zero-Shot Acc",
      value: `${(zeroShotSummary.accuracy * 100).toFixed(1)}%`,
      change: { value: "Prompt-based", trend: "neutral" },
    },
    {
      label: "Collapse Status",
      value: dataset.collapse_summary.is_collapsed ? "COLLAPSED" : "HEALTHY",
      change: {
        value: "Diversity",
        trend: dataset.collapse_summary.is_collapsed ? "down" : "up",
      },
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col gap-4 pb-2 border-b border-slate-800/80">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-100">Multimodal Alignment & Retrieval</h1>
              <Badge variant="accent" size="sm">
                Cross-Modal Space
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Analyze vision-language alignment, shared embedding geometry, bidirectional retrieval, and prompt sensitivity.
            </p>
          </div>
        </div>

        <MultimodalHeader
          architectures={dataset.metadata.architectures}
          selectedArch={selectedArch}
          onSelectArch={setSelectedArch}
          sampleIds={samples.map((s) => s.sample_id)}
          selectedSampleId={selectedSampleId}
          onSelectSampleId={setSelectedSampleId}
          promptTemplates={dataset.metadata.prompt_templates}
          selectedTemplate={selectedTemplate}
          onSelectTemplate={setSelectedTemplate}
          retrievalDirection={retrievalDirection}
          onSelectDirection={setRetrievalDirection}
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

      {/* Main Content Area */}
      <div className="min-h-[500px]">
        {activeTab === "alignment" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-7">
                <PairedSampleViewer sample={selectedSample} />
              </div>
              <div className="lg:col-span-5">
                <TokenInspector tokenized={selectedSample.tokenized} />
              </div>
            </div>
            <SharedEmbeddingScatterPlot
              samples={samples}
              selectedSampleId={selectedSampleId}
              onSelectSampleId={setSelectedSampleId}
              explainedVarianceRatio={sharedGeometry.explained_variance_ratio}
            />
          </div>
        )}

        {activeTab === "retrieval" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-6">
              <RetrievalExplorer
                direction={retrievalDirection}
                summary={retrievalSummary}
                selectedSample={selectedSample}
                allSamples={samples}
                onSelectSampleId={setSelectedSampleId}
              />
            </div>
            <div className="lg:col-span-6">
              <ZeroShotClassificationCard
                summary={zeroShotSummary}
                selectedSample={selectedSample}
              />
            </div>
          </div>
        )}

        {activeTab === "robustness" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-6">
                <PromptSensitivityPanel
                  sensitivity={promptSensitivity}
                  selectedTemplate={selectedTemplate}
                  onSelectTemplate={setSelectedTemplate}
                />
              </div>
              <div className="lg:col-span-6">
                <MultimodalRobustnessCard robustness={robustness} />
              </div>
            </div>
            <MultimodalObjectiveComparisonCard
              comparisons={objectiveComparisons}
            />
          </div>
        )}

        {activeTab === "failures" && (
          <MultimodalFailureExplorer
            failures={failures}
            onSelectSampleId={setSelectedSampleId}
          />
        )}
      </div>
    </div>
  );
};
