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
} from "../data/multimodalData";
import { MultimodalHeader } from "./MultimodalHeader";
import { PairedSampleViewer } from "./PairedSampleViewer";
import { TokenInspector } from "./TokenInspector";
import { RetrievalExplorer } from "./RetrievalExplorer";
import { SharedEmbeddingScatterPlot } from "./SharedEmbeddingScatterPlot";
import { ZeroShotClassificationCard } from "./ZeroShotClassificationCard";
import { PromptSensitivityPanel } from "./PromptSensitivityPanel";
import { MultimodalObjectiveComparisonCard } from "./MultimodalObjectiveComparisonCard";
import { MultimodalRobustnessCard } from "./MultimodalRobustnessCard";
import { MultimodalFailureExplorer } from "./MultimodalFailureExplorer";

export const MultimodalLaboratoryView: React.FC = () => {
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

  const selectedSample =
    samples.find((s) => s.sample_id === selectedSampleId) || samples[0];

  if (!selectedSample) {
    return (
      <div className="p-8 text-center text-slate-400">
        No multimodal samples loaded.
      </div>
    );
  }

  // Header quick overview stats
  const meanMatchedCosine = sharedGeometry.mean_paired_cosine || 0.84;
  const similarityGap = dataset.collapse_summary.similarity_gap || 0.65;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col pb-16">
      {/* Top Laboratory Control Bar */}
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

      <div className="max-w-7xl w-full mx-auto px-4 mt-6 flex flex-col gap-6">
        {/* Metric Overview Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl shadow">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Matched Cosine
            </span>
            <span className="text-lg font-bold font-mono text-emerald-400">
              {meanMatchedCosine.toFixed(3)}
            </span>
          </div>
          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl shadow">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Similarity Gap
            </span>
            <span className="text-lg font-bold font-mono text-cyan-400">
              +{similarityGap.toFixed(3)}
            </span>
          </div>
          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl shadow">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Image → Text R@1
            </span>
            <span className="text-lg font-bold font-mono text-emerald-400">
              {(retrievalSummary.image_to_text_r1 * 100).toFixed(1)}%
            </span>
          </div>
          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl shadow">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Text → Image R@1
            </span>
            <span className="text-lg font-bold font-mono text-indigo-400">
              {(retrievalSummary.text_to_image_r1 * 100).toFixed(1)}%
            </span>
          </div>
          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl shadow">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Zero-Shot Acc
            </span>
            <span className="text-lg font-bold font-mono text-amber-400">
              {(zeroShotSummary.accuracy * 100).toFixed(1)}%
            </span>
          </div>
          <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl shadow">
            <span className="text-[10px] font-mono text-slate-400 block uppercase">
              Collapse Status
            </span>
            <span className="text-sm font-bold font-mono text-emerald-400 mt-1 inline-block">
              {dataset.collapse_summary.is_collapsed ? "COLLAPSED" : "HEALTHY DIVERSITY"}
            </span>
          </div>
        </div>

        {/* Primary Row: Paired Sample & Token Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <PairedSampleViewer sample={selectedSample} />
          </div>
          <div className="lg:col-span-5">
            <TokenInspector tokenized={selectedSample.tokenized} />
          </div>
        </div>

        {/* Secondary Row: Cross-Modal Retrieval & Shared PCA Geometry */}
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
            <SharedEmbeddingScatterPlot
              samples={samples}
              selectedSampleId={selectedSampleId}
              onSelectSampleId={setSelectedSampleId}
              explainedVarianceRatio={sharedGeometry.explained_variance_ratio}
            />
          </div>
        </div>

        {/* Tertiary Row: Zero-Shot Classification & Prompt Sensitivity */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-6">
            <ZeroShotClassificationCard
              summary={zeroShotSummary}
              selectedSample={selectedSample}
            />
          </div>
          <div className="lg:col-span-6">
            <PromptSensitivityPanel
              sensitivity={promptSensitivity}
              selectedTemplate={selectedTemplate}
              onSelectTemplate={setSelectedTemplate}
            />
          </div>
        </div>

        {/* Quaternary Row: Objective Comparison & Robustness */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-6">
            <MultimodalObjectiveComparisonCard
              comparisons={objectiveComparisons}
            />
          </div>
          <div className="lg:col-span-6">
            <MultimodalRobustnessCard robustness={robustness} />
          </div>
        </div>

        {/* Final Row: Failure Explorer */}
        <MultimodalFailureExplorer
          failures={failures}
          onSelectSampleId={setSelectedSampleId}
        />
      </div>
    </div>
  );
};
