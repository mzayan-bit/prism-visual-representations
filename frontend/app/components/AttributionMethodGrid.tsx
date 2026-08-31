"use client";

import React from "react";
import { AttributionHeatmapCard } from "./AttributionHeatmapCard";
import { ExplainabilitySamplePayload } from "../types";

interface AttributionMethodGridProps {
  sample: ExplainabilitySamplePayload;
  selectedArch: string;
  selectedLayer: string;
}

export const AttributionMethodGrid: React.FC<AttributionMethodGridProps> = ({
  sample,
  selectedArch,
  selectedLayer,
}) => {
  const archAttributions = sample.attributions[selectedArch] || {};

  const igResult = archAttributions["input_gradient"];
  const gxiResult = archAttributions["gradient_x_input"];
  const occResult = archAttributions["occlusion_sensitivity"];
  const camResult = archAttributions["grad_cam"];
  const attnResult = archAttributions["vit_attention"];

  const isConvFamily = selectedArch === "cnn" || selectedArch === "resnet";
  const isTransformerFamily = selectedArch === "vit";

  return (
    <section className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-black text-white flex items-center gap-2 font-mono">
            <span>🔬</span> SPATIAL ATTRIBUTION COMPARISON GRID
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Side-by-side visualization of gradient, perturbation, and activation evidence on sample {sample.sample_id}.
          </p>
        </div>
        <div className="text-xs font-mono text-slate-500 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
          Target: <span className="text-cyan-400 font-bold">{sample.class_name}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. Input Gradient Saliency */}
        <AttributionHeatmapCard
          title="1. Input Gradient"
          subtitle="Vanilla saliency (dS_c / dx)"
          result={igResult}
          imageTensor={sample.image_tensor}
        />

        {/* 2. Gradient x Input */}
        <AttributionHeatmapCard
          title="2. Gradient × Input"
          subtitle="Elementwise sensitivity (dS_c / dx) * x"
          result={gxiResult}
          imageTensor={sample.image_tensor}
        />

        {/* 3. Occlusion Sensitivity */}
        <AttributionHeatmapCard
          title="3. Occlusion Sensitivity"
          subtitle="Perturbation drop (S_clean - S_occ)"
          result={occResult}
          imageTensor={sample.image_tensor}
        />

        {/* 4. Grad-CAM OR ViT Attention */}
        {isConvFamily ? (
          <AttributionHeatmapCard
            title={`4. Grad-CAM (${selectedLayer})`}
            subtitle="Activation weighted by pooled gradients"
            result={camResult}
            imageTensor={sample.image_tensor}
          />
        ) : isTransformerFamily ? (
          <AttributionHeatmapCard
            title="4. CLS-to-Patch Attention"
            subtitle="Multi-head query attention to patch tokens"
            result={attnResult}
            imageTensor={sample.image_tensor}
          />
        ) : (
          <AttributionHeatmapCard
            title="4. Architecture Specific"
            imageTensor={sample.image_tensor}
            isUnsupported={true}
            unsupportedReason="No architecture-specific method configured"
          />
        )}
      </div>
    </section>
  );
};
