"use client";

import React from "react";
import { AttributionHeatmapCard } from "./AttributionHeatmapCard";
import { ExplainabilitySamplePayload } from "../types";

interface CrossArchitectureAttributionPanelProps {
  sample: ExplainabilitySamplePayload;
}

export const CrossArchitectureAttributionPanel: React.FC<CrossArchitectureAttributionPanelProps> = ({
  sample,
}) => {
  const cnnCam = sample.attributions["cnn"]?.["grad_cam"];
  const resnetCam = sample.attributions["resnet"]?.["grad_cam"];
  const vitAttn = sample.attributions["vit"]?.["vit_attention"];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl mb-8">
      <div className="mb-6 pb-4 border-b border-slate-800/80">
        <h3 className="text-sm font-black text-white font-mono flex items-center gap-2">
          <span>🌐</span> CROSS-ARCHITECTURE ATTRIBUTION COMPARISON
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Observing how inductive biases shape spatial evidence: CNN Grad-CAM vs ResNet Grad-CAM vs ViT Patch Attention on matched input.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* CNN */}
        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800 font-mono text-xs">
            <span className="font-bold text-cyan-400">CONVNET (CNN)</span>
            <span className="text-slate-400">Grad-CAM</span>
          </div>
          <AttributionHeatmapCard
            title="CNN Grad-CAM"
            subtitle="Final Conv Layer Activation"
            result={cnnCam}
            imageTensor={sample.image_tensor}
          />
        </div>

        {/* ResNet */}
        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800 font-mono text-xs">
            <span className="font-bold text-emerald-400">RESIDUAL CNN</span>
            <span className="text-slate-400">Grad-CAM</span>
          </div>
          <AttributionHeatmapCard
            title="ResNet Grad-CAM"
            subtitle="Final Residual Stage Output"
            result={resnetCam}
            imageTensor={sample.image_tensor}
          />
        </div>

        {/* ViT */}
        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800 font-mono text-xs">
            <span className="font-bold text-purple-400">VISION TRANSFORMER</span>
            <span className="text-slate-400">CLS Attention</span>
          </div>
          <AttributionHeatmapCard
            title="ViT Patch Attention"
            subtitle="Encoder CLS Query Attention Map"
            result={vitAttn}
            imageTensor={sample.image_tensor}
          />
        </div>
      </div>

      <div className="mt-4 p-3 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-400 font-mono flex items-center gap-2">
        <span className="text-amber-400 font-bold">Scientific Note:</span>
        Convolutional Grad-CAM reflects gradient-weighted spatial activations, whereas ViT Attention reflects query-key routing weights.
        Direct visual differences should not be interpreted as absolute model superiority.
      </div>
    </div>
  );
};
