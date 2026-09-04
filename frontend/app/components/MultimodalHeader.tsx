"use client";

import React from "react";

interface MultimodalHeaderProps {
  architectures: string[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  sampleIds: string[];
  selectedSampleId: string;
  onSelectSampleId: (id: string) => void;
  promptTemplates: string[];
  selectedTemplate: string;
  onSelectTemplate: (tpl: string) => void;
  retrievalDirection: "image_to_text" | "text_to_image";
  onSelectDirection: (dir: "image_to_text" | "text_to_image") => void;
}

export const MultimodalHeader: React.FC<MultimodalHeaderProps> = ({
  architectures,
  selectedArch,
  onSelectArch,
  sampleIds,
  selectedSampleId,
  onSelectSampleId,
  promptTemplates,
  selectedTemplate,
  onSelectTemplate,
  retrievalDirection,
  onSelectDirection,
}) => {
  return (
    <div className="bg-slate-900 border-b border-slate-800 p-4 sticky top-12 z-30 shadow-md">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Title & Badge */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-bold text-lg shadow-inner">
            🌌
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-100">
                Vision-Language Representation Alignment Laboratory
              </h1>
              <span className="bg-cyan-950/80 text-cyan-400 text-[10px] font-mono font-semibold px-2 py-0.5 rounded border border-cyan-500/30">
                PHASE 22
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Dual-Encoder Contrastive Space • Cross-Modal Retrieval • Zero-Shot Classification
            </p>
          </div>
        </div>

        {/* Global Selectors */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Architecture Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Vision Backbone:</span>
            <select
              value={selectedArch}
              onChange={(e) => onSelectArch(e.target.value)}
              className="bg-slate-900 text-cyan-400 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              {architectures.map((arch) => (
                <option key={arch} value={arch}>
                  {arch.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Sample Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Paired Sample:</span>
            <select
              value={selectedSampleId}
              onChange={(e) => onSelectSampleId(e.target.value)}
              className="bg-slate-900 text-indigo-400 font-mono font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              {sampleIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>

          {/* Prompt Template */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Prompt Template:</span>
            <select
              value={selectedTemplate}
              onChange={(e) => onSelectTemplate(e.target.value)}
              className="bg-slate-900 text-emerald-400 font-mono text-[11px] rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-emerald-500 cursor-pointer"
            >
              {promptTemplates.map((tpl) => (
                <option key={tpl} value={tpl}>
                  &quot;{tpl}&quot;
                </option>
              ))}
            </select>
          </div>

          {/* Retrieval Direction Toggle */}
          <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700/80">
            <button
              onClick={() => onSelectDirection("image_to_text")}
              className={`px-2 py-1 rounded text-xs font-semibold transition-colors ${
                retrievalDirection === "image_to_text"
                  ? "bg-cyan-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Image → Text
            </button>
            <button
              onClick={() => onSelectDirection("text_to_image")}
              className={`px-2 py-1 rounded text-xs font-semibold transition-colors ${
                retrievalDirection === "text_to_image"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Text → Image
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
