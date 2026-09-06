"use client";

import React from "react";
import { CorruptionType } from "../types";

interface RobustnessHeaderProps {
  architectures: string[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  corruptionTypes: string[];
  selectedCorruption: CorruptionType;
  onSelectCorruption: (c: CorruptionType) => void;
  selectedSeverity: number;
  onSelectSeverity: (sev: number) => void;
  activeTab: "overview" | "pca" | "severity_curves" | "failures" | "attention" | "cross_arch";
  onChangeTab: (tab: "overview" | "pca" | "severity_curves" | "failures" | "attention" | "cross_arch") => void;
  isViT: boolean;
}

const CORRUPTION_LABELS: Record<string, { label: string; icon: string; desc: string }> = {
  gaussian_noise: { label: "Gaussian Noise", icon: "⚡", desc: "Additive zero-mean sensor noise" },
  blur: { label: "Spatial Blur", icon: "💧", desc: "Gaussian optical defocus blur" },
  brightness: { label: "Brightness Shift", icon: "☀️", desc: "Global additive intensity shift" },
  contrast: { label: "Contrast Shift", icon: "🌓", desc: "Dynamic range scaling shift" },
  occlusion: { label: "Occlusion", icon: "⬛", desc: "Rectangular central masking" },
  resolution_degradation: { label: "Resolution Loss", icon: "🧱", desc: "Downsample & nearest upsample" },
};

export default function RobustnessHeader({
  architectures,
  selectedArch,
  onSelectArch,
  corruptionTypes,
  selectedCorruption,
  onSelectCorruption,
  selectedSeverity,
  onSelectSeverity,
  activeTab,
  onChangeTab,
  isViT,
}: RobustnessHeaderProps) {
  return (
    <header className="bg-slate-900/80 backdrop-blur-md border-b border-slate-800 sticky top-0 z-30 px-6 py-4">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        {/* Left: Title & Subtitle */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold text-lg">
            🛡️
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white">
                Robustness & Distribution Shift Laboratory
              </h1>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 uppercase">
                Robustness Analysis
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Controlled corruptions, manifold displacement, shared PCA basis & failure taxonomy
            </p>
          </div>
        </div>

        {/* Center: Architecture & Corruption Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Architecture Selector */}
          <div className="flex items-center bg-slate-950/60 p-1 rounded-lg border border-slate-800">
            {architectures.map((arch) => (
              <button
                key={arch}
                id={`arch-btn-${arch}`}
                onClick={() => onSelectArch(arch)}
                className={`px-3 py-1 text-xs font-semibold rounded-md transition-all uppercase ${
                  selectedArch === arch
                    ? "bg-cyan-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {arch}
              </button>
            ))}
          </div>

          {/* Corruption Selector */}
          <div className="flex items-center bg-slate-950/60 p-1 rounded-lg border border-slate-800">
            <select
              id="corruption-select"
              value={selectedCorruption}
              onChange={(e) => onSelectCorruption(e.target.value as CorruptionType)}
              className="bg-transparent text-xs font-medium text-slate-200 px-2 py-1 outline-none cursor-pointer"
            >
              {corruptionTypes.map((c) => {
                const info = CORRUPTION_LABELS[c] || { label: c, icon: "🔍" };
                return (
                  <option key={c} value={c} className="bg-slate-900 text-slate-200">
                    {info.icon} {info.label}
                  </option>
                );
              })}
            </select>
          </div>

          {/* Severity Slider */}
          <div className="flex items-center gap-2 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
            <span className="text-xs text-slate-400 font-medium">Severity:</span>
            <input
              id="severity-slider"
              type="range"
              min="1"
              max="5"
              step="1"
              value={selectedSeverity}
              onChange={(e) => onSelectSeverity(parseInt(e.target.value, 10))}
              className="w-24 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 min-w-[20px] text-center border border-cyan-500/30">
              {selectedSeverity}
            </span>
          </div>
        </div>
      </div>

      {/* Sub-Navigation Tabs */}
      <div className="flex items-center gap-2 mt-4 pt-3 border-t border-slate-800/80 overflow-x-auto text-xs">
        <button
          id="tab-overview"
          onClick={() => onChangeTab("overview")}
          className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "overview"
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <span>📊</span> Overview & Metrics
        </button>
        <button
          id="tab-pca"
          onClick={() => onChangeTab("pca")}
          className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "pca"
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <span>🧭</span> Shared PCA Drift Plot
        </button>
        <button
          id="tab-severity-curves"
          onClick={() => onChangeTab("severity_curves")}
          className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "severity_curves"
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <span>📉</span> Severity Curves & AUC
        </button>
        <button
          id="tab-failures"
          onClick={() => onChangeTab("failures")}
          className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "failures"
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <span>⚠️</span> Failure Taxonomy
        </button>
        {isViT && (
          <button
            id="tab-attention"
            onClick={() => onChangeTab("attention")}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
              activeTab === "attention"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <span>👁️</span> ViT Attention Drift
          </button>
        )}
        <button
          id="tab-cross-arch"
          onClick={() => onChangeTab("cross_arch")}
          className={`px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "cross_arch"
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <span>⚖️</span> Cross-Architecture Benchmark
        </button>
      </div>
    </header>
  );
}
