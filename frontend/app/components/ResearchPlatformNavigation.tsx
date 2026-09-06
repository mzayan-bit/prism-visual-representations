"use client";

import React, { useState } from "react";

export type AppMode =
  | "benchmark"
  | "observatory"
  | "robustness"
  | "explainability"
  | "transfer"
  | "ssl"
  | "reconstruction"
  | "spatial"
  | "temporal"
  | "multimodal"
  | "uncertainty";

export interface LabDefinition {
  id: AppMode;
  name: string;
  shortName: string;
  icon: string;
  description: string;
  themeColor: string;
}

export interface DomainCategory {
  id: string;
  title: string;
  icon: string;
  description: string;
  labs: LabDefinition[];
}

export const RESEARCH_DOMAINS: DomainCategory[] = [
  {
    id: "synthesis",
    title: "Synthesis & Benchmark",
    icon: "🏛️",
    description: "Cross-paradigm evidence synthesis, Pareto frontiers & findings",
    labs: [
      {
        id: "benchmark",
        name: "Benchmark Observatory",
        shortName: "Benchmark",
        icon: "🏛️",
        description: "Unified cross-paradigm evidence matrix, Pareto explorer & reports",
        themeColor: "cyan",
      },
    ],
  },
  {
    id: "representation",
    title: "Representation Geometry",
    icon: "🔬",
    description: "Manifold structure, corruption drift & attribution heatmaps",
    labs: [
      {
        id: "observatory",
        name: "Geometry Observatory",
        shortName: "Geometry",
        icon: "🔬",
        description: "PCA projections, intrinsic dimension, compactness & separation",
        themeColor: "cyan",
      },
      {
        id: "robustness",
        name: "Robustness Lab",
        shortName: "Robustness",
        icon: "🛡️",
        description: "Common corruptions, manifold displacement & shared PCA basis",
        themeColor: "cyan",
      },
      {
        id: "explainability",
        name: "Explainability Lab",
        shortName: "Attribution",
        icon: "🔍",
        description: "Integrated gradients, Grad-CAM & attribution agreement",
        themeColor: "cyan",
      },
    ],
  },
  {
    id: "learning",
    title: "Learning Paradigms",
    icon: "🔄",
    description: "Transfer probing, SimCLR contrastive & masked autoencoding",
    labs: [
      {
        id: "transfer",
        name: "Transfer Learning Lab",
        shortName: "Transfer",
        icon: "🔄",
        description: "Linear probe vs fine-tuning, layer transferability & data budgets",
        themeColor: "emerald",
      },
      {
        id: "ssl",
        name: "Self-Supervised Lab",
        shortName: "Contrastive SSL",
        icon: "🌌",
        description: "SimCLR augmentation invariance, collapse diagnostics & efficiency",
        themeColor: "indigo",
      },
      {
        id: "reconstruction",
        name: "Reconstruction Lab",
        shortName: "Reconstruction",
        icon: "🧩",
        description: "Masked autoencoding, patch dynamics & spatial feature retention",
        themeColor: "violet",
      },
    ],
  },
  {
    id: "downstream",
    title: "Downstream Probes",
    icon: "🎯",
    description: "Dense spatial transfer, video temporal sequence & vision-language",
    labs: [
      {
        id: "spatial",
        name: "Spatial Transfer Lab",
        shortName: "Spatial",
        icon: "🎯",
        description: "Dense detection mIoU, segmentation transfer & pooling policy",
        themeColor: "amber",
      },
      {
        id: "temporal",
        name: "Temporal Lab",
        shortName: "Temporal",
        icon: "🎬",
        description: "Frame representation dynamics & temporal consistency",
        themeColor: "amber",
      },
      {
        id: "multimodal",
        name: "Vision-Language Lab",
        shortName: "Multimodal",
        icon: "🌌",
        description: "Contrastive alignment, zero-shot classification & cross-retrieval",
        themeColor: "cyan",
      },
    ],
  },
  {
    id: "reliability",
    title: "Reliability & Calibration",
    icon: "🎲",
    description: "Uncertainty calibration, reliability diagrams & OOD detection",
    labs: [
      {
        id: "uncertainty",
        name: "Uncertainty & OOD Lab",
        shortName: "Uncertainty",
        icon: "🎲",
        description: "ECE, reliability diagrams, temperature scaling & OOD AUROC",
        themeColor: "amber",
      },
    ],
  },
];

interface NavigationProps {
  currentMode: AppMode;
  onSelectMode: (mode: AppMode) => void;
}

export const ResearchPlatformNavigation: React.FC<NavigationProps> = ({
  currentMode,
  onSelectMode,
}) => {
  // Find which domain contains the current mode
  const currentDomain =
    RESEARCH_DOMAINS.find((d) => d.labs.some((l) => l.id === currentMode)) ||
    RESEARCH_DOMAINS[0];

  const currentLab =
    currentDomain.labs.find((l) => l.id === currentMode) || currentDomain.labs[0];

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav className="bg-slate-900/95 backdrop-blur-md border-b border-slate-800 sticky top-0 z-40 shadow-2xl">
      {/* Top Header Strip */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 flex items-center justify-between gap-4 border-b border-slate-800/60">
        {/* Brand & Platform Identity */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 flex items-center justify-center font-mono font-black text-white text-sm shadow-md shadow-cyan-500/20">
            Ψ
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-black tracking-wider text-slate-100 font-mono">
                PRISM
              </span>
              <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-800/80">
                v1.0.0
              </span>
              <span className="hidden sm:inline-block px-1.5 py-0.2 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
                Controlled Synthetic Evidence
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block truncate max-w-md">
              Probing the Evolution of Visual Representations
            </p>
          </div>
        </div>

        {/* Right Info Badges & Mobile Toggle */}
        <div className="flex items-center gap-2.5">
          <div className="hidden lg:flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-950/80 px-3 py-1 rounded-full border border-slate-800">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Showcase: <strong className="text-slate-200">810 Verified Cells</strong></span>
            <span className="text-slate-600">•</span>
            <span>Architectures: <strong className="text-cyan-400">CNN, ResNet, ViT</strong></span>
          </div>

          {/* Mobile Domain Selector Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-200 border border-slate-700 flex items-center gap-1.5"
            aria-label="Toggle navigation menu"
          >
            <span>{currentLab.icon}</span>
            <span className="truncate max-w-[120px]">{currentLab.name}</span>
            <span className="text-slate-400">▼</span>
          </button>
        </div>
      </div>

      {/* Domain Navigation Tabs (Desktop) */}
      <div className="hidden lg:block bg-slate-950/70 px-4 sm:px-6 lg:px-8 border-b border-slate-800/40">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-1 overflow-x-auto py-1">
          <div className="flex items-center gap-1">
            {RESEARCH_DOMAINS.map((domain) => {
              const isDomainActive = domain.id === currentDomain.id;
              return (
                <div key={domain.id} className="relative group">
                  <button
                    onClick={() => {
                      if (!isDomainActive) {
                        onSelectMode(domain.labs[0].id);
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
                      isDomainActive
                        ? "bg-slate-800/90 text-cyan-300 font-bold border border-cyan-500/30 shadow-sm"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                    }`}
                  >
                    <span>{domain.icon}</span>
                    <span>{domain.title}</span>
                    {domain.labs.length > 1 && (
                      <span className="text-[10px] px-1 rounded bg-slate-800 text-slate-400 font-mono">
                        {domain.labs.length}
                      </span>
                    )}
                  </button>
                </div>
              );
            })}
          </div>

          <div className="text-[11px] font-mono text-slate-500">
            Current Workspace: <span className="text-cyan-400 font-bold">{currentLab.name}</span>
          </div>
        </div>
      </div>

      {/* Sub-Navigation Strip for Active Domain Labs */}
      <div className="bg-slate-900/80 px-4 sm:px-6 lg:px-8 py-1.5">
        <div className="max-w-7xl mx-auto flex items-center gap-2 overflow-x-auto">
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 font-semibold whitespace-nowrap mr-1 hidden sm:inline">
            {currentDomain.title} :
          </span>

          <div className="flex items-center gap-1.5 flex-nowrap">
            {currentDomain.labs.map((lab) => {
              const isActive = lab.id === currentMode;
              return (
                <button
                  key={lab.id}
                  id={`nav-mode-${lab.id}`}
                  onClick={() => onSelectMode(lab.id)}
                  title={lab.description}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
                    isActive
                      ? "bg-cyan-500 text-slate-950 font-black shadow-md shadow-cyan-500/20"
                      : "bg-slate-950/60 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800/80"
                  }`}
                >
                  <span>{lab.icon}</span>
                  <span>{lab.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Mobile Drawer Navigation */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-slate-950 border-b border-slate-800 p-4 space-y-4 shadow-2xl animate-in fade-in duration-150">
          <div className="text-xs font-mono text-slate-400 uppercase font-bold tracking-wider mb-2">
            PRISM Research Laboratories
          </div>
          {RESEARCH_DOMAINS.map((domain) => (
            <div key={domain.id} className="space-y-1.5">
              <div className="text-[11px] font-mono text-cyan-400 font-bold flex items-center gap-1.5">
                <span>{domain.icon}</span>
                <span>{domain.title}</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 pl-4">
                {domain.labs.map((lab) => (
                  <button
                    key={lab.id}
                    onClick={() => {
                      onSelectMode(lab.id);
                      setMobileMenuOpen(false);
                    }}
                    className={`px-3 py-2 rounded-lg text-xs font-medium text-left flex items-center justify-between gap-2 ${
                      lab.id === currentMode
                        ? "bg-cyan-500 text-slate-950 font-bold"
                        : "bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800"
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <span>{lab.icon}</span>
                      <span>{lab.name}</span>
                    </span>
                    {lab.id === currentMode && <span>✓</span>}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </nav>
  );
};
