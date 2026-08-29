"use client";

import React from "react";
import { DistanceMetric, NormalizationPolicy, SpatialTransformation } from "../types";

interface ObservatoryHeaderProps {
  architectures: string[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  availableLayers: string[];
  selectedLayer: string;
  onSelectLayer: (layer: string) => void;
  dataBudgets: number[];
  selectedBudget: number;
  onSelectBudget: (budget: number) => void;
  spatialPolicy: SpatialTransformation;
  onSelectSpatialPolicy: (policy: SpatialTransformation) => void;
  normPolicy: NormalizationPolicy;
  onSelectNormPolicy: (policy: NormalizationPolicy) => void;
  distanceMetric: DistanceMetric;
  onSelectDistanceMetric: (metric: DistanceMetric) => void;
  activeTab: "geometry" | "evolution" | "comparison";
  onSelectTab: (tab: "geometry" | "evolution" | "comparison") => void;
}

export const ObservatoryHeader: React.FC<ObservatoryHeaderProps> = ({
  architectures,
  selectedArch,
  onSelectArch,
  availableLayers,
  selectedLayer,
  onSelectLayer,
  dataBudgets,
  selectedBudget,
  onSelectBudget,
  spatialPolicy,
  onSelectSpatialPolicy,
  normPolicy,
  onSelectNormPolicy,
  distanceMetric,
  onSelectDistanceMetric,
  activeTab,
  onSelectTab,
}) => {
  return (
    <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Brand & Title */}
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-indigo-600 shadow-lg shadow-cyan-500/20 text-white font-mono font-bold text-lg">
              Ψ
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold font-mono tracking-tight text-zinc-100">
                  PRISM OBSERVATORY
                </h1>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
                  Phase 14 &bull; Representation Geometry
                </span>
              </div>
              <p className="text-xs text-zinc-400">
                Probing manifold structure, nearest neighbors, compactness, & separation
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center bg-zinc-900/90 p-1 rounded-lg border border-zinc-800">
            <button
              onClick={() => onSelectTab("geometry")}
              className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
                activeTab === "geometry"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              1. Geometry Inspector
            </button>
            <button
              onClick={() => onSelectTab("evolution")}
              className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
                activeTab === "evolution"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              2. Layer Evolution
            </button>
            <button
              onClick={() => onSelectTab("comparison")}
              className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
                activeTab === "comparison"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              3. Cross-Architecture Benchmarks
            </button>
          </div>
        </div>

        {/* Parametric Controls Bar */}
        <div className="mt-3 pt-3 border-t border-zinc-800/60 flex flex-wrap items-center gap-3 text-xs">
          {/* Architecture Selector */}
          <div className="flex items-center gap-1.5 bg-zinc-900/60 px-2.5 py-1.5 rounded-md border border-zinc-800">
            <span className="text-zinc-500 font-mono">Arch:</span>
            <select
              value={selectedArch}
              onChange={(e) => onSelectArch(e.target.value)}
              className="bg-transparent text-zinc-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              {architectures.map((arch) => (
                <option key={arch} value={arch} className="bg-zinc-900 text-zinc-200">
                  {arch.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Layer Selector */}
          <div className="flex items-center gap-1.5 bg-zinc-900/60 px-2.5 py-1.5 rounded-md border border-zinc-800">
            <span className="text-zinc-500 font-mono">Layer:</span>
            <select
              value={selectedLayer}
              onChange={(e) => onSelectLayer(e.target.value)}
              className="bg-transparent text-zinc-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              {availableLayers.map((layer) => (
                <option key={layer} value={layer} className="bg-zinc-900 text-zinc-200">
                  {layer}
                </option>
              ))}
            </select>
          </div>

          {/* Data Budget */}
          <div className="flex items-center gap-1.5 bg-zinc-900/60 px-2.5 py-1.5 rounded-md border border-zinc-800">
            <span className="text-zinc-500 font-mono">Budget:</span>
            <select
              value={selectedBudget}
              onChange={(e) => onSelectBudget(parseFloat(e.target.value))}
              className="bg-transparent text-zinc-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              {dataBudgets.map((b) => (
                <option key={b} value={b} className="bg-zinc-900 text-zinc-200">
                  {Math.round(b * 100)}% Data
                </option>
              ))}
            </select>
          </div>

          {/* Transform Policy */}
          <div className="flex items-center gap-1.5 bg-zinc-900/60 px-2.5 py-1.5 rounded-md border border-zinc-800">
            <span className="text-zinc-500 font-mono">Transform:</span>
            <select
              value={spatialPolicy}
              onChange={(e) => onSelectSpatialPolicy(e.target.value as SpatialTransformation)}
              className="bg-transparent text-zinc-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              <option value="global_average_pool" className="bg-zinc-900 text-zinc-200">
                Global Avg Pool
              </option>
              <option value="flatten" className="bg-zinc-900 text-zinc-200">
                Flatten
              </option>
            </select>
          </div>

          {/* Normalization */}
          <div className="flex items-center gap-1.5 bg-zinc-900/60 px-2.5 py-1.5 rounded-md border border-zinc-800">
            <span className="text-zinc-500 font-mono">Norm:</span>
            <select
              value={normPolicy}
              onChange={(e) => onSelectNormPolicy(e.target.value as NormalizationPolicy)}
              className="bg-transparent text-zinc-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              <option value="none" className="bg-zinc-900 text-zinc-200">
                None
              </option>
              <option value="l2_normalize" className="bg-zinc-900 text-zinc-200">
                L2 Normalize
              </option>
              <option value="standardize" className="bg-zinc-900 text-zinc-200">
                Standardize
              </option>
            </select>
          </div>

          {/* Metric */}
          <div className="flex items-center gap-1.5 bg-zinc-900/60 px-2.5 py-1.5 rounded-md border border-zinc-800">
            <span className="text-zinc-500 font-mono">Metric:</span>
            <select
              value={distanceMetric}
              onChange={(e) => onSelectDistanceMetric(e.target.value as DistanceMetric)}
              className="bg-transparent text-zinc-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              <option value="euclidean" className="bg-zinc-900 text-zinc-200">
                Euclidean
              </option>
              <option value="cosine_similarity" className="bg-zinc-900 text-zinc-200">
                Cosine Sim
              </option>
              <option value="cosine_distance" className="bg-zinc-900 text-zinc-200">
                Cosine Dist
              </option>
            </select>
          </div>
        </div>
      </div>
    </header>
  );
};
