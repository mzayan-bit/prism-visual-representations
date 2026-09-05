"use client";

import React from "react";

interface UncertaintyHeaderProps {
  architectures: string[];
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  objectives: string[];
  selectedObjective: string;
  onSelectObjective: (obj: string) => void;
  calibrationModes: string[];
  selectedCalibrationMode: string;
  onSelectCalibrationMode: (mode: string) => void;
  oodScoreMethods: string[];
  selectedOODMethod: string;
  onSelectOODMethod: (method: string) => void;
  corruptions: string[];
  selectedCorruption: string;
  onSelectCorruption: (corr: string) => void;
  binCounts: number[];
  selectedBinCount: number;
  onSelectBinCount: (bins: number) => void;
}

export const UncertaintyHeader: React.FC<UncertaintyHeaderProps> = ({
  architectures,
  selectedArch,
  onSelectArch,
  objectives,
  selectedObjective,
  onSelectObjective,
  calibrationModes,
  selectedCalibrationMode,
  onSelectCalibrationMode,
  oodScoreMethods,
  selectedOODMethod,
  onSelectOODMethod,
  corruptions,
  selectedCorruption,
  onSelectCorruption,
  binCounts,
  selectedBinCount,
  onSelectBinCount,
}) => {
  return (
    <div className="bg-slate-900 border-b border-slate-800 p-4 sticky top-12 z-30 shadow-md">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Title & Badge */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400 font-bold text-lg shadow-inner">
            🎲
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-100">
                Uncertainty, Calibration & Out-of-Distribution Laboratory
              </h1>
              <span className="bg-amber-950/80 text-amber-400 text-[10px] font-mono font-semibold px-2 py-0.5 rounded border border-amber-500/30">
                PHASE 23
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Predictive Confidence • Reliability Diagrams • Temperature Scaling • OOD Representation Novelty • Corruption Uncertainty
            </p>
          </div>
        </div>

        {/* Global Selectors */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          {/* Architecture Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Arch:</span>
            <select
              id="select-uncertainty-arch"
              value={selectedArch}
              onChange={(e) => onSelectArch(e.target.value)}
              className="bg-slate-900 text-cyan-400 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              {architectures.map((arch) => (
                <option key={arch} value={arch}>
                  {arch}
                </option>
              ))}
            </select>
          </div>

          {/* Objective Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Objective:</span>
            <select
              id="select-uncertainty-objective"
              value={selectedObjective}
              onChange={(e) => onSelectObjective(e.target.value)}
              className="bg-slate-900 text-indigo-400 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              {objectives.map((obj) => (
                <option key={obj} value={obj}>
                  {obj.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Calibration Mode Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Mode:</span>
            <select
              id="select-uncertainty-mode"
              value={selectedCalibrationMode}
              onChange={(e) => onSelectCalibrationMode(e.target.value)}
              className="bg-slate-900 text-emerald-400 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-emerald-500 cursor-pointer"
            >
              {calibrationModes.map((mode) => (
                <option key={mode} value={mode}>
                  {mode === "temperature_scaled" ? "Temp-Scaled" : "Uncalibrated"}
                </option>
              ))}
            </select>
          </div>

          {/* OOD Score Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">OOD Score:</span>
            <select
              id="select-uncertainty-ood-score"
              value={selectedOODMethod}
              onChange={(e) => onSelectOODMethod(e.target.value)}
              className="bg-slate-900 text-amber-400 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-amber-500 cursor-pointer"
            >
              {oodScoreMethods.map((m) => (
                <option key={m} value={m}>
                  {m.replace(/_/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Corruption Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Corruption:</span>
            <select
              id="select-uncertainty-corruption"
              value={selectedCorruption}
              onChange={(e) => onSelectCorruption(e.target.value)}
              className="bg-slate-900 text-rose-400 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              {corruptions.map((c) => (
                <option key={c} value={c}>
                  {c.replace(/_/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Bins Selector */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1.5 rounded-lg border border-slate-700/80">
            <span className="text-slate-400 font-medium">Bins:</span>
            <select
              id="select-uncertainty-bins"
              value={selectedBinCount}
              onChange={(e) => onSelectBinCount(Number(e.target.value))}
              className="bg-slate-900 text-cyan-300 font-semibold rounded px-2 py-0.5 border border-slate-700 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              {binCounts.map((b) => (
                <option key={b} value={b}>
                  {b} Bins
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};
