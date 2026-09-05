"use client";

import React, { useState } from "react";
import {
  UncertaintyArchitectureComparisonPayload,
  UncertaintyObjectiveComparisonPayload,
} from "../types";

interface UncertaintyObjectiveComparisonCardProps {
  objectiveComparisons: UncertaintyObjectiveComparisonPayload[];
  architectureComparisons: UncertaintyArchitectureComparisonPayload[];
  selectedObjective: string;
  onSelectObjective: (obj: string) => void;
  selectedArch: string;
  onSelectArch: (arch: string) => void;
}

export const UncertaintyObjectiveComparisonCard: React.FC<
  UncertaintyObjectiveComparisonCardProps
> = ({
  objectiveComparisons,
  architectureComparisons,
  selectedObjective,
  onSelectObjective,
  selectedArch,
  onSelectArch,
}) => {
  const [activeTab, setActiveTab] = useState<"objectives" | "architectures">(
    "objectives"
  );
  const [sortKey, setSortKey] = useState<
    "accuracy" | "ece" | "brier_score" | "nll" | "ood_msp_auroc" | "ood_knn_auroc"
  >("ece");
  const [sortAsc, setSortAsc] = useState(true);

  // Sorting helper
  const handleSort = (key: typeof sortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === "ece" || key === "brier_score" || key === "nll"); // lower is better for error metrics
    }
  };

  const sortedObjectives = [...objectiveComparisons].sort((a, b) => {
    const valA = a[sortKey] ?? 0;
    const valB = b[sortKey] ?? 0;
    return sortAsc ? valA - valB : valB - valA;
  });

  const sortedArchitectures = [...architectureComparisons].sort((a, b) => {
    const valA = a[sortKey as keyof UncertaintyArchitectureComparisonPayload] as number ?? 0;
    const valB = b[sortKey as keyof UncertaintyArchitectureComparisonPayload] as number ?? 0;
    return sortAsc ? valA - valB : valB - valA;
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 font-bold text-sm">
            ⚖️
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-200">
              Comparative Uncertainty & Calibration Benchmarks
            </h2>
            <p className="text-xs text-slate-400">
              Evaluating how pretraining objectives and inductive biases influence calibration quality and OOD separation
            </p>
          </div>
        </div>

        {/* Tab switch */}
        <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab("objectives")}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
              activeTab === "objectives"
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            By Pretraining Objective
          </button>
          <button
            onClick={() => setActiveTab("architectures")}
            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
              activeTab === "architectures"
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            By Architecture
          </button>
        </div>
      </div>

      {/* Key Insights Callout Banner */}
      <div className="bg-indigo-950/30 border border-indigo-500/30 rounded-lg p-3.5 flex items-start gap-3">
        <div className="text-indigo-400 text-base mt-0.5">💡</div>
        <div className="text-xs text-indigo-200 leading-relaxed">
          <span className="font-semibold text-indigo-100">Empirical Finding: </span>
          Self-supervised objectives (<span className="font-mono text-cyan-300">SimCLR</span>) and multi-modal alignment (<span className="font-mono text-indigo-300">Vision-Language</span>) produce representations with stronger geometric separation in feature space (higher kNN & Centroid AUROC), whereas <span className="font-mono text-emerald-300">Supervised</span> models tend to be overconfident on out-of-distribution inputs (lower MSP AUROC) despite higher in-domain test accuracy.
        </div>
      </div>

      {/* Comparative Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse font-mono">
          <thead>
            <tr className="border-b border-slate-800 text-[11px] text-slate-400 bg-slate-950/60 uppercase">
              <th className="py-2.5 px-3">
                {activeTab === "objectives" ? "Objective" : "Architecture"}
              </th>
              <th
                onClick={() => handleSort("accuracy")}
                className="py-2.5 px-2.5 text-right cursor-pointer hover:text-cyan-300 transition-colors"
              >
                Accuracy {sortKey === "accuracy" && (sortAsc ? "▲" : "▼")}
              </th>
              <th
                onClick={() => handleSort("ece")}
                className="py-2.5 px-2.5 text-right cursor-pointer hover:text-cyan-300 transition-colors"
              >
                ECE ↓ {sortKey === "ece" && (sortAsc ? "▲" : "▼")}
              </th>
              <th
                onClick={() => handleSort("brier_score")}
                className="py-2.5 px-2.5 text-right cursor-pointer hover:text-cyan-300 transition-colors"
              >
                Brier ↓ {sortKey === "brier_score" && (sortAsc ? "▲" : "▼")}
              </th>
              <th
                onClick={() => handleSort("nll")}
                className="py-2.5 px-2.5 text-right cursor-pointer hover:text-cyan-300 transition-colors"
              >
                NLL ↓ {sortKey === "nll" && (sortAsc ? "▲" : "▼")}
              </th>
              <th
                onClick={() => handleSort("ood_msp_auroc")}
                className="py-2.5 px-2.5 text-right cursor-pointer hover:text-cyan-300 transition-colors"
              >
                MSP AUROC ↑ {sortKey === "ood_msp_auroc" && (sortAsc ? "▲" : "▼")}
              </th>
              <th
                onClick={() => handleSort("ood_knn_auroc")}
                className="py-2.5 px-2.5 text-right cursor-pointer hover:text-cyan-300 transition-colors"
              >
                kNN AUROC ↑ {sortKey === "ood_knn_auroc" && (sortAsc ? "▲" : "▼")}
              </th>
              {activeTab === "objectives" && (
                <th className="py-2.5 px-2.5 text-right">Opt Temp (T*)</th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-[11px]">
            {activeTab === "objectives"
              ? sortedObjectives.map((obj) => {
                  const isSelected =
                    selectedObjective.toLowerCase() ===
                    obj.objective.toLowerCase();
                  return (
                    <tr
                      key={obj.objective}
                      onClick={() => onSelectObjective(obj.objective)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-indigo-950/40 border-l-2 border-l-indigo-500"
                          : "hover:bg-slate-800/40"
                      }`}
                    >
                      <td className="py-2 px-3 font-semibold text-slate-200 flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isSelected ? "bg-indigo-400" : "bg-slate-600"
                          }`}
                        />
                        {obj.objective.toUpperCase()}
                      </td>
                      <td className="py-2 px-2.5 text-right text-emerald-400 font-bold">
                        {(obj.accuracy * 100).toFixed(1)}%
                      </td>
                      <td
                        className={`py-2 px-2.5 text-right font-bold ${
                          obj.ece < 0.08 ? "text-emerald-400" : "text-amber-400"
                        }`}
                      >
                        {obj.ece.toFixed(4)}
                      </td>
                      <td className="py-2 px-2.5 text-right text-slate-300">
                        {obj.brier_score.toFixed(4)}
                      </td>
                      <td className="py-2 px-2.5 text-right text-slate-300">
                        {obj.nll.toFixed(3)}
                      </td>
                      <td className="py-2 px-2.5 text-right text-cyan-400">
                        {(obj.ood_msp_auroc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-2.5 text-right text-purple-400 font-bold">
                        {(obj.ood_knn_auroc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-2.5 text-right text-amber-300">
                        {obj.temperature.toFixed(2)}
                      </td>
                    </tr>
                  );
                })
              : sortedArchitectures.map((arch) => {
                  const isSelected =
                    selectedArch.toLowerCase() ===
                    arch.architecture.toLowerCase();
                  return (
                    <tr
                      key={arch.architecture}
                      onClick={() => onSelectArch(arch.architecture)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-cyan-950/40 border-l-2 border-l-cyan-500"
                          : "hover:bg-slate-800/40"
                      }`}
                    >
                      <td className="py-2 px-3 font-semibold text-slate-200 flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isSelected ? "bg-cyan-400" : "bg-slate-600"
                          }`}
                        />
                        {arch.architecture}
                      </td>
                      <td className="py-2 px-2.5 text-right text-emerald-400 font-bold">
                        {(arch.accuracy * 100).toFixed(1)}%
                      </td>
                      <td
                        className={`py-2 px-2.5 text-right font-bold ${
                          arch.ece < 0.08
                            ? "text-emerald-400"
                            : "text-amber-400"
                        }`}
                      >
                        {arch.ece.toFixed(4)}
                      </td>
                      <td className="py-2 px-2.5 text-right text-slate-300">
                        {arch.brier_score.toFixed(4)}
                      </td>
                      <td className="py-2 px-2.5 text-right text-slate-300">
                        {arch.nll.toFixed(3)}
                      </td>
                      <td className="py-2 px-2.5 text-right text-cyan-400">
                        {(arch.ood_msp_auroc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-2.5 text-right text-purple-400 font-bold">
                        {(arch.ood_knn_auroc * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
