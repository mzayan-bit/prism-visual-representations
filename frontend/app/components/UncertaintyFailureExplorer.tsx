"use client";

import React, { useState } from "react";
import { UncertaintySampleItemPayload } from "../types";

interface UncertaintyFailureExplorerProps {
  samples: UncertaintySampleItemPayload[];
  failureCounts: Record<string, number>;
  onSelectSample?: (sample: UncertaintySampleItemPayload) => void;
}

const FAILURE_MODE_INFO: Record<
  string,
  { label: string; desc: string; badgeColor: string; icon: string }
> = {
  high_confidence_error: {
    label: "High-Confidence Error",
    desc: "Model predicted an incorrect class with over 80% confidence.",
    badgeColor: "bg-rose-500/20 text-rose-300 border-rose-500/40",
    icon: "🚨",
  },
  high_confidence_ood: {
    label: "High-Confidence OOD",
    desc: "Model assigned high confidence to an unfamiliar out-of-distribution sample.",
    badgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    icon: "⚠️",
  },
  low_confidence_correct: {
    label: "Low-Confidence Correct",
    desc: "Correct prediction, but model hesitated with under 45% confidence.",
    badgeColor: "bg-blue-500/20 text-blue-300 border-blue-500/40",
    icon: "❓",
  },
  id_representation_outlier: {
    label: "ID Representation Outlier",
    desc: "In-distribution sample situated far from its class cluster centroid.",
    badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/40",
    icon: "🌌",
  },
  ood_near_known_structure: {
    label: "OOD Near Known Structure",
    desc: "OOD input whose learned representation intrudes into an in-distribution class cluster.",
    badgeColor: "bg-orange-500/20 text-orange-300 border-orange-500/40",
    icon: "🎭",
  },
  corruption_overconfidence: {
    label: "Corruption Overconfidence",
    desc: "Model maintained uncalibrated high confidence despite heavy image corruption.",
    badgeColor: "bg-red-500/20 text-red-300 border-red-500/40",
    icon: "💥",
  },
};

export const UncertaintyFailureExplorer: React.FC<UncertaintyFailureExplorerProps> = ({
  samples,
  failureCounts,
  onSelectSample,
}) => {
  const [selectedFailureFilter, setSelectedFailureFilter] = useState<string>("all");
  const [selectedSampleItem, setSelectedSampleItem] =
    useState<UncertaintySampleItemPayload | null>(null);

  // Filter samples based on heuristic / category
  const filteredSamples = samples.filter((s) => {
    if (selectedFailureFilter === "all") return true;
    if (selectedFailureFilter === "high_confidence_error") {
      return !s.is_correct && s.category === "in_distribution" && s.confidence >= 0.75;
    }
    if (selectedFailureFilter === "high_confidence_ood") {
      return s.category !== "in_distribution" && s.confidence >= 0.7;
    }
    if (selectedFailureFilter === "low_confidence_correct") {
      return s.is_correct && s.confidence < 0.5;
    }
    if (selectedFailureFilter === "id_representation_outlier") {
      return s.category === "in_distribution" && s.centroid_distance > 1.2;
    }
    if (selectedFailureFilter === "ood_near_known_structure") {
      return s.category !== "in_distribution" && s.centroid_distance < 0.8;
    }
    return true;
  });

  const totalFailures = Object.values(failureCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-red-500/20 border border-red-500/40 flex items-center justify-center text-red-400 font-bold text-sm">
            🔍
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-200">
              Uncertainty & Calibration Failure Mode Taxonomy
            </h2>
            <p className="text-xs text-slate-400">
              Systematic categorization of overconfident errors, OOD illusions, and manifold boundary anomalies
            </p>
          </div>
        </div>

        <div className="text-xs bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-400">
          Total Flagged Failures:{" "}
          <span className="font-mono font-bold text-rose-400">
            {totalFailures}
          </span>
        </div>
      </div>

      {/* Failure Mode Filter Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <button
          onClick={() => setSelectedFailureFilter("all")}
          className={`p-2.5 rounded-lg border text-left transition-all flex flex-col justify-between ${
            selectedFailureFilter === "all"
              ? "bg-slate-800 border-cyan-500/60 shadow"
              : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-200">All Cases</span>
            <span className="text-xs font-mono font-bold text-slate-300">
              {samples.length}
            </span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Full sample pool</div>
        </button>

        {Object.entries(FAILURE_MODE_INFO).map(([key, info]) => {
          const count = failureCounts[key] ?? 0;
          const isSelected = selectedFailureFilter === key;
          return (
            <button
              key={key}
              onClick={() => setSelectedFailureFilter(key)}
              className={`p-2.5 rounded-lg border text-left transition-all flex flex-col justify-between ${
                isSelected
                  ? `bg-slate-800 ${info.badgeColor} shadow`
                  : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200 flex items-center gap-1 truncate">
                  <span>{info.icon}</span>
                  <span className="truncate">{info.label.split(" ")[0]}</span>
                </span>
                <span className="text-xs font-mono font-bold text-amber-400 ml-1">
                  {count}
                </span>
              </div>
              <div className="text-[10px] text-slate-500 truncate mt-1">
                {info.label}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Filter Description */}
      {selectedFailureFilter !== "all" && FAILURE_MODE_INFO[selectedFailureFilter] && (
        <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 text-xs text-slate-300 flex items-center gap-2">
          <span className="text-base">
            {FAILURE_MODE_INFO[selectedFailureFilter].icon}
          </span>
          <div>
            <span className="font-semibold text-slate-100">
              {FAILURE_MODE_INFO[selectedFailureFilter].label}:{" "}
            </span>
            <span className="text-slate-400">
              {FAILURE_MODE_INFO[selectedFailureFilter].desc}
            </span>
          </div>
        </div>
      )}

      {/* Samples Table and Detail View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Table list */}
        <div className="lg:col-span-2 overflow-x-auto max-h-80 overflow-y-auto border border-slate-800 rounded-lg">
          <table className="w-full text-left text-xs border-collapse font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-[10px] text-slate-400 uppercase bg-slate-950/80 sticky top-0">
                <th className="py-2 px-2.5">Sample ID</th>
                <th className="py-2 px-2">Category</th>
                <th className="py-2 px-2">Pred / True</th>
                <th className="py-2 px-2 text-right">Conf</th>
                <th className="py-2 px-2 text-right">Entropy</th>
                <th className="py-2 px-2 text-right">Centroid Dist</th>
                <th className="py-2 px-2 text-center">OOD Flag</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-[11px]">
              {filteredSamples.map((sample) => {
                const isSelected =
                  selectedSampleItem?.sample_id === sample.sample_id;
                const isOOD = sample.category !== "in_distribution";

                return (
                  <tr
                    key={sample.sample_id}
                    onClick={() => {
                      setSelectedSampleItem(sample);
                      if (onSelectSample) onSelectSample(sample);
                    }}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-slate-800 border-l-2 border-l-amber-400"
                        : "hover:bg-slate-800/40"
                    }`}
                  >
                    <td className="py-2 px-2.5 font-semibold text-slate-200">
                      {sample.sample_id}
                    </td>
                    <td className="py-2 px-2">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase ${
                          isOOD
                            ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                            : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                        }`}
                      >
                        {sample.category.replace(/_/g, " ")}
                      </span>
                    </td>
                    <td className="py-2 px-2">
                      <span
                        className={
                          sample.is_correct ? "text-emerald-400" : "text-rose-400"
                        }
                      >
                        C{sample.predicted_class}
                      </span>
                      {sample.true_class !== null && (
                        <span className="text-slate-500 text-[10px]">
                          {" "}
                          / C{sample.true_class}
                        </span>
                      )}
                    </td>
                    <td
                      className={`py-2 px-2 text-right font-bold ${
                        sample.confidence > 0.8
                          ? !sample.is_correct || isOOD
                            ? "text-rose-400 font-extrabold"
                            : "text-emerald-400"
                          : "text-amber-400"
                      }`}
                    >
                      {(sample.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-2 text-right text-slate-300">
                      {sample.entropy.toFixed(3)}
                    </td>
                    <td className="py-2 px-2 text-right text-cyan-400">
                      {sample.centroid_distance.toFixed(3)}
                    </td>
                    <td className="py-2 px-2 text-center">
                      <span
                        className={`inline-block w-2 h-2 rounded-full ${
                          sample.is_ood_detected
                            ? "bg-rose-500"
                            : "bg-emerald-500"
                        }`}
                        title={
                          sample.is_ood_detected
                            ? "Flagged as OOD"
                            : "Accepted as ID"
                        }
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Selected Sample Detail Box */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 flex flex-col justify-between">
          <div>
            <div className="text-xs font-semibold text-slate-300 pb-2 border-b border-slate-800 flex items-center justify-between">
              <span>Failure Case Diagnostics</span>
              {selectedSampleItem && (
                <span className="font-mono text-cyan-400">
                  {selectedSampleItem.sample_id}
                </span>
              )}
            </div>

            {selectedSampleItem ? (
              <div className="mt-3 flex flex-col gap-2.5 text-xs font-mono">
                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">Input Domain:</span>
                  <span
                    className={
                      selectedSampleItem.category === "in_distribution"
                        ? "text-emerald-400 font-bold"
                        : "text-amber-400 font-bold uppercase"
                    }
                  >
                    {selectedSampleItem.category.replace(/_/g, " ")}
                  </span>
                </div>

                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">Classification:</span>
                  <span
                    className={
                      selectedSampleItem.is_correct
                        ? "text-emerald-400"
                        : "text-rose-400 font-bold"
                    }
                  >
                    {selectedSampleItem.is_correct ? "CORRECT" : "INCORRECT"}{" "}
                    (Pred {selectedSampleItem.predicted_class})
                  </span>
                </div>

                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">Max Softmax Prob:</span>
                  <span className="text-cyan-300 font-bold">
                    {(selectedSampleItem.confidence * 100).toFixed(2)}%
                  </span>
                </div>

                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">Predictive Entropy:</span>
                  <span className="text-slate-200">
                    {selectedSampleItem.entropy.toFixed(4)}
                  </span>
                </div>

                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">Class Centroid Dist:</span>
                  <span className="text-amber-400">
                    {selectedSampleItem.centroid_distance.toFixed(4)}
                  </span>
                </div>

                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">kNN Distance:</span>
                  <span className="text-purple-400">
                    {selectedSampleItem.knn_distance.toFixed(4)}
                  </span>
                </div>

                <div className="flex justify-between items-center py-1 border-b border-slate-900">
                  <span className="text-slate-500">OOD Detector Status:</span>
                  <span
                    className={`font-bold ${
                      selectedSampleItem.is_ood_detected
                        ? "text-rose-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {selectedSampleItem.is_ood_detected ? "DETECTED OOD" : "PASSED (ID)"}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-slate-500 text-xs">
                Select any sample row from the table to view its diagnostic breakdown.
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-[10px] text-slate-500">
            PRISM Uncertainty Taxonomy • CPU-deterministic failure auditing
          </div>
        </div>
      </div>
    </div>
  );
};
