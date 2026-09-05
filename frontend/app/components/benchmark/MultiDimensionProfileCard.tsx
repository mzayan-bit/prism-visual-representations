"use client";

import React, { useState } from "react";
import { RepresentationProfilePayload } from "../../benchmarkData";

interface MultiDimensionProfileCardProps {
  profiles: RepresentationProfilePayload[];
}

export const MultiDimensionProfileCard: React.FC<MultiDimensionProfileCardProps> = ({
  profiles,
}) => {
  const [selectedProfileId, setSelectedProfileId] = useState<string>(
    profiles[0]?.profile_id || ""
  );

  const activeProfile =
    profiles.find((p) => p.profile_id === selectedProfileId) || profiles[0];

  const axes: Array<{
    key: keyof Omit<RepresentationProfilePayload, "profile_id" | "architecture" | "objective" | "metadata">;
    label: string;
    desc: string;
  }> = [
    { key: "semantic_performance", label: "Semantic Accuracy", desc: "In-domain test accuracy" },
    { key: "geometry", label: "Geometry & Compactness", desc: "Inter-class separation & neighbor consistency" },
    { key: "label_efficiency", label: "Label Efficiency", desc: "Low data budget transfer gain" },
    { key: "transferability", label: "Linear Transferability", desc: "Downstream linear probe accuracy" },
    { key: "robustness", label: "Perturbation Robustness", desc: "1.0 - corruption accuracy drop" },
    { key: "spatial_transfer", label: "Dense Spatial Retention", desc: "Segmentation & detection probe mIoU" },
    { key: "temporal_transfer", label: "Temporal Video Transfer", desc: "Video sequence classification accuracy" },
    { key: "calibration", label: "Uncertainty Calibration", desc: "1.0 - Expected Calibration Error" },
    { key: "ood_separation", label: "OOD Discrimination", desc: "Out-of-distribution AUROC" },
    { key: "multimodal_alignment", label: "Multimodal Alignment", desc: "Cross-modal retrieval R@1 & zero-shot" },
  ];

  if (!activeProfile) {
    return (
      <div className="p-6 bg-slate-900/90 rounded-2xl border border-slate-800 text-center text-slate-400">
        No representation profiles available.
      </div>
    );
  }

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>🕸️</span> 10-Dimensional Representation Profile
          </h3>
          <p className="text-xs text-slate-400">
            Multi-axis representation characteristics without collapsing into a single scalar
          </p>
        </div>

        {/* Profile Selector */}
        <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs">
          <span className="text-slate-400 font-medium">Profile:</span>
          <select
            value={selectedProfileId}
            onChange={(e) => setSelectedProfileId(e.target.value)}
            className="bg-transparent text-cyan-300 font-bold outline-none cursor-pointer capitalize"
          >
            {profiles.map((p) => (
              <option key={p.profile_id} value={p.profile_id} className="bg-slate-900 text-slate-200">
                {p.architecture.toUpperCase()} • {p.objective}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Profile Header Banner */}
      <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 flex items-center justify-between text-xs font-mono">
        <span>
          Architecture: <strong className="text-cyan-400 uppercase">{activeProfile.architecture}</strong>
        </span>
        <span>
          Objective: <strong className="text-cyan-400 capitalize">{activeProfile.objective}</strong>
        </span>
        <span className="text-slate-400">
          ID: <code className="text-slate-300">{activeProfile.profile_id}</code>
        </span>
      </div>

      {/* 10 Axis Horizontal Bars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
        {axes.map((ax) => {
          const val = activeProfile[ax.key];
          const hasVal = val !== null && val !== undefined;
          const displayVal = hasVal ? (val * 100).toFixed(1) : "—";
          const barWidth = hasVal ? `${Math.min(100, Math.max(0, val * 100))}%` : "0%";

          return (
            <div
              key={ax.key}
              className="p-3 bg-slate-950/80 rounded-xl border border-slate-800/60 flex flex-col justify-between space-y-1.5"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300 font-bold">{ax.label}</span>
                <span className="text-cyan-400 font-bold">{displayVal}{hasVal ? "%" : ""}</span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-full rounded-full transition-all duration-500"
                  style={{ width: barWidth }}
                />
              </div>
              <span className="text-[10px] text-slate-500 truncate">{ax.desc}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
