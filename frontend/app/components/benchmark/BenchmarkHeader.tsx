"use client";

import React from "react";
import { BenchmarkCampaignPayload, CampaignCoverageSummaryPayload } from "../../benchmarkData";

interface BenchmarkHeaderProps {
  campaign: BenchmarkCampaignPayload;
  coverage: CampaignCoverageSummaryPayload;
  selectedArch: string;
  onSelectArch: (arch: string) => void;
  selectedObjective: string;
  onSelectObjective: (obj: string) => void;
  activeTab: "overview" | "profiles" | "synthesis" | "pareto" | "findings" | "gaps";
  onSelectTab: (tab: "overview" | "profiles" | "synthesis" | "pareto" | "findings" | "gaps") => void;
  onOpenReportBuilder: () => void;
}

export const BenchmarkHeader: React.FC<BenchmarkHeaderProps> = ({
  campaign,
  coverage,
  selectedArch,
  onSelectArch,
  selectedObjective,
  onSelectObjective,
  activeTab,
  onSelectTab,
  onOpenReportBuilder,
}) => {
  const percentStr = `${(coverage.completion_fraction * 100).toFixed(1)}%`;

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 sticky top-12 z-30 shadow-xl backdrop-blur-md bg-slate-900/95">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Title and Fingerprint */}
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-800/80 uppercase">
              Phase 24
            </span>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <span>🏛️</span> Cross-Paradigm Benchmark & Evidence Synthesis Observatory
            </h1>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
            <span>Campaign: <strong className="text-slate-200">{campaign.campaign_id}</strong></span>
            <span className="text-slate-600">•</span>
            <span className="truncate max-w-xs" title={campaign.fingerprint}>
              Fingerprint: <code className="text-cyan-400">{campaign.fingerprint.slice(0, 16)}...</code>
            </span>
            <span className="text-slate-600">•</span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-emerald-400 font-bold">{percentStr} Verified</span>
            </span>
          </div>
        </div>

        {/* Right: Controls & Actions */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Architecture Filter */}
          <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs">
            <span className="text-slate-400 font-medium">Arch:</span>
            <select
              value={selectedArch}
              onChange={(e) => onSelectArch(e.target.value)}
              className="bg-transparent text-cyan-300 font-bold outline-none cursor-pointer"
            >
              <option value="all" className="bg-slate-900 text-slate-200">All Architectures</option>
              {campaign.architectures.map((a) => (
                <option key={a} value={a} className="bg-slate-900 text-slate-200 uppercase">
                  {a}
                </option>
              ))}
            </select>
          </div>

          {/* Pretraining Objective Filter */}
          <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs">
            <span className="text-slate-400 font-medium">Objective:</span>
            <select
              value={selectedObjective}
              onChange={(e) => onSelectObjective(e.target.value)}
              className="bg-transparent text-cyan-300 font-bold outline-none cursor-pointer"
            >
              <option value="all" className="bg-slate-900 text-slate-200">All Objectives</option>
              {campaign.objectives.map((o) => (
                <option key={o} value={o} className="bg-slate-900 text-slate-200 capitalize">
                  {o}
                </option>
              ))}
            </select>
          </div>

          {/* Report Builder Button */}
          <button
            id="btn-open-report-builder"
            onClick={onOpenReportBuilder}
            className="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-600/20 transition-all flex items-center gap-1.5"
          >
            <span>📜</span> Generate Report
          </button>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex items-center gap-2 mt-4 pt-3 border-t border-slate-800/80 overflow-x-auto">
        {(
          [
            { id: "overview", label: "📊 Overview & Tables", icon: "📊" },
            { id: "profiles", label: "🕸️ 10D Representation Profiles", icon: "🕸️" },
            { id: "synthesis", label: "🔬 Architecture & Objective Synthesis", icon: "🔬" },
            { id: "pareto", label: "📈 Pareto & Tradeoffs", icon: "📈" },
            { id: "findings", label: "💡 Evidence-Backed Findings", icon: "💡" },
            { id: "gaps", label: "🔍 Evidence Gaps & Planning", icon: "🔍" },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            id={`tab-benchmark-${tab.id}`}
            onClick={() => onSelectTab(tab.id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === tab.id
                ? "bg-cyan-500 text-slate-950 font-black shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <span>{tab.icon}</span> {tab.label}
          </button>
        ))}
      </div>
    </header>
  );
};
