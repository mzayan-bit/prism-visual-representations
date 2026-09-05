"use client";

import React, { useState } from "react";

interface ReportBuilderCardProps {
  onGenerateReport: (options: {
    format: "markdown" | "json" | "csv";
    sections: string[];
  }) => void;
}

export const ReportBuilderCard: React.FC<ReportBuilderCardProps> = ({
  onGenerateReport,
}) => {
  const [selectedFormat, setSelectedFormat] = useState<"markdown" | "json" | "csv">("markdown");
  const [selectedSections, setSelectedSections] = useState<string[]>([
    "executive_summary",
    "methodology",
    "benchmark_tables",
    "profiles",
    "findings",
    "pareto",
    "gaps",
    "reproducibility",
  ]);

  const availableSections = [
    { id: "executive_summary", label: "Executive Summary & Campaign Metadata" },
    { id: "methodology", label: "Experimental Methodology & Controlled Factors" },
    { id: "benchmark_tables", label: "Canonical Benchmark Result Tables" },
    { id: "profiles", label: "10-Dimensional Representation Profiles" },
    { id: "findings", label: "Evidence-Backed Scientific Findings" },
    { id: "pareto", label: "Multi-Objective Pareto Frontiers & Tradeoffs" },
    { id: "gaps", label: "Evidence Gaps & Missing Experiment Plans" },
    { id: "reproducibility", label: "Reproducibility Manifest & Checksums" },
  ];

  const toggleSection = (id: string) => {
    if (selectedSections.includes(id)) {
      setSelectedSections(selectedSections.filter((s) => s !== id));
    } else {
      setSelectedSections([...selectedSections, id]);
    }
  };

  return (
    <div className="p-5 bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl flex flex-col space-y-4">
      <div>
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>⚙️</span> Research Report Builder
        </h3>
        <p className="text-xs text-slate-400">
          Configure report sections and compilation format for publication-ready export
        </p>
      </div>

      {/* Format Selector */}
      <div className="space-y-1.5">
        <label className="text-xs font-mono font-bold text-slate-300">Export Format:</label>
        <div className="flex items-center gap-2">
          {(["markdown", "json", "csv"] as const).map((fmt) => (
            <button
              key={fmt}
              onClick={() => setSelectedFormat(fmt)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold uppercase transition-all ${
                selectedFormat === fmt
                  ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/20"
                  : "bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200"
              }`}
            >
              {fmt}
            </button>
          ))}
        </div>
      </div>

      {/* Sections Selector */}
      <div className="space-y-2">
        <label className="text-xs font-mono font-bold text-slate-300">Included Sections:</label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {availableSections.map((sec) => {
            const isChecked = selectedSections.includes(sec.id);

            return (
              <button
                key={sec.id}
                onClick={() => toggleSection(sec.id)}
                className={`p-2.5 rounded-lg border text-left text-xs font-mono transition-all flex items-center gap-2 ${
                  isChecked
                    ? "bg-cyan-950/40 border-cyan-700/80 text-cyan-300"
                    : "bg-slate-950 border-slate-800 text-slate-400"
                }`}
              >
                <span className={`w-3.5 h-3.5 rounded flex items-center justify-center border text-[10px] ${
                  isChecked ? "bg-cyan-600 border-cyan-400 text-white" : "border-slate-700"
                }`}>
                  {isChecked ? "✓" : ""}
                </span>
                <span className="truncate">{sec.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <button
        onClick={() =>
          onGenerateReport({
            format: selectedFormat,
            sections: selectedSections,
          })
        }
        className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white rounded-xl text-xs font-bold font-mono shadow-lg shadow-cyan-600/20 transition-all flex items-center justify-center gap-2"
      >
        <span>📄</span> Compile & Preview Report
      </button>
    </div>
  );
};
