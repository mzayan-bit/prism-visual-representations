"use client";

import React, { useState } from "react";
import {
  getBenchmarkCampaign,
  getBenchmarkDataset,
  getBenchmarkTables,
  getCoverageMatrix,
  getCoverageSummary,
  getEvidenceGaps,
  getMissingExperimentPlan,
  getParetoAnalysis,
  getRepresentationProfiles,
  getResearchFindings,
  getTradeoffAnalysis,
} from "../../benchmarkData";
import { ArchitectureSynthesisCard } from "./ArchitectureSynthesisCard";
import { BenchmarkHeader } from "./BenchmarkHeader";
import { BenchmarkOverviewStrip } from "./BenchmarkOverviewStrip";
import { BenchmarkTableCard } from "./BenchmarkTableCard";
import { CoverageMatrixCard } from "./CoverageMatrixCard";
import { EvidenceGapPanelCard } from "./EvidenceGapPanelCard";
import { FindingsPanelCard } from "./FindingsPanelCard";
import { MultiDimensionProfileCard } from "./MultiDimensionProfileCard";
import { ObjectiveSynthesisCard } from "./ObjectiveSynthesisCard";
import { ParetoExplorerCard } from "./ParetoExplorerCard";
import { ProvenanceDrawer } from "./ProvenanceDrawer";
import { ReportBuilderCard } from "./ReportBuilderCard";
import { ReportPreviewModal } from "./ReportPreviewModal";
import { TradeoffExplorerCard } from "./TradeoffExplorerCard";

export const BenchmarkObservatoryView: React.FC = () => {
  const dataset = getBenchmarkDataset();
  const campaign = getBenchmarkCampaign();
  const coverageSummary = getCoverageSummary();
  const coverageMatrix = getCoverageMatrix();
  const tables = getBenchmarkTables();
  const profiles = getRepresentationProfiles();
  const pareto = getParetoAnalysis();
  const tradeoffs = getTradeoffAnalysis();
  const findings = getResearchFindings();
  const gaps = getEvidenceGaps();
  const missingPlan = getMissingExperimentPlan();

  // State
  const [selectedArch, setSelectedArch] = useState<string>("all");
  const [selectedObjective, setSelectedObjective] = useState<string>("all");
  const [activeTab, setActiveTab] = useState<
    "overview" | "profiles" | "synthesis" | "pareto" | "findings" | "gaps"
  >("overview");

  // Provenance drawer state
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedProvenanceInfo, setSelectedProvenanceInfo] = useState<Record<string, unknown> | null>(null);

  // Report Modal state
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [reportFormat, setReportFormat] = useState<"markdown" | "json" | "csv">(
    "markdown"
  );

  const handleCellClick = (info: Record<string, unknown>) => {
    setSelectedProvenanceInfo(info);
    setIsDrawerOpen(true);
  };

  const handleOpenReportBuilder = () => {
    setIsReportModalOpen(true);
  };

  const handleGenerateReport = (opts: {
    format: "markdown" | "json" | "csv";
    sections: string[];
  }) => {
    setReportFormat(opts.format);
    setIsReportModalOpen(true);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Benchmark Header */}
      <BenchmarkHeader
        campaign={campaign}
        coverage={coverageSummary}
        selectedArch={selectedArch}
        onSelectArch={setSelectedArch}
        selectedObjective={selectedObjective}
        onSelectObjective={setSelectedObjective}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onOpenReportBuilder={handleOpenReportBuilder}
      />

      {/* Metrics Overview Strip */}
      <BenchmarkOverviewStrip
        coverage={coverageSummary}
        totalObservations={dataset.all_cells?.length || 0}
      />

      {/* Main Content Area */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* TAB 1: OVERVIEW & TABLES */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <CoverageMatrixCard
                  coverageMatrix={coverageMatrix}
                  onSelectCell={(row, col) =>
                    handleCellClick({
                      rowFactor: row,
                      colFactor: col,
                      matrix: "coverage",
                    })
                  }
                />
              </div>
              <div className="lg:col-span-2">
                <BenchmarkTableCard
                  tables={tables}
                  onSelectCell={handleCellClick}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <FindingsPanelCard findings={findings.slice(0, 3)} />
              <ReportBuilderCard onGenerateReport={handleGenerateReport} />
            </div>
          </div>
        )}

        {/* TAB 2: PROFILES */}
        {activeTab === "profiles" && (
          <div className="space-y-6">
            <MultiDimensionProfileCard profiles={profiles} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ArchitectureSynthesisCard
                synthesis={dataset.architecture_synthesis}
              />
              <ObjectiveSynthesisCard
                synthesis={dataset.objective_synthesis}
              />
            </div>
          </div>
        )}

        {/* TAB 3: SYNTHESIS */}
        {activeTab === "synthesis" && (
          <div className="space-y-6">
            <ArchitectureSynthesisCard
              synthesis={dataset.architecture_synthesis}
            />
            <ObjectiveSynthesisCard
              synthesis={dataset.objective_synthesis}
            />
          </div>
        )}

        {/* TAB 4: PARETO & TRADEOFFS */}
        {activeTab === "pareto" && (
          <div className="space-y-6">
            <ParetoExplorerCard
              pareto={pareto}
              onSelectExperiment={(expId) =>
                handleCellClick({ experiment_id: expId, context: "pareto" })
              }
            />
            <TradeoffExplorerCard tradeoffs={tradeoffs} />
          </div>
        )}

        {/* TAB 5: FINDINGS */}
        {activeTab === "findings" && (
          <div className="space-y-6">
            <FindingsPanelCard findings={findings} />
          </div>
        )}

        {/* TAB 6: GAPS */}
        {activeTab === "gaps" && (
          <div className="space-y-6">
            <EvidenceGapPanelCard gaps={gaps} missingPlan={missingPlan} />
          </div>
        )}
      </main>

      {/* Provenance Drawer */}
      <ProvenanceDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        selectedInfo={selectedProvenanceInfo}
      />

      {/* Report Preview Modal */}
      <ReportPreviewModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        dataset={dataset}
        format={reportFormat}
      />
    </div>
  );
};
