import React, { useState } from "react";
import { NavigationRail } from "./NavigationRail";
import { ContextBar } from "./ContextBar";
import { ContextInspector, InspectorMetaItem } from "./ContextInspector";
import { AppMode } from "../ResearchPlatformNavigation";

export interface AppShellProps {
  currentMode: AppMode | "overview";
  onSelectMode: (mode: AppMode | "overview") => void;
  breadcrumb?: string[];
  activeDataset?: string;
  activeSeed?: number;
  isSynthetic?: boolean;
  inspectorTitle?: string;
  inspectorSubtitle?: string;
  inspectorMeta?: InspectorMetaItem[];
  inspectorProvenance?: {
    runId?: string;
    experimentId?: string;
    seed?: number;
    hardware?: string;
    fingerprint?: string;
  };
  inspectorContent?: React.ReactNode;
  headerActions?: React.ReactNode;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentMode,
  onSelectMode,
  breadcrumb = ["PRISM", "Overview"],
  activeDataset,
  activeSeed,
  isSynthetic = true,
  inspectorTitle,
  inspectorSubtitle,
  inspectorMeta,
  inspectorProvenance,
  inspectorContent,
  headerActions,
  children,
}) => {
  const [isNavCollapsed, setIsNavCollapsed] = useState(false);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex font-sans antialiased">
      {/* Left Navigation Rail */}
      <NavigationRail
        currentMode={currentMode}
        onSelectMode={onSelectMode}
        isCollapsed={isNavCollapsed}
        onToggleCollapse={() => setIsNavCollapsed((prev) => !prev)}
      />

      {/* Main Workspace Area */}
      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-200 ${
          isNavCollapsed ? "pl-16" : "pl-60"
        } ${isInspectorOpen ? "pr-80 lg:pr-96" : "pr-0"}`}
      >
        {/* Top Context Bar */}
        <ContextBar
          breadcrumb={breadcrumb}
          activeDataset={activeDataset}
          activeSeed={activeSeed}
          isSynthetic={isSynthetic}
          isInspectorOpen={isInspectorOpen}
          onToggleInspector={() => setIsInspectorOpen((prev) => !prev)}
          actions={headerActions}
        />

        {/* Content Body */}
        <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
          {children}
        </main>
      </div>

      {/* Right Contextual Inspector */}
      <ContextInspector
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        title={inspectorTitle}
        subtitle={inspectorSubtitle}
        metadata={inspectorMeta}
        provenance={inspectorProvenance}
      >
        {inspectorContent}
      </ContextInspector>
    </div>
  );
};
