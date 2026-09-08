import React from "react";
import {
  Compass,
  BarChart3,
  Layers,
  Sparkles,
  Target,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
} from "lucide-react";
import { AppMode } from "../ResearchPlatformNavigation";

export interface NavSection {
  id: string;
  title: string;
  items: {
    id: AppMode | "overview";
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    description: string;
  }[];
}

export const NAV_SECTIONS: NavSection[] = [
  {
    id: "core",
    title: "Platform",
    items: [
      {
        id: "overview",
        label: "Overview",
        icon: Compass,
        description: "Research hub & questions",
      },
      {
        id: "benchmark",
        label: "Benchmark",
        icon: BarChart3,
        description: "Cross-paradigm synthesis",
      },
    ],
  },
  {
    id: "representation",
    title: "Representation",
    items: [
      {
        id: "observatory",
        label: "Geometry",
        icon: Layers,
        description: "Manifolds, PCA, CKA & rank",
      },
      {
        id: "robustness",
        label: "Robustness",
        icon: ShieldCheck,
        description: "Corruptions & latent drift",
      },
      {
        id: "explainability",
        label: "Attribution",
        icon: Sparkles,
        description: "Grad-CAM & saliency heatmaps",
      },
    ],
  },
  {
    id: "learning",
    title: "Learning",
    items: [
      {
        id: "transfer",
        label: "Transfer",
        icon: Target,
        description: "Linear probes & fine-tuning",
      },
      {
        id: "ssl",
        label: "Self-Supervised",
        icon: FlaskConical,
        description: "SimCLR & collapse checks",
      },
      {
        id: "reconstruction",
        label: "Reconstruction",
        icon: Layers,
        description: "Masked autoencoding & DAE",
      },
    ],
  },
  {
    id: "downstream",
    title: "Downstream",
    items: [
      {
        id: "spatial",
        label: "Spatial",
        icon: Target,
        description: "Detection & segmentation",
      },
      {
        id: "temporal",
        label: "Temporal",
        icon: Layers,
        description: "Video sequence dynamics",
      },
      {
        id: "multimodal",
        label: "Multimodal",
        icon: Sparkles,
        description: "Vision-language alignment",
      },
    ],
  },
  {
    id: "reliability",
    title: "Reliability",
    items: [
      {
        id: "uncertainty",
        label: "Uncertainty & OOD",
        icon: ShieldCheck,
        description: "Calibration & failure scoring",
      },
    ],
  },
];

export interface NavigationRailProps {
  currentMode: AppMode | "overview";
  onSelectMode: (mode: AppMode | "overview") => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export const NavigationRail: React.FC<NavigationRailProps> = ({
  currentMode,
  onSelectMode,
  isCollapsed,
  onToggleCollapse,
}) => {
  return (
    <aside
      className={`fixed top-0 left-0 bottom-0 z-40 flex flex-col bg-slate-950 border-r border-slate-800/80 transition-all duration-200 select-none ${
        isCollapsed ? "w-16" : "w-60"
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between h-14 px-3.5 border-b border-slate-800/80">
        <div
          onClick={() => onSelectMode("overview")}
          className="flex items-center gap-2.5 cursor-pointer group overflow-hidden"
        >
          <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 font-bold shrink-0 transition-all group-hover:bg-sky-500/20 group-hover:border-sky-500/50">
            P
          </div>
          {!isCollapsed && (
            <div className="flex flex-col truncate">
              <span className="text-xs font-bold tracking-wider text-slate-100 uppercase">
                PRISM
              </span>
              <span className="text-[10px] text-slate-500 font-mono leading-none">
                Research OS
              </span>
            </div>
          )}
        </div>

        {!isCollapsed && (
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-md text-slate-500 hover:text-slate-200 hover:bg-slate-900 transition-colors"
            title="Collapse Sidebar"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Nav List */}
      <div className="flex-1 overflow-y-auto py-3 px-2 space-y-4 scrollbar-thin scrollbar-thumb-slate-800">
        {NAV_SECTIONS.map((section) => (
          <div key={section.id} className="space-y-1">
            {!isCollapsed && (
              <div className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500 font-mono">
                {section.title}
              </div>
            )}
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const isActive = currentMode === item.id;
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => onSelectMode(item.id)}
                    title={isCollapsed ? `${item.label} — ${item.description}` : undefined}
                    className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs font-medium transition-all group ${
                      isActive
                        ? "bg-sky-500/10 text-sky-400 border border-sky-500/30 font-semibold"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/80 border border-transparent"
                    } ${isCollapsed ? "justify-center px-0" : ""}`}
                  >
                    <Icon
                      className={`w-4 h-4 shrink-0 transition-colors ${
                        isActive ? "text-sky-400" : "text-slate-500 group-hover:text-slate-300"
                      }`}
                    />
                    {!isCollapsed && (
                      <div className="flex flex-col text-left truncate">
                        <span className="truncate">{item.label}</span>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-2 border-t border-slate-800/80 flex items-center justify-between text-slate-500 text-[10px]">
        {isCollapsed ? (
          <button
            onClick={onToggleCollapse}
            className="w-full py-1.5 flex justify-center text-slate-500 hover:text-slate-200 hover:bg-slate-900 rounded-md"
            title="Expand Sidebar"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        ) : (
          <>
            <span className="font-mono px-2">v1.1.0</span>
            <span className="text-emerald-400 font-mono flex items-center gap-1 px-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Ready
            </span>
          </>
        )}
      </div>
    </aside>
  );
};
