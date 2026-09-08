import React from "react";
import { X, Info, GitBranch, Cpu, Hash } from "lucide-react";
import { Badge } from "../ui/Badge";

export interface InspectorMetaItem {
  label: string;
  value: string | number;
  isMono?: boolean;
}

export interface ContextInspectorProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  subtitle?: string;
  metadata?: InspectorMetaItem[];
  provenance?: {
    runId?: string;
    experimentId?: string;
    seed?: number;
    hardware?: string;
    fingerprint?: string;
  };
  children?: React.ReactNode;
}

export const ContextInspector: React.FC<ContextInspectorProps> = ({
  isOpen,
  onClose,
  title = "Context Inspector",
  subtitle = "Selected sample and execution details",
  metadata = [],
  provenance,
  children,
}) => {
  if (!isOpen) return null;

  return (
    <aside className="fixed top-14 right-0 bottom-0 z-30 w-80 lg:w-96 flex flex-col bg-slate-950/95 backdrop-blur-md border-l border-slate-800/90 shadow-2xl transition-all duration-200">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-slate-800/80 bg-slate-900/40">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-sky-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-100">
              {title}
            </h3>
          </div>
          <p className="text-[11px] text-slate-400 truncate max-w-[240px]">{subtitle}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          title="Close Inspector"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs scrollbar-thin scrollbar-thumb-slate-800">
        {/* Dynamic Context Children */}
        {children}

        {/* Key Metrics / Parameters */}
        {metadata.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider font-mono">
              Properties
            </h4>
            <div className="space-y-1 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/70">
              {metadata.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between py-1 border-b border-slate-800/40 last:border-0">
                  <span className="text-slate-400">{item.label}</span>
                  <span
                    className={`text-slate-200 font-medium ${
                      item.isMono !== false ? "font-mono" : ""
                    }`}
                  >
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Provenance & Lineage */}
        {provenance && (
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider font-mono flex items-center gap-1.5">
              <GitBranch className="w-3.5 h-3.5 text-slate-400" />
              Provenance
            </h4>
            <div className="space-y-2 bg-slate-900/40 p-3 rounded-lg border border-slate-800/70 text-[11px]">
              {provenance.experimentId && (
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase tracking-wider">
                    Experiment ID
                  </span>
                  <code className="text-slate-300 font-mono select-all">
                    {provenance.experimentId}
                  </code>
                </div>
              )}
              {provenance.runId && (
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase tracking-wider">
                    Run ID
                  </span>
                  <code className="text-slate-300 font-mono select-all">
                    {provenance.runId}
                  </code>
                </div>
              )}
              {provenance.fingerprint && (
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase tracking-wider flex items-center gap-1">
                    <Hash className="w-2.5 h-2.5" />
                    Fingerprint
                  </span>
                  <code className="text-sky-400 font-mono text-[10px] select-all break-all">
                    {provenance.fingerprint}
                  </code>
                </div>
              )}
              <div className="flex items-center justify-between pt-1 border-t border-slate-800/60">
                <span className="text-slate-500 flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  {provenance.hardware || "Apple Silicon / CPU"}
                </span>
                {provenance.seed !== undefined && (
                  <Badge variant="synthetic" size="sm">
                    seed={provenance.seed}
                  </Badge>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
