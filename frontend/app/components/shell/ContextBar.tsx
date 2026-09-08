import React from "react";
import { SlidersHorizontal, Layers, Database } from "lucide-react";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";

export interface ContextBarProps {
  breadcrumb: string[];
  activeDataset?: string;
  activeSeed?: number;
  isSynthetic?: boolean;
  isInspectorOpen: boolean;
  onToggleInspector: () => void;
  actions?: React.ReactNode;
}

export const ContextBar: React.FC<ContextBarProps> = ({
  breadcrumb,
  activeDataset = "cifar10",
  activeSeed = 42,
  isSynthetic = true,
  isInspectorOpen,
  onToggleInspector,
  actions,
}) => {
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-14 px-6 bg-slate-950/90 backdrop-blur-md border-b border-slate-800/80">
      {/* Breadcrumb Navigation */}
      <div className="flex items-center gap-2 text-xs font-medium">
        {breadcrumb.map((crumb, idx) => (
          <React.Fragment key={idx}>
            {idx > 0 && <span className="text-slate-600">/</span>}
            <span
              className={
                idx === breadcrumb.length - 1
                  ? "text-slate-100 font-semibold"
                  : "text-slate-400 hover:text-slate-200 transition-colors cursor-default"
              }
            >
              {crumb}
            </span>
          </React.Fragment>
        ))}
      </div>

      {/* Right Meta Controls */}
      <div className="flex items-center gap-3">
        <div className="hidden sm:flex items-center gap-2">
          <Badge variant="synthetic" size="sm">
            <Database className="w-3 h-3 text-slate-400 inline mr-1" />
            {activeDataset}
          </Badge>

          <Badge variant="synthetic" size="sm">
            <Layers className="w-3 h-3 text-slate-400 inline mr-1" />
            seed={activeSeed}
          </Badge>

          {isSynthetic && (
            <Badge variant="synthetic" size="sm">
              Controlled Synthetic
            </Badge>
          )}
        </div>

        {actions}

        <Button
          variant="secondary"
          size="sm"
          isActive={isInspectorOpen}
          onClick={onToggleInspector}
          icon={<SlidersHorizontal className="w-3.5 h-3.5" />}
          title="Toggle Contextual Inspector"
        >
          <span className="hidden md:inline">Inspector</span>
        </Button>
      </div>
    </header>
  );
};
