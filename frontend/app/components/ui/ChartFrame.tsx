import React from "react";
import { Badge } from "./Badge";

export interface ChartFrameProps {
  title: string;
  subtitle?: string;
  badge?: string;
  headerRight?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  minHeight?: string;
}

export const ChartFrame: React.FC<ChartFrameProps> = ({
  title,
  subtitle,
  badge,
  headerRight,
  footer,
  children,
  className = "",
  minHeight = "min-h-[320px]",
}) => {
  return (
    <div
      className={`flex flex-col bg-slate-900/70 border border-slate-800 rounded-lg overflow-hidden transition-colors ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-slate-800/80 bg-slate-900/40">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-100 tracking-tight">{title}</h3>
            {badge && (
              <Badge variant="neutral" size="sm">
                {badge}
              </Badge>
            )}
          </div>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
        {headerRight && <div className="flex items-center gap-2">{headerRight}</div>}
      </div>

      <div className={`p-4 flex-1 flex flex-col ${minHeight}`}>{children}</div>

      {footer && (
        <div className="px-4 py-2.5 border-t border-slate-800/60 bg-slate-950/40 text-xs text-slate-400">
          {footer}
        </div>
      )}
    </div>
  );
};
