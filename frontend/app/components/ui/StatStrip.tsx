import React from "react";

export interface StatItem {
  label: string;
  value: string | number;
  subValue?: string;
  change?: {
    value: string | number;
    trend: "up" | "down" | "neutral";
  };
  tooltip?: string;
  isMono?: boolean;
}

export interface StatStripProps {
  items: StatItem[];
  className?: string;
}

export const StatStrip: React.FC<StatStripProps> = ({ items, className = "" }) => {
  return (
    <div
      className={`grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-flow-col xl:auto-cols-fr gap-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 ${className}`}
    >
      {items.map((item, idx) => (
        <div key={idx} className="flex flex-col gap-0.5 px-2 py-1">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider select-none truncate">
            {item.label}
          </span>
          <div className="flex items-baseline gap-2">
            <span
              className={`text-base font-semibold text-slate-100 ${
                item.isMono !== false ? "font-mono" : ""
              }`}
            >
              {item.value}
            </span>
            {item.subValue && (
              <span className="text-xs text-slate-500 font-mono">{item.subValue}</span>
            )}
            {item.change && (
              <span
                className={`text-[10px] font-medium font-mono ${
                  item.change.trend === "up"
                    ? "text-emerald-400"
                    : item.change.trend === "down"
                    ? "text-rose-400"
                    : "text-slate-400"
                }`}
              >
                {item.change.trend === "up" ? "↑" : item.change.trend === "down" ? "↓" : ""}
                {item.change.value}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
