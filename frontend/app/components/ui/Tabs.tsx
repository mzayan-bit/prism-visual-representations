import React from "react";

export interface TabItem<T extends string = string> {
  id: T;
  label: string;
  badge?: string | number;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export interface TabsProps<T extends string = string> {
  tabs: TabItem<T>[];
  activeTab: T;
  onChange: (tabId: T) => void;
  variant?: "pill" | "line" | "segmented";
  size?: "sm" | "md";
  className?: string;
}

export function Tabs<T extends string = string>({
  tabs,
  activeTab,
  onChange,
  variant = "pill",
  className = "",
}: TabsProps<T>) {
  if (variant === "segmented") {
    return (
      <div
        className={`inline-flex items-center bg-slate-900 p-0.5 rounded-lg border border-slate-800 ${className}`}
      >
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              disabled={tab.disabled}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all select-none disabled:opacity-40 disabled:cursor-not-allowed ${
                isActive
                  ? "bg-slate-800 text-sky-400 shadow-sm font-semibold"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
              }`}
            >
              {tab.icon && <span className="w-3.5 h-3.5 shrink-0">{tab.icon}</span>}
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    isActive ? "bg-sky-500/20 text-sky-300" : "bg-slate-800 text-slate-500"
                  }`}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    );
  }

  if (variant === "line") {
    return (
      <div className={`flex items-center gap-6 border-b border-slate-800 ${className}`}>
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              disabled={tab.disabled}
              className={`flex items-center gap-2 py-2.5 text-xs font-medium border-b-2 -mb-px transition-colors select-none disabled:opacity-40 disabled:cursor-not-allowed ${
                isActive
                  ? "border-sky-500 text-sky-400 font-semibold"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              {tab.icon && <span className="w-3.5 h-3.5 shrink-0">{tab.icon}</span>}
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                    isActive ? "bg-sky-500/10 text-sky-400" : "bg-slate-800 text-slate-400"
                  }`}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    );
  }

  // Pill variant
  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            disabled={tab.disabled}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all select-none disabled:opacity-40 disabled:cursor-not-allowed ${
              isActive
                ? "bg-sky-500/10 text-sky-400 border border-sky-500/40 font-semibold"
                : "bg-slate-900/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-slate-800/80"
            }`}
          >
            {tab.icon && <span className="w-3.5 h-3.5 shrink-0">{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.badge !== undefined && (
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                  isActive ? "bg-sky-500/20 text-sky-300" : "bg-slate-800 text-slate-500"
                }`}
              >
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
