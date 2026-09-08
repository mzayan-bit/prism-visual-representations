import React from "react";
import { Inbox } from "lucide-react";
import { Button } from "./Button";

export interface EmptyStateProps {
  title?: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = "No data available",
  description,
  actionLabel,
  onAction,
  icon,
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center p-8 border border-dashed border-slate-800 rounded-lg bg-slate-900/20 ${className}`}
    >
      <div className="w-10 h-10 rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400 mb-3">
        {icon || <Inbox className="w-5 h-5" />}
      </div>
      <h4 className="text-sm font-semibold text-slate-200 mb-1">{title}</h4>
      <p className="text-xs text-slate-400 max-w-sm mb-4 leading-relaxed">{description}</p>
      {actionLabel && onAction && (
        <Button variant="secondary" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
};
