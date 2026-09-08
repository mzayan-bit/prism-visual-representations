import React from "react";

export interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  hoverable?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = "",
  onClick,
  hoverable = false,
}) => {
  return (
    <div
      onClick={onClick}
      className={`bg-slate-900/70 border border-slate-800/90 rounded-lg overflow-hidden transition-all ${
        hoverable ? "hover:border-slate-700 hover:bg-slate-900/90 cursor-pointer" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<{
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}> = ({ title, subtitle, badge, action, className = "" }) => (
  <div
    className={`flex items-center justify-between gap-3 px-4 py-3 border-b border-slate-800/80 bg-slate-900/40 ${className}`}
  >
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-2">
        <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
        {badge}
      </div>
      {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
    </div>
    {action && <div className="shrink-0">{action}</div>}
  </div>
);

export const CardContent: React.FC<{
  children: React.ReactNode;
  className?: string;
}> = ({ children, className = "" }) => (
  <div className={`p-4 ${className}`}>{children}</div>
);
