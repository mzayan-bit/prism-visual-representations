import React from "react";

export type BadgeVariant =
  | "neutral"
  | "accent"
  | "success"
  | "warning"
  | "danger"
  | "synthetic"
  | "outline";

export interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  size?: "sm" | "md";
  dot?: boolean;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "neutral",
  size = "md",
  dot = false,
  className = "",
}) => {
  const sizeStyles = {
    sm: "text-[10px] px-1.5 py-0.5 tracking-wider gap-1",
    md: "text-xs px-2 py-0.5 tracking-normal gap-1.5",
  }[size];

  const variantStyles = {
    neutral: "bg-slate-800/80 text-slate-300 border border-slate-700/50",
    accent: "bg-sky-500/10 text-sky-400 border border-sky-500/30",
    success: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
    warning: "bg-amber-500/10 text-amber-400 border border-amber-500/30",
    danger: "bg-rose-500/10 text-rose-400 border border-rose-500/30",
    synthetic: "bg-slate-800/50 text-slate-400 border border-slate-700/60 font-mono",
    outline: "bg-transparent text-slate-400 border border-slate-800",
  }[variant];

  const dotStyles = {
    neutral: "bg-slate-400",
    accent: "bg-sky-400",
    success: "bg-emerald-400",
    warning: "bg-amber-400",
    danger: "bg-rose-400",
    synthetic: "bg-slate-400",
    outline: "bg-slate-500",
  }[variant];

  return (
    <span
      className={`inline-flex items-center font-medium rounded shrink-0 select-none ${sizeStyles} ${variantStyles} ${className}`}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${dotStyles}`} />}
      {children}
    </span>
  );
};
