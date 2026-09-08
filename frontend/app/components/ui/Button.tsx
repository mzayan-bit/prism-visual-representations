import React from "react";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "outline";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: React.ReactNode;
  iconRight?: React.ReactNode;
  isActive?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "secondary",
  size = "md",
  icon,
  iconRight,
  isActive = false,
  className = "",
  disabled,
  ...props
}) => {
  const baseStyles =
    "inline-flex items-center justify-center font-medium transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:opacity-40 disabled:cursor-not-allowed select-none rounded-md";

  const sizeStyles = {
    sm: "text-xs px-2.5 py-1 gap-1.5 h-7",
    md: "text-xs px-3 py-1.5 gap-2 h-8.5",
    lg: "text-sm px-4 py-2 gap-2.5 h-10",
  }[size];

  const variantStyles = {
    primary:
      "bg-sky-500 text-slate-950 hover:bg-sky-400 active:bg-sky-600 font-semibold shadow-sm",
    secondary: isActive
      ? "bg-slate-800 text-sky-400 border border-sky-500/40"
      : "bg-slate-900/80 text-slate-200 hover:bg-slate-800 hover:text-white border border-slate-800",
    outline: isActive
      ? "bg-sky-500/10 text-sky-400 border border-sky-500/50"
      : "bg-transparent text-slate-300 hover:bg-slate-900 border border-slate-800 hover:border-slate-700",
    ghost: isActive
      ? "bg-slate-800/80 text-sky-400"
      : "bg-transparent text-slate-400 hover:text-slate-100 hover:bg-slate-800/50",
    danger:
      "bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20 active:bg-rose-500/30",
  }[variant];

  return (
    <button
      className={`${baseStyles} ${sizeStyles} ${variantStyles} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {children && <span>{children}</span>}
      {iconRight && <span className="shrink-0">{iconRight}</span>}
    </button>
  );
};
