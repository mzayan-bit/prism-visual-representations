import React from "react";
import { ChevronDown } from "lucide-react";

export interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "onChange" | "size"> {
  label?: string;
  options: SelectOption[];
  value: string | number;
  onChange: (value: string) => void;
  size?: "sm" | "md";
  className?: string;
}

export const Select: React.FC<SelectProps> = ({
  label,
  options,
  value,
  onChange,
  size = "md",
  className = "",
  disabled,
  ...props
}) => {
  const sizeStyles = {
    sm: "text-xs h-7 pl-2.5 pr-7 py-1",
    md: "text-xs h-8.5 pl-3 pr-8 py-1.5",
  }[size];

  return (
    <div className="inline-flex flex-col gap-1">
      {label && (
        <label className="text-[11px] font-medium text-slate-400 uppercase tracking-wider select-none">
          {label}
        </label>
      )}
      <div className="relative inline-block">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={`appearance-none w-full bg-slate-900 text-slate-200 border border-slate-800 rounded-md font-medium transition-colors hover:border-slate-700 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 disabled:opacity-40 disabled:cursor-not-allowed ${sizeStyles} ${className}`}
          {...props}
        >
          {options.map((opt) => (
            <option key={String(opt.value)} value={opt.value} disabled={opt.disabled} className="bg-slate-900 text-slate-200 py-1">
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
      </div>
    </div>
  );
};
