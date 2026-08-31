"use client";

import React, { useEffect, useRef, useState } from "react";
import { AttributionResult } from "../types";

interface AttributionHeatmapCardProps {
  title: string;
  subtitle?: string;
  result?: AttributionResult | null;
  imageTensor: number[][][]; // [C, H, W]
  isUnsupported?: boolean;
  unsupportedReason?: string;
  defaultMode?: "overlay" | "heatmap" | "original";
}

type ColormapName = "turbo" | "plasma" | "viridis" | "signed_diverging";

// Perceptual colormap interpolations
function getColormapColor(val: number, cmap: ColormapName): [number, number, number] {
  const v = Math.max(0, Math.min(1, val));
  if (cmap === "plasma") {
    // Plasma: dark purple -> magenta -> orange -> bright yellow
    const r = Math.min(255, Math.max(0, Math.round(255 * (0.05 + 0.95 * Math.pow(v, 0.7)))));
    const g = Math.min(255, Math.max(0, Math.round(255 * (0.0 + 0.9 * Math.sin(v * Math.PI)))));
    const b = Math.min(255, Math.max(0, Math.round(255 * (0.5 * (1 - v) + 0.5 * Math.cos(v * Math.PI)))));
    return [r, g, b];
  } else if (cmap === "viridis") {
    // Viridis: purple -> teal -> green -> yellow
    const r = Math.min(255, Math.max(0, Math.round(255 * (0.2 + 0.8 * Math.pow(v, 1.5)))));
    const g = Math.min(255, Math.max(0, Math.round(255 * (0.1 + 0.85 * v))));
    const b = Math.min(255, Math.max(0, Math.round(255 * (0.5 * (1 - v) + 0.1))));
    return [r, g, b];
  } else if (cmap === "signed_diverging") {
    // Blue for negative (-1), dark slate for zero (0), Red/Orange for positive (+1)
    if (v >= 0.5) {
      const pos = (v - 0.5) * 2;
      return [Math.round(240 * pos + 30 * (1 - pos)), Math.round(80 * pos + 30 * (1 - pos)), Math.round(30)];
    } else {
      const neg = (0.5 - v) * 2;
      return [Math.round(30), Math.round(120 * neg + 30 * (1 - neg)), Math.round(240 * neg + 30 * (1 - neg))];
    }
  } else {
    // Turbo / Jet: blue -> cyan -> yellow -> red
    const r = Math.min(255, Math.max(0, Math.round(255 * Math.sin(v * Math.PI * 0.7))));
    const g = Math.min(255, Math.max(0, Math.round(255 * Math.sin(v * Math.PI))));
    const b = Math.min(255, Math.max(0, Math.round(255 * Math.cos(v * Math.PI * 0.8))));
    return [r, g, b];
  }
}

export const AttributionHeatmapCard: React.FC<AttributionHeatmapCardProps> = ({
  title,
  subtitle,
  result,
  imageTensor,
  isUnsupported = false,
  unsupportedReason = "Method not applicable to this architecture family",
  defaultMode = "overlay",
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [viewMode, setViewMode] = useState<"overlay" | "heatmap" | "original">(defaultMode);
  const [colormap, setColormap] = useState<ColormapName>("turbo");
  const [opacity, setOpacity] = useState<number>(0.65);
  const [showCenterOfMass, setShowCenterOfMass] = useState<boolean>(true);

  const c = imageTensor.length;
  const h = imageTensor[0]?.length || 8;
  const w = imageTensor[0]?.[0]?.length || 8;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || isUnsupported) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = w;
    canvas.height = h;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const normMap = result?.normalized_attribution_map || [];

    for (let r = 0; r < h; r++) {
      for (let col = 0; col < w; col++) {
        const idx = (r * w + col) * 4;

        // Base image RGB [0..1] -> [0..255]
        const origR = Math.round((imageTensor[0]?.[r]?.[col] || 0) * 255);
        const origG = Math.round((imageTensor[Math.min(1, c - 1)]?.[r]?.[col] || 0) * 255);
        const origB = Math.round((imageTensor[Math.min(2, c - 1)]?.[r]?.[col] || 0) * 255);

        // Heatmap color
        const heatVal = normMap[r]?.[col] ?? 0;
        const [heatR, heatG, heatB] = getColormapColor(heatVal, colormap);

        if (viewMode === "original") {
          data[idx] = origR;
          data[idx + 1] = origG;
          data[idx + 2] = origB;
          data[idx + 3] = 255;
        } else if (viewMode === "heatmap") {
          data[idx] = heatR;
          data[idx + 1] = heatG;
          data[idx + 2] = heatB;
          data[idx + 3] = 255;
        } else {
          // Overlay mode
          const alpha = opacity;
          data[idx] = Math.round(origR * (1 - alpha) + heatR * alpha);
          data[idx + 1] = Math.round(origG * (1 - alpha) + heatG * alpha);
          data[idx + 2] = Math.round(origB * (1 - alpha) + heatB * alpha);
          data[idx + 3] = 255;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [imageTensor, result, viewMode, colormap, opacity, isUnsupported, h, w, c]);

  if (isUnsupported) {
    return (
      <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between h-full min-h-[320px]">
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-slate-400 font-mono">{title}</h4>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] font-mono text-slate-500">
              N/A
            </span>
          </div>
          {subtitle && <p className="text-[11px] text-slate-500 mb-4">{subtitle}</p>}
        </div>

        <div className="my-auto flex flex-col items-center justify-center p-6 border border-dashed border-slate-800 rounded-lg text-center bg-slate-950/40">
          <span className="text-2xl mb-2">🚫</span>
          <span className="text-xs font-bold text-slate-400">Method Unsupported</span>
          <p className="text-[11px] text-slate-500 mt-1 max-w-[200px]">{unsupportedReason}</p>
        </div>

        <div className="text-[10px] text-slate-600 font-mono mt-2">
          Architecture constraint
        </div>
      </div>
    );
  }

  const stats = result?.statistics;

  return (
    <div className="bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all rounded-xl p-4 flex flex-col justify-between shadow-lg">
      {/* Top Title Bar */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <h4 className="text-xs font-black text-slate-200 font-mono tracking-wide">{title}</h4>
          <span className="px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-[10px] font-mono text-cyan-400 font-semibold">
            {result?.method.toUpperCase()}
          </span>
        </div>
        {subtitle && <p className="text-[11px] text-slate-400 mb-3">{subtitle}</p>}

        {/* View Mode Switcher */}
        <div className="flex items-center justify-between gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 mb-3 text-[11px]">
          <button
            onClick={() => setViewMode("overlay")}
            className={`flex-1 py-1 rounded font-medium transition-all ${
              viewMode === "overlay"
                ? "bg-cyan-600 text-white font-bold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Overlay
          </button>
          <button
            onClick={() => setViewMode("heatmap")}
            className={`flex-1 py-1 rounded font-medium transition-all ${
              viewMode === "heatmap"
                ? "bg-cyan-600 text-white font-bold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Heatmap
          </button>
          <button
            onClick={() => setViewMode("original")}
            className={`flex-1 py-1 rounded font-medium transition-all ${
              viewMode === "original"
                ? "bg-cyan-600 text-white font-bold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Source
          </button>
        </div>
      </div>

      {/* Main Canvas View with Crosshair Overlay */}
      <div className="relative aspect-square w-full max-w-[240px] mx-auto my-2 rounded-lg overflow-hidden border border-slate-800 bg-slate-950 shadow-inner flex items-center justify-center">
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain"
          style={{ imageRendering: "pixelated" }}
        />

        {/* Center of Mass Indicator */}
        {showCenterOfMass && stats && viewMode !== "original" && (
          <div
            className="absolute pointer-events-none transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
            style={{
              top: `${((stats.center_of_mass_row + 0.5) / h) * 100}%`,
              left: `${((stats.center_of_mass_col + 0.5) / w) * 100}%`,
            }}
          >
            <div className="w-3.5 h-3.5 rounded-full border-2 border-amber-400 bg-amber-500/30 animate-pulse shadow-md shadow-amber-500/50" />
            <div className="absolute w-5 h-[1.5px] bg-amber-400/80" />
            <div className="absolute h-5 w-[1.5px] bg-amber-400/80" />
          </div>
        )}
      </div>

      {/* Controls: Opacity & Colormap */}
      <div className="space-y-2 mt-2 pt-2 border-t border-slate-800/80">
        {viewMode === "overlay" && (
          <div className="flex items-center justify-between text-[11px] gap-2 text-slate-400">
            <span>Opacity:</span>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="flex-1 accent-cyan-500 h-1 bg-slate-800 rounded-lg cursor-pointer"
            />
            <span className="font-mono text-[10px] w-7 text-right">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        )}

        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span>Colormap:</span>
          <div className="flex items-center gap-1">
            {(["turbo", "plasma", "viridis", "signed_diverging"] as ColormapName[]).map((cm) => (
              <button
                key={cm}
                onClick={() => setColormap(cm)}
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono capitalize transition-all ${
                  colormap === cm
                    ? "bg-slate-700 text-cyan-300 font-bold border border-cyan-500/40"
                    : "bg-slate-950 text-slate-500 hover:text-slate-300"
                }`}
              >
                {cm === "signed_diverging" ? "Diverge" : cm}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span>Center of Mass:</span>
          <button
            onClick={() => setShowCenterOfMass(!showCenterOfMass)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono transition-all ${
              showCenterOfMass
                ? "bg-amber-950/80 border border-amber-600/60 text-amber-400 font-semibold"
                : "bg-slate-950 text-slate-500 hover:text-slate-300 border border-slate-800"
            }`}
          >
            {showCenterOfMass ? "Visible" : "Hidden"}
          </button>
        </div>
      </div>

      {/* Metrics Strip */}
      {stats && (
        <div className="grid grid-cols-2 gap-1.5 mt-3 pt-2 border-t border-slate-800/80 text-[10px] font-mono">
          <div className="bg-slate-950/80 p-1.5 rounded border border-slate-800/60 flex flex-col">
            <span className="text-slate-500 text-[9px]">CONCENTRATION</span>
            <span className="text-cyan-400 font-bold">{stats.concentration_score.toFixed(3)}</span>
          </div>
          <div className="bg-slate-950/80 p-1.5 rounded border border-slate-800/60 flex flex-col">
            <span className="text-slate-500 text-[9px]">TOP 10% MASS</span>
            <span className="text-emerald-400 font-bold">{(stats.top_10_percent_mass_fraction * 100).toFixed(1)}%</span>
          </div>
          <div className="bg-slate-950/80 p-1.5 rounded border border-slate-800/60 flex flex-col">
            <span className="text-slate-500 text-[9px]">ENTROPY</span>
            <span className="text-purple-400 font-bold">{stats.spatial_entropy.toFixed(3)}</span>
          </div>
          <div className="bg-slate-950/80 p-1.5 rounded border border-slate-800/60 flex flex-col">
            <span className="text-slate-500 text-[9px]">CENTER (R, C)</span>
            <span className="text-amber-400 font-bold">
              ({stats.center_of_mass_row.toFixed(1)}, {stats.center_of_mass_col.toFixed(1)})
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
