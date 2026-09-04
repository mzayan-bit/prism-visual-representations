"use client";

import React, { useEffect, useRef } from "react";
import { MultimodalSamplePayload } from "../types";

interface PairedSampleViewerProps {
  sample: MultimodalSamplePayload;
}

export const PairedSampleViewer: React.FC<PairedSampleViewerProps> = ({ sample }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sample.image) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const c = sample.image.length;
    const h = sample.image[0].length;
    const w = sample.image[0][0].length;

    canvas.width = w;
    canvas.height = h;

    const imgData = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const r = c > 0 ? Math.min(255, Math.max(0, Math.round(sample.image[0][y][x] * 255))) : 0;
        const g = c > 1 ? Math.min(255, Math.max(0, Math.round(sample.image[1][y][x] * 255))) : r;
        const b = c > 2 ? Math.min(255, Math.max(0, Math.round(sample.image[2][y][x] * 255))) : r;

        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [sample]);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-bold text-sm">🖼️ Paired Sample Instance</span>
          <span className="bg-slate-800 text-slate-300 font-mono text-xs px-2 py-0.5 rounded border border-slate-700">
            {sample.sample_id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-400 font-mono">Class:</span>
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-500/30">
            {sample.class_name || "N/A"}
          </span>
        </div>
      </div>

      {/* Main Visual & Text Pairing Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
        {/* Rendered Image Canvas */}
        <div className="md:col-span-4 flex flex-col items-center justify-center p-3 bg-slate-950/80 rounded-lg border border-slate-800/80">
          <div className="relative w-36 h-36 bg-black rounded-lg overflow-hidden border border-slate-700 flex items-center justify-center shadow-inner">
            <canvas
              ref={canvasRef}
              className="w-full h-full"
              style={{ imageRendering: "pixelated" }}
            />
          </div>
          <span className="text-[10px] text-slate-500 mt-2 font-mono">Synthetic RGB Tensor (3×16×16)</span>
        </div>

        {/* Descriptive Text & Tokenized Info */}
        <div className="md:col-span-8 flex flex-col gap-3">
          {/* Canonical Paired Text Box */}
          <div className="bg-slate-950/60 p-3 rounded-lg border border-cyan-500/20">
            <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block mb-1">
              Paired Text Description
            </span>
            <p className="text-sm font-medium text-slate-100 italic bg-slate-900/80 px-3 py-2 rounded border border-slate-800">
              &quot;{sample.text}&quot;
            </p>
          </div>

          {/* Cross-Modal Metric Badges */}
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-slate-800/60 p-2 rounded-lg border border-slate-700/60">
              <span className="text-[10px] text-slate-400 block font-mono">Cosine Sim</span>
              <span className="text-sm font-bold font-mono text-emerald-400">
                {sample.paired_cosine.toFixed(3)}
              </span>
            </div>
            <div className="bg-slate-800/60 p-2 rounded-lg border border-slate-700/60">
              <span className="text-[10px] text-slate-400 block font-mono">Shared Distance</span>
              <span className="text-sm font-bold font-mono text-cyan-400">
                {sample.paired_distance.toFixed(3)}
              </span>
            </div>
            <div className="bg-slate-800/60 p-2 rounded-lg border border-slate-700/60">
              <span className="text-[10px] text-slate-400 block font-mono">Retrieval Rank</span>
              <span
                className={`text-sm font-bold font-mono ${
                  sample.i2t_rank === 1 ? "text-emerald-400" : "text-amber-400"
                }`}
              >
                #{sample.i2t_rank}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
