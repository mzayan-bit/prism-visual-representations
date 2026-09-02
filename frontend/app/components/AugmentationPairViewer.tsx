"use client";

import React from "react";

export function AugmentationPairViewer() {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-white tracking-tight">
            Deterministic Augmentation Pair Pipeline
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Two independently transformed views derived from the same source sample for instance contrast.
          </p>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
          Positive Pair: (View A, View B)
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Source Sample */}
        <div className="bg-slate-950 border border-slate-800/80 rounded-lg p-3 flex flex-col items-center">
          <span className="text-[11px] font-medium text-slate-400 mb-2">Original Source Image</span>
          <div className="w-24 h-24 rounded bg-gradient-to-br from-blue-600 via-indigo-700 to-purple-800 flex items-center justify-center shadow-inner border border-slate-700/50">
            <span className="text-xs font-mono text-white/80">3 &times; 32 &times; 32</span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2">ID: cifar_sample_42</span>
          <div className="mt-2 text-[10px] text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
            Ground Truth: Airplane (unlabeled in SSL)
          </div>
        </div>

        {/* View A */}
        <div className="bg-slate-950 border border-indigo-900/40 rounded-lg p-3 flex flex-col items-center">
          <div className="flex items-center justify-between w-full mb-2">
            <span className="text-[11px] font-semibold text-indigo-400">View A (Branch 0)</span>
            <span className="text-[9px] font-mono bg-indigo-950 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-800/40">
              seed: 0x4f8a2
            </span>
          </div>
          <div className="w-24 h-24 rounded bg-gradient-to-tr from-indigo-700 via-purple-600 to-pink-700 flex items-center justify-center shadow-inner border border-indigo-600/40">
            <span className="text-[11px] font-mono text-white">View A</span>
          </div>
          <div className="mt-2.5 w-full space-y-1">
            <div className="text-[10px] text-slate-300 flex justify-between bg-slate-900/80 px-2 py-0.5 rounded">
              <span>Horizontal Flip:</span>
              <span className="text-emerald-400 font-medium">Applied</span>
            </div>
            <div className="text-[10px] text-slate-300 flex justify-between bg-slate-900/80 px-2 py-0.5 rounded">
              <span>Random Crop:</span>
              <span className="text-indigo-400 font-mono">pad=2, (r=1, c=2)</span>
            </div>
            <div className="text-[10px] text-slate-300 flex justify-between bg-slate-900/80 px-2 py-0.5 rounded">
              <span>Color Jitter:</span>
              <span className="text-slate-400 font-mono">b=1.12, c=0.94</span>
            </div>
          </div>
        </div>

        {/* View B */}
        <div className="bg-slate-950 border border-purple-900/40 rounded-lg p-3 flex flex-col items-center">
          <div className="flex items-center justify-between w-full mb-2">
            <span className="text-[11px] font-semibold text-purple-400">View B (Branch 1)</span>
            <span className="text-[9px] font-mono bg-purple-950 text-purple-300 px-1.5 py-0.5 rounded border border-purple-800/40">
              seed: 0x9b1c7
            </span>
          </div>
          <div className="w-24 h-24 rounded bg-gradient-to-bl from-purple-800 via-pink-600 to-amber-600 flex items-center justify-center shadow-inner border border-purple-600/40">
            <span className="text-[11px] font-mono text-white">View B</span>
          </div>
          <div className="mt-2.5 w-full space-y-1">
            <div className="text-[10px] text-slate-300 flex justify-between bg-slate-900/80 px-2 py-0.5 rounded">
              <span>Horizontal Flip:</span>
              <span className="text-slate-500">Not Applied</span>
            </div>
            <div className="text-[10px] text-slate-300 flex justify-between bg-slate-900/80 px-2 py-0.5 rounded">
              <span>Random Crop:</span>
              <span className="text-purple-400 font-mono">pad=2, (r=3, c=1)</span>
            </div>
            <div className="text-[10px] text-slate-300 flex justify-between bg-slate-900/80 px-2 py-0.5 rounded">
              <span>Color Jitter:</span>
              <span className="text-slate-400 font-mono">b=0.88, c=1.18</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
