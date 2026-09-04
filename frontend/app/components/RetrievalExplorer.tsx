"use client";

import React from "react";
import { CrossModalRetrievalSummaryPayload, MultimodalSamplePayload } from "../types";

interface RetrievalExplorerProps {
  direction: "image_to_text" | "text_to_image";
  summary: CrossModalRetrievalSummaryPayload;
  selectedSample: MultimodalSamplePayload;
  allSamples: MultimodalSamplePayload[];
  onSelectSampleId: (id: string) => void;
}

export const RetrievalExplorer: React.FC<RetrievalExplorerProps> = ({
  direction,
  summary,
  selectedSample,
  allSamples,
  onSelectSampleId,
}) => {
  const isImageToText = direction === "image_to_text";
  const r1 = isImageToText ? summary.image_to_text_r1 : summary.text_to_image_r1;
  const r3 = isImageToText ? summary.image_to_text_r3 : summary.text_to_image_r3;
  const r5 = isImageToText ? summary.image_to_text_r5 : summary.text_to_image_r5;
  const mrr = isImageToText ? summary.image_to_text_mrr : summary.text_to_image_mrr;

  const candidates = isImageToText
    ? selectedSample.top_text_candidates
    : selectedSample.top_image_candidates;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-4">
      {/* Title and Direction Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 font-bold text-sm">🎯 Cross-Modal Retrieval Explorer</span>
          <span className="text-xs px-2 py-0.5 rounded font-mono bg-cyan-950 text-cyan-300 border border-cyan-500/30">
            {isImageToText ? "Image Query → Text Candidates" : "Text Query → Image Candidates"}
          </span>
        </div>
      </div>

      {/* Aggregate Retrieval Summary Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 block font-mono">Recall@1</span>
          <span className="text-base font-bold font-mono text-emerald-400">
            {(r1 * 100).toFixed(1)}%
          </span>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 block font-mono">Recall@3</span>
          <span className="text-base font-bold font-mono text-cyan-400">
            {(r3 * 100).toFixed(1)}%
          </span>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 block font-mono">Recall@5</span>
          <span className="text-base font-bold font-mono text-indigo-400">
            {(r5 * 100).toFixed(1)}%
          </span>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-center">
          <span className="text-[10px] text-slate-400 block font-mono">MRR</span>
          <span className="text-base font-bold font-mono text-amber-400">
            {mrr.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Query & Ranked Candidates List */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Ranked Candidates for Query ({selectedSample.sample_id})
        </span>

        <div className="flex flex-col gap-2 max-h-56 overflow-y-auto">
          {candidates.map((cand, idx) => {
            const candSample = allSamples.find((s) => s.sample_id === cand.sample_id);
            const isMatched = cand.sample_id === selectedSample.sample_id;
            const rank = idx + 1;

            return (
              <div
                key={cand.sample_id}
                onClick={() => onSelectSampleId(cand.sample_id)}
                className={`flex items-center justify-between p-2.5 rounded-lg border cursor-pointer transition-all ${
                  isMatched
                    ? "bg-emerald-950/40 border-emerald-500/50 hover:bg-emerald-950/60"
                    : "bg-slate-950/40 border-slate-800/80 hover:bg-slate-800/50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`w-6 h-6 rounded-full flex items-center justify-center font-mono text-xs font-bold ${
                      rank === 1
                        ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                        : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {rank}
                  </span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-semibold text-slate-200">
                        {cand.sample_id}
                      </span>
                      {isMatched && (
                        <span className="bg-emerald-500/20 text-emerald-400 text-[9px] font-bold px-1.5 py-0.2 rounded border border-emerald-500/30">
                          TRUE PAIR
                        </span>
                      )}
                    </div>
                    {candSample && (
                      <p className="text-[11px] text-slate-400 italic truncate max-w-xs">
                        &quot;{candSample.text}&quot;
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="w-20 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-cyan-500 h-full rounded-full"
                      style={{
                        width: `${Math.max(0, Math.min(100, cand.similarity * 100))}%`,
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs text-slate-300 w-12 text-right">
                    {cand.similarity.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
