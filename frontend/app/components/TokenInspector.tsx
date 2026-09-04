"use client";

import React from "react";
import { TokenizedTextPayload } from "../types";

interface TokenInspectorProps {
  tokenized: TokenizedTextPayload;
}

export const TokenInspector: React.FC<TokenInspectorProps> = ({ tokenized }) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <span className="text-indigo-400 font-bold text-sm">🔤 Token Sequence Diagnostics</span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          Valid Length: <strong className="text-cyan-400">{tokenized.sequence_length}</strong> /{" "}
          {tokenized.token_ids.length}
        </span>
      </div>

      {/* Token Sequence Badges */}
      <div className="flex flex-col gap-2">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          Token Sequence & Masking
        </span>
        <div className="flex flex-wrap gap-1.5 p-2.5 bg-slate-950/70 rounded-lg border border-slate-800/80 max-h-36 overflow-y-auto">
          {tokenized.token_strings.map((tok, idx) => {
            const isSpecial = tok.startsWith("<") && tok.endsWith(">");
            const isPad = tok === "<PAD>";
            const tid = tokenized.token_ids[idx];
            const mask = tokenized.attention_mask[idx];

            return (
              <div
                key={idx}
                className={`flex flex-col items-center px-2 py-1 rounded border text-xs font-mono transition-all ${
                  isPad
                    ? "bg-slate-900/40 border-slate-800/60 text-slate-600 opacity-60"
                    : isSpecial
                    ? "bg-amber-950/60 border-amber-500/40 text-amber-300"
                    : "bg-cyan-950/60 border-cyan-500/40 text-cyan-200"
                }`}
              >
                <span className="font-semibold">{tok}</span>
                <div className="flex items-center gap-1 text-[9px] mt-0.5 opacity-80">
                  <span>id:{tid}</span>
                  <span>•</span>
                  <span>m:{mask}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
