"use client";

import React, { useEffect, useRef, useState } from "react";
import { TemporalVideoSamplePayload } from "../types";

interface VideoFrameStripProps {
  sample: TemporalVideoSamplePayload | undefined;
  activeFrameIndex: number;
  onSelectFrame: (index: number) => void;
}

export const VideoFrameStrip: React.FC<VideoFrameStripProps> = ({
  sample,
  activeFrameIndex,
  onSelectFrame,
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!sample) return;
    if (isPlaying) {
      playTimerRef.current = setInterval(() => {
        onSelectFrame((activeFrameIndex + 1) % sample.frame_count);
      }, 600);
    } else if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, sample, activeFrameIndex, onSelectFrame]);

  if (!sample) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center text-slate-500">
        No video sequence selected.
      </div>
    );
  }

  const trajectory = sample.motion_trajectory;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Top Header: Video ID, Trajectory & Playback Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-800/40 flex items-center justify-center text-cyan-400 font-mono text-xs font-bold">
            {sample.frame_count}F
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold text-slate-200">
                {sample.video_id}
              </span>
              <span className="text-[10px] bg-slate-800 text-cyan-400 px-2 py-0.5 rounded font-mono font-medium border border-slate-700">
                {trajectory?.direction.replace(/_/g, " ").toUpperCase() || "CUSTOM"}
              </span>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
              <span>Velocity: {trajectory ? trajectory.velocity_magnitude.toFixed(3) : "0.000"} px/step</span>
              <span>•</span>
              <span>Shape: {String(sample.metadata?.shape_type || "geometric")}</span>
            </div>
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={() =>
              onSelectFrame(
                (activeFrameIndex - 1 + sample.frame_count) % sample.frame_count
              )
            }
            className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white text-xs transition-colors"
            title="Step Backward"
          >
            ⏮
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              isPlaying
                ? "bg-amber-600 text-white shadow-md shadow-amber-500/20"
                : "bg-slate-950 border border-slate-800 hover:border-slate-700 text-slate-200"
            }`}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
          <button
            onClick={() =>
              onSelectFrame((activeFrameIndex + 1) % sample.frame_count)
            }
            className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white text-xs transition-colors"
            title="Step Forward"
          >
            ⏭
          </button>
        </div>
      </div>

      {/* Frame Strip Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {sample.frame_tensors.map((frame, t_idx) => {
          const isActive = t_idx === activeFrameIndex;
          const pos = trajectory?.per_frame_positions[t_idx] || [0.5, 0.5];
          const height = frame[0]?.length || 16;
          const width = frame[0]?.[0]?.length || 16;

          return (
            <div
              key={sample.frame_ids[t_idx] || t_idx}
              onClick={() => onSelectFrame(t_idx)}
              className={`cursor-pointer group relative rounded-xl border p-2.5 transition-all ${
                isActive
                  ? "bg-slate-950 border-amber-500 ring-2 ring-amber-500/20 shadow-lg"
                  : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
              }`}
            >
              {/* Header inside frame card */}
              <div className="flex items-center justify-between text-[10px] font-mono mb-2">
                <span
                  className={`font-bold px-1.5 py-0.5 rounded ${
                    isActive
                      ? "bg-amber-500 text-slate-950"
                      : "bg-slate-800 text-slate-400 group-hover:text-slate-200"
                  }`}
                >
                  t = {t_idx}
                </span>
                <span className="text-slate-500 text-[9px]">
                  ({pos[0].toFixed(2)}, {pos[1].toFixed(2)})
                </span>
              </div>

              {/* Pixel Canvas Grid Container */}
              <div className="relative aspect-square w-full rounded-lg overflow-hidden border border-slate-800/80 bg-slate-950 flex items-center justify-center">
                <svg
                  viewBox={`0 0 ${width} ${height}`}
                  className="w-full h-full object-contain"
                  style={{ imageRendering: "pixelated" }}
                >
                  {frame[0]?.map((row, y) =>
                    row.map((_, x) => {
                      const r = Math.round((frame[0]?.[y]?.[x] ?? 0) * 255);
                      const g = Math.round((frame[1]?.[y]?.[x] ?? 0) * 255);
                      const b = Math.round((frame[2]?.[y]?.[x] ?? 0) * 255);
                      return (
                        <rect
                          key={`${x}-${y}`}
                          x={x}
                          y={y}
                          width={1}
                          height={1}
                          fill={`rgb(${r},${g},${b})`}
                        />
                      );
                    })
                  )}
                </svg>

                {/* Trajectory Center Crosshair */}
                <div
                  className="absolute w-2.5 h-2.5 -translate-x-1/2 -translate-y-1/2 pointer-events-none rounded-full border border-amber-400/80 bg-amber-500/30"
                  style={{
                    left: `${pos[0] * 100}%`,
                    top: `${pos[1] * 100}%`,
                  }}
                />
              </div>

              {/* Footer info */}
              <div className="mt-2 text-[10px] font-mono text-slate-400 text-center truncate">
                {sample.frame_ids[t_idx]}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
