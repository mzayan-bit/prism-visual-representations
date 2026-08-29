"use client";

import React from "react";
import { NearestNeighborEntry, SampleNeighborhood } from "../types";

interface NeighborhoodPanelProps {
  neighborhood: SampleNeighborhood | null;
  selectedSampleId: string | null;
  onSelectNeighbor: (sampleId: string) => void;
}

export const NeighborhoodPanel: React.FC<NeighborhoodPanelProps> = ({
  neighborhood,
  selectedSampleId,
  onSelectNeighbor,
}) => {
  if (!selectedSampleId || !neighborhood) {
    return (
      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-5 text-center flex flex-col items-center justify-center min-h-[300px] text-zinc-500 font-mono text-xs">
        <div className="w-12 h-12 rounded-full bg-zinc-800/50 flex items-center justify-center text-xl mb-3 text-zinc-400">
          &#8857;
        </div>
        <div className="font-bold text-zinc-300 mb-1">No Sample Selected</div>
        <p className="max-w-xs text-zinc-500">
          Click any point in the PCA scatter plot to inspect its local neighborhood, nearest neighbors, and distance to class centroids.
        </p>
      </div>
    );
  }

  const {
    query_sample_id,
    query_label,
    neighbors,
    same_class_fraction,
    distance_to_own_centroid,
    nearest_competing_centroid_distance,
  } = neighborhood;

  return (
    <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-3">
        <div>
          <div className="text-[11px] font-mono text-cyan-400 font-bold uppercase">
            Query Sample
          </div>
          <div className="text-base font-bold font-mono text-zinc-100 mt-0.5">
            {query_sample_id}
          </div>
        </div>
        <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-cyan-950/80 border border-cyan-800 text-cyan-300">
          Class {query_label}
        </span>
      </div>

      {/* Centroid Distance Metrics */}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="p-2.5 bg-zinc-950/60 rounded-lg border border-zinc-800/60">
          <div className="text-[10px] text-zinc-400">Distance to Own Centroid</div>
          <div className="text-sm font-bold text-emerald-300 mt-0.5">
            {distance_to_own_centroid !== null
              ? distance_to_own_centroid.toFixed(4)
              : "N/A"}
          </div>
        </div>
        <div className="p-2.5 bg-zinc-950/60 rounded-lg border border-zinc-800/60">
          <div className="text-[10px] text-zinc-400">Distance to Foreign Centroid</div>
          <div className="text-sm font-bold text-indigo-300 mt-0.5">
            {nearest_competing_centroid_distance !== null
              ? nearest_competing_centroid_distance.toFixed(4)
              : "N/A"}
          </div>
        </div>
      </div>

      {/* Neighborhood Label Consistency Bar */}
      <div className="space-y-1.5 font-mono text-xs">
        <div className="flex justify-between text-zinc-400">
          <span>Neighborhood Consistency (k={neighbors.length})</span>
          <span
            className={`font-bold ${
              same_class_fraction >= 0.8
                ? "text-emerald-400"
                : same_class_fraction >= 0.5
                ? "text-amber-400"
                : "text-rose-400"
            }`}
          >
            {(same_class_fraction * 100).toFixed(0)}%
          </span>
        </div>
        <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              same_class_fraction >= 0.8
                ? "bg-emerald-500"
                : same_class_fraction >= 0.5
                ? "bg-amber-500"
                : "bg-rose-500"
            }`}
            style={{ width: `${same_class_fraction * 100}%` }}
          />
        </div>
      </div>

      {/* Nearest Neighbors Table */}
      <div className="space-y-2">
        <div className="text-xs font-mono font-bold text-zinc-300 flex items-center justify-between">
          <span>Ranked Nearest Neighbors</span>
          <span className="text-[10px] text-zinc-500">Ascending Distance</span>
        </div>
        <div className="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
          {neighbors.map((entry: NearestNeighborEntry) => (
            <div
              key={entry.neighbor_sample_id}
              onClick={() => onSelectNeighbor(entry.neighbor_sample_id)}
              className="flex items-center justify-between p-2 rounded-lg bg-zinc-950/70 border border-zinc-800/60 hover:border-zinc-700 cursor-pointer transition-colors text-xs font-mono"
            >
              <div className="flex items-center gap-2">
                <span className="w-5 h-5 rounded bg-zinc-800 text-zinc-400 flex items-center justify-center text-[10px] font-bold">
                  #{entry.rank}
                </span>
                <span className="font-medium text-zinc-200">
                  {entry.neighbor_sample_id}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-zinc-400 text-[11px]">
                  d = {entry.distance.toFixed(4)}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                    entry.same_class
                      ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800"
                      : "bg-rose-950/80 text-rose-300 border border-rose-800"
                  }`}
                >
                  Class {entry.neighbor_label} {entry.same_class ? "✓" : "✗"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
