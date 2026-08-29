"use client";

import React from "react";
import { RepresentationGeometryReport } from "../types";

interface MetricOverviewStripProps {
  report: RepresentationGeometryReport | null;
}

export const MetricOverviewStrip: React.FC<MetricOverviewStripProps> = ({
  report,
}) => {
  if (!report) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 py-4 animate-pulse">
        {Array.from({ length: 7 }).map((_, i) => (
          <div key={i} className="h-20 bg-zinc-900/60 rounded-xl border border-zinc-800/60" />
        ))}
      </div>
    );
  }

  const {
    num_samples,
    feature_dim,
    num_classes,
    centroid_geometry,
    neighborhood_geometry,
    pca_projection,
    warnings,
  } = report;

  const pcaVarianceSum =
    pca_projection.cumulative_explained_variance.length >= 2
      ? pca_projection.cumulative_explained_variance[1] * 100
      : (pca_projection.cumulative_explained_variance[0] || 0) * 100;

  return (
    <div className="space-y-2 py-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {/* Samples & Dimension */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-zinc-400">Dataset Shape</div>
          <div className="text-lg font-bold font-mono text-zinc-100 mt-1">
            {num_samples} <span className="text-zinc-500 font-normal text-xs">&times; {feature_dim}D</span>
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">{num_classes} Classes</div>
        </div>

        {/* Intra-Class Compactness */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-cyan-400">Intra Compactness (d&#772;)</div>
          <div className="text-lg font-bold font-mono text-cyan-200 mt-1">
            {centroid_geometry.mean_intra_class_distance.toFixed(3)}
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">Sample &rarr; Centroid</div>
        </div>

        {/* Inter-Class Separation */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-indigo-400">Inter Separation (&#916;)</div>
          <div className="text-lg font-bold font-mono text-indigo-200 mt-1">
            {centroid_geometry.mean_inter_class_centroid_distance.toFixed(3)}
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">Centroid &rarr; Centroid</div>
        </div>

        {/* Separation / Compactness Ratio */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-purple-400">Sep / Comp Ratio</div>
          <div className="text-lg font-bold font-mono text-purple-200 mt-1">
            {centroid_geometry.separation_to_compactness_ratio.toFixed(2)}x
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">Higher is better</div>
        </div>

        {/* k-NN Consistency */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-emerald-400">
            {neighborhood_geometry.k}-NN Consistency
          </div>
          <div className="text-lg font-bold font-mono text-emerald-200 mt-1">
            {(neighborhood_geometry.mean_label_consistency * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">
            Median: {(neighborhood_geometry.median_label_consistency * 100).toFixed(1)}%
          </div>
        </div>

        {/* PCA Variance */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-amber-400">PCA Variance (2D)</div>
          <div className="text-lg font-bold font-mono text-amber-200 mt-1">
            {pcaVarianceSum.toFixed(1)}%
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">
            PC1: {((pca_projection.explained_variance_ratio[0] || 0) * 100).toFixed(1)}%
          </div>
        </div>

        {/* Candidate Failures */}
        <div className="p-3 bg-zinc-900/70 border border-zinc-800/80 rounded-xl">
          <div className="text-[11px] font-mono text-rose-400">Candidate Failures</div>
          <div className="text-lg font-bold font-mono text-rose-200 mt-1">
            {report.candidate_failures.length}
          </div>
          <div className="text-[10px] text-zinc-500 font-mono mt-0.5">
            {report.candidate_failures.length > 0 ? "Ambiguous points" : "Clean geometry"}
          </div>
        </div>
      </div>

      {/* Warnings bar */}
      {warnings && warnings.length > 0 && (
        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-amber-950/40 border border-amber-800/50 text-amber-300 text-xs font-mono">
          <span className="text-amber-400 font-bold">&#9888; Note:</span>
          <span>{warnings.join(" | ")}</span>
        </div>
      )}
    </div>
  );
};
