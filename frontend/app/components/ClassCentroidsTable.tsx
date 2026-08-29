"use client";

import React from "react";
import { CentroidGeometryReport } from "../types";

interface ClassCentroidsTableProps {
  centroidGeometry: CentroidGeometryReport;
}

export const ClassCentroidsTable: React.FC<ClassCentroidsTableProps> = ({
  centroidGeometry,
}) => {
  const { class_order, class_centroids } = centroidGeometry;

  return (
    <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-zinc-800/80">
        <div>
          <h3 className="text-xs font-mono font-bold text-zinc-100 flex items-center gap-2">
            <span>Class Centroid Geometry & Compactness</span>
            <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/40">
              {class_order.length} Classes
            </span>
          </h3>
          <p className="text-[10px] text-zinc-400">
            Centroid norms, intra-class dispersion radii, and competing class boundaries
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-zinc-800 text-[10px] text-zinc-400 uppercase">
              <th className="pb-2 font-medium">Class</th>
              <th className="pb-2 font-medium">Samples</th>
              <th className="pb-2 font-medium">Centroid Norm</th>
              <th className="pb-2 font-medium">Intra Mean &plusmn; Std</th>
              <th className="pb-2 font-medium">Radius 90%</th>
              <th className="pb-2 font-medium">Nearest Competitor</th>
              <th className="pb-2 font-medium">Separation &Delta;</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {class_order.map((cId) => {
              const info = class_centroids[cId];
              if (!info) return null;

              return (
                <tr key={cId} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="py-2.5 font-bold text-zinc-200">
                    <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                      Class {info.class_name || cId}
                    </span>
                  </td>
                  <td className="py-2.5 text-zinc-400">{info.sample_count}</td>
                  <td className="py-2.5 text-cyan-300">{info.centroid_norm.toFixed(3)}</td>
                  <td className="py-2.5 text-emerald-300">
                    {info.intra_class_mean_distance.toFixed(3)}{" "}
                    <span className="text-zinc-500 text-[10px]">
                      &plusmn; {info.intra_class_std_distance.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-2.5 text-amber-300">
                    {info.intra_class_radius_90.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-zinc-300">
                    {info.nearest_competing_class ? (
                      <span className="px-1.5 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800 text-[10px]">
                        Class {info.nearest_competing_class}
                      </span>
                    ) : (
                      "N/A"
                    )}
                  </td>
                  <td className="py-2.5 text-indigo-300 font-bold">
                    {info.distance_to_nearest_competing_centroid !== null
                      ? info.distance_to_nearest_competing_centroid.toFixed(3)
                      : "N/A"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
