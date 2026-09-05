"use client";

import React from "react";
import { CalibrationReportPayload, TemperatureScalingResultPayload } from "../types";

interface TemperatureScalingCardProps {
  uncalibratedReport: CalibrationReportPayload;
  calibratedReport: CalibrationReportPayload | null;
  temperatureScaling: TemperatureScalingResultPayload | null;
}

export const TemperatureScalingCard: React.FC<TemperatureScalingCardProps> = ({
  uncalibratedReport,
  calibratedReport,
  temperatureScaling,
}) => {
  if (!temperatureScaling) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h2 className="text-sm font-bold text-slate-100">Temperature Scaling Calibration</h2>
        <p className="text-xs text-slate-400 mt-2">
          Validation logits not available for post-hoc temperature optimization.
        </p>
      </div>
    );
  }

  const calReport = calibratedReport || uncalibratedReport;
  const eceDelta = (calReport.ece - uncalibratedReport.ece) * 100;
  const nllBefore = uncalibratedReport.nll ?? uncalibratedReport.negative_log_likelihood ?? 0;
  const nllAfter = calReport.nll ?? calReport.negative_log_likelihood ?? 0;
  const nllDelta = nllAfter - nllBefore;
  const brierDelta = calReport.brier_score - uncalibratedReport.brier_score;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-100">
              Post-Hoc Temperature Scaling Optimization
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-violet-950 text-violet-400 border border-violet-800/40">
              1D VALIDATION NLL FIT
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Scalar parameter T &gt; 0 scales logits z / T before softmax without modifying network weights
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-mono">Fitted T*:</span>
          <span className="text-sm font-bold font-mono px-2.5 py-1 rounded-lg bg-rose-950/80 text-rose-300 border border-rose-800/50">
            T = {temperatureScaling.fitted_temperature.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Comparison Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 my-4">
        {/* ECE */}
        <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Expected Calib Error</div>
          <div className="flex items-baseline justify-between mt-1">
            <span className="text-xs text-slate-400 line-through">
              {(uncalibratedReport.ece * 100).toFixed(2)}%
            </span>
            <span className="text-base font-bold font-mono text-emerald-400">
              {(calReport.ece * 100).toFixed(2)}%
            </span>
          </div>
          <div className="text-[10px] text-emerald-400 font-mono mt-0.5">
            {eceDelta <= 0 ? `${eceDelta.toFixed(2)}% ECE` : `+${eceDelta.toFixed(2)}% ECE`}
          </div>
        </div>

        {/* NLL */}
        <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Validation NLL</div>
          <div className="flex items-baseline justify-between mt-1">
            <span className="text-xs text-slate-400 line-through">
              {temperatureScaling.validation_nll_before.toFixed(3)}
            </span>
            <span className="text-base font-bold font-mono text-cyan-400">
              {temperatureScaling.validation_nll_after.toFixed(3)}
            </span>
          </div>
          <div className="text-[10px] text-cyan-400 font-mono mt-0.5">
            {nllDelta <= 0 ? `${nllDelta.toFixed(3)} NLL` : `+${nllDelta.toFixed(3)} NLL`}
          </div>
        </div>

        {/* Brier Score */}
        <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Brier Score</div>
          <div className="flex items-baseline justify-between mt-1">
            <span className="text-xs text-slate-400 line-through">
              {uncalibratedReport.brier_score.toFixed(3)}
            </span>
            <span className="text-base font-bold font-mono text-indigo-400">
              {calReport.brier_score.toFixed(3)}
            </span>
          </div>
          <div className="text-[10px] text-indigo-400 font-mono mt-0.5">
            {brierDelta <= 0 ? `${brierDelta.toFixed(3)} BS` : `+${brierDelta.toFixed(3)} BS`}
          </div>
        </div>

        {/* Accuracy Invariant */}
        <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
          <div className="text-[10px] uppercase font-mono text-slate-400">Accuracy Invariant</div>
          <div className="flex items-baseline justify-between mt-1">
            <span className="text-xs text-slate-400">
              {(uncalibratedReport.accuracy * 100).toFixed(1)}%
            </span>
            <span className="text-base font-bold font-mono text-emerald-400">
              {(calReport.accuracy * 100).toFixed(1)}%
            </span>
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-0.5">
            Argmax strictly preserved
          </div>
        </div>
      </div>

      {/* Proof / Verification Footer */}
      <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 text-xs text-slate-400 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span>
            Search Range: [{temperatureScaling.search_range[0]}, {temperatureScaling.search_range[1]}] • Method: {temperatureScaling.fitting_method}
          </span>
        </div>
        <div className="font-mono text-[11px] text-slate-400">
          Fit Iterations: {temperatureScaling.iterations}
        </div>
      </div>
    </div>
  );
};
