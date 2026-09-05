"use client";

import React from "react";
import { CalibrationReportPayload, OODBinaryEvaluationSummaryPayload, TemperatureScalingResultPayload } from "../types";

interface UncertaintyOverviewStripProps {
  calibrationReport: CalibrationReportPayload;
  calibratedReport: CalibrationReportPayload | null;
  mode?: string;
  calibrationMode?: string;
  temperatureScaling: TemperatureScalingResultPayload | null;
  oodEvaluations?: Record<string, OODBinaryEvaluationSummaryPayload>;
  activeOODEval?: OODBinaryEvaluationSummaryPayload;
  selectedOODMethod?: string;
}

export const UncertaintyOverviewStrip: React.FC<UncertaintyOverviewStripProps> = ({
  calibrationReport,
  calibratedReport,
  mode,
  calibrationMode,
  temperatureScaling,
  oodEvaluations,
  activeOODEval,
  selectedOODMethod,
}) => {
  const currentMode = calibrationMode || mode || "uncalibrated";
  const activeCal =
    currentMode === "temperature_scaled" && calibratedReport
      ? calibratedReport
      : calibrationReport;

  const currentOODEval =
    activeOODEval ||
    (oodEvaluations && selectedOODMethod ? oodEvaluations[selectedOODMethod] : undefined) ||
    (oodEvaluations ? Object.values(oodEvaluations)[0] : undefined);

  const accPercent = ((activeCal.accuracy ?? 0) * 100).toFixed(1);
  const confPercent = ((activeCal.mean_confidence ?? 0) * 100).toFixed(1);
  const ecePercent = ((activeCal.ece ?? 0) * 100).toFixed(2);
  const brier = (activeCal.brier_score ?? 0).toFixed(3);
  const nll = (activeCal.nll ?? activeCal.negative_log_likelihood ?? 0).toFixed(3);
  const entropy = (activeCal.mean_predictive_entropy ?? 0).toFixed(3);
  const auroc = currentOODEval ? (currentOODEval.auroc * 100).toFixed(1) : "N/A";
  const fittedT = temperatureScaling ? temperatureScaling.fitted_temperature.toFixed(2) : "1.00";

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
      {/* Accuracy */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Accuracy</div>
        <div className="text-xl font-black font-mono text-emerald-400 mt-1">{accPercent}%</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Top-1 Empirical</div>
      </div>

      {/* Mean Confidence */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Mean Confidence</div>
        <div className="text-xl font-black font-mono text-cyan-400 mt-1">{confPercent}%</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Softmax Max Prob</div>
      </div>

      {/* Expected Calibration Error */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">ECE (10 Bins)</div>
        <div className="text-xl font-black font-mono text-amber-400 mt-1">{ecePercent}%</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Calibration Error</div>
      </div>

      {/* Multiclass Brier Score */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Brier Score</div>
        <div className="text-xl font-black font-mono text-slate-200 mt-1">{brier}</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Mean Sq Error</div>
      </div>

      {/* Negative Log Likelihood */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Eval NLL</div>
        <div className="text-xl font-black font-mono text-violet-400 mt-1">{nll}</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Cross-Entropy Loss</div>
      </div>

      {/* Predictive Entropy */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Entropy H(p)</div>
        <div className="text-xl font-black font-mono text-indigo-400 mt-1">{entropy}</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Nats per sample</div>
      </div>

      {/* OOD AUROC */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">OOD AUROC</div>
        <div className="text-xl font-black font-mono text-amber-300 mt-1">{auroc}%</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Binary Sep (OOD+)</div>
      </div>

      {/* Fitted Temperature */}
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl shadow-sm">
        <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Fitted Temp T*</div>
        <div className="text-xl font-black font-mono text-rose-400 mt-1">{fittedT}</div>
        <div className="text-[10px] text-slate-500 mt-0.5">Val-Fit Scalar</div>
      </div>
    </div>
  );
};
