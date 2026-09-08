import React from "react";
import {
  Layers,
  ShieldCheck,
  Sparkles,
  BarChart3,
  ArrowRight,
  Target,
  Scale,
  Database,
  CheckCircle2,
} from "lucide-react";
import { Card } from "../../ui/Card";
import { Button } from "../../ui/Button";
import { Badge } from "../../ui/Badge";
import { AppMode } from "../../ResearchPlatformNavigation";

export interface OverviewHubProps {
  onNavigate: (mode: AppMode) => void;
}

export const OverviewHub: React.FC<OverviewHubProps> = ({ onNavigate }) => {
  const researchQuestions = [
    {
      title: "How do learned representation manifolds evolve across layer depth?",
      category: "Geometry",
      icon: Layers,
      description:
        "Probe layer-wise representations using Jacobi PCA, singular value spectra, and separation-to-compactness ratios.",
      targetMode: "observatory" as AppMode,
      accent: "text-sky-400",
    },
    {
      title: "How robust are representations against common corruptions?",
      category: "Robustness",
      icon: ShieldCheck,
      description:
        "Measure internal latent representation drift vectors and accuracy drops across 6 perturbation families.",
      targetMode: "robustness" as AppMode,
      accent: "text-emerald-400",
    },
    {
      title: "What visual features do models attribute to class predictions?",
      category: "Explainability",
      icon: Sparkles,
      description:
        "Inspect Grad-CAM, input gradients, sliding-window occlusion, and CLS attention rollout agreement.",
      targetMode: "explainability" as AppMode,
      accent: "text-amber-400",
    },
    {
      title: "What representations transfer best to downstream tasks?",
      category: "Transfer & Pretraining",
      icon: Target,
      description:
        "Compare supervised, contrastive SimCLR, and masked autoencoder representations on spatial, temporal, and low-data transfer.",
      targetMode: "transfer" as AppMode,
      accent: "text-purple-400",
    },
    {
      title: "How honest is model predictive confidence under distribution shift?",
      category: "Reliability & Calibration",
      icon: Scale,
      description:
        "Analyze Expected Calibration Error (ECE), Platt temperature scaling, and multi-method OOD detection AUROC.",
      targetMode: "uncertainty" as AppMode,
      accent: "text-rose-400",
    },
    {
      title: "What conclusions are genuinely supported across all experiments?",
      category: "Synthesis",
      icon: BarChart3,
      description:
        "Explore the 810-cell cross-paradigm benchmark matrix, Pareto trade-off frontiers, and evidence-gap planning.",
      targetMode: "benchmark" as AppMode,
      accent: "text-sky-400",
    },
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto py-2">
      {/* Hero / Scientific Statement */}
      <div className="flex flex-col gap-3 pb-6 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Badge variant="accent" size="sm">
            Research Platform
          </Badge>
          <Badge variant="synthetic" size="sm">
            v1.1.0 Public Release
          </Badge>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100">
          Probing the Evolution of Visual Representations
        </h1>
        <p className="text-sm sm:text-base text-slate-400 max-w-3xl leading-relaxed">
          PRISM is a controlled computer-vision research platform designed to investigate what deep
          neural networks learn internally. Rather than solely tracking task accuracy, PRISM probes
          manifold geometry, transfer dynamics, corruption invariance, attribution heatmaps,
          multimodal alignment, and calibration.
        </p>

        {/* Quick Launch Buttons */}
        <div className="flex flex-wrap items-center gap-3 pt-2">
          <Button
            variant="primary"
            size="md"
            onClick={() => onNavigate("benchmark")}
            icon={<BarChart3 className="w-4 h-4" />}
          >
            Launch Benchmark Observatory
          </Button>
          <Button
            variant="secondary"
            size="md"
            onClick={() => onNavigate("observatory")}
            icon={<Layers className="w-4 h-4" />}
          >
            Explore Representation Geometry
          </Button>
        </div>
      </div>

      {/* Research Question Navigator */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100">Research Inquiries</h2>
            <p className="text-xs text-slate-400">
              Select a scientific question to open the specialized research workspace.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {researchQuestions.map((q, idx) => {
            const Icon = q.icon;
            return (
              <Card
                key={idx}
                hoverable
                onClick={() => onNavigate(q.targetMode)}
                className="group flex flex-col justify-between p-5 bg-slate-900/60 hover:bg-slate-900/90 border-slate-800 hover:border-slate-700 transition-all"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono font-medium text-slate-400 uppercase tracking-wider">
                      {q.category}
                    </span>
                    <Icon className={`w-4 h-4 ${q.accent}`} />
                  </div>
                  <h3 className="text-sm font-semibold text-slate-200 group-hover:text-sky-300 transition-colors leading-snug">
                    {q.title}
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">{q.description}</p>
                </div>

                <div className="pt-4 mt-2 flex items-center gap-1.5 text-xs font-medium text-slate-500 group-hover:text-sky-400 transition-colors">
                  <span>Enter workspace</span>
                  <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* System Overview Strip */}
      <div className="p-5 rounded-lg bg-slate-900/40 border border-slate-800 space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
          Platform Coverage & Methodological Standards
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div className="space-y-1">
            <span className="text-slate-500 text-[11px]">Architectures</span>
            <div className="text-slate-200 font-semibold flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" />
              CNN / ResNet / ViT
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-slate-500 text-[11px]">Learning Paradigms</span>
            <div className="text-slate-200 font-semibold flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              Supervised / SimCLR / MAE / CLIP
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-slate-500 text-[11px]">Evaluated Matrix</span>
            <div className="text-slate-200 font-mono font-semibold flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-purple-400" />
              810 Observed Cells
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-slate-500 text-[11px]">Quality Gates</span>
            <div className="text-slate-200 font-mono font-semibold flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              657 / 657 Tests Passing
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
