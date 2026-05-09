import type { LucideIcon } from "lucide-react";
import { BarChart3, BookOpen, Rocket, Users } from "lucide-react";

export type StellarTabId = "analyse" | "train" | "testing" | "deploy";

export type StellarTab = {
  id: StellarTabId;
  label: string;
  icon: LucideIcon;
};

export type StellarMetric = {
  label: string;
  value: string;
};

export type StellarChecklistItem = {
  label: string;
  complete: boolean;
};

export type StellarOverlay = {
  eyebrow: string;
  title: string;
  description: string;
  accentClass: string;
  progress?: number;
  metrics?: StellarMetric[];
  steps?: string[];
  checklist?: StellarChecklistItem[];
  successLabel?: string;
  ctaLabel?: string;
};

export const stellarTabs: StellarTab[] = [
  {
    id: "analyse",
    label: "Analyse",
    icon: BarChart3
  },
  {
    id: "train",
    label: "Train",
    icon: BookOpen
  },
  {
    id: "testing",
    label: "Testing",
    icon: Users
  },
  {
    id: "deploy",
    label: "Deploy",
    icon: Rocket
  }
];

export const stellarVideoSource =
  "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260319_165750_358b1e72-c921-48b7-aaac-f200994f32fb.mp4";

export const stellarOverlays: Record<StellarTabId, StellarOverlay> = {
  analyse: {
    eyebrow: "Analyse",
    title: "Set Up Your AI Workspace",
    description: "Connect your stack, define the objective, and stage a shared workspace for your team.",
    accentClass: "bg-violet-500",
    progress: 25,
    steps: ["Workspace", "Sources", "Automations", "Review"]
  },
  train: {
    eyebrow: "Train",
    title: "AI Model Training",
    description: "Tune model behavior against real usage data with live metrics and checkpoint snapshots.",
    accentClass: "bg-orange-400",
    progress: 67,
    metrics: [
      { label: "Samples", value: "84.2K" },
      { label: "Accuracy", value: "96.4%" },
      { label: "Latency", value: "142ms" },
      { label: "GPU Load", value: "71%" }
    ]
  },
  testing: {
    eyebrow: "Testing",
    title: "Test Suite Results",
    description: "Validation flows are green across performance, reliability, and regression coverage.",
    accentClass: "bg-emerald-500",
    successLabel: "127/127 tests",
    metrics: [
      { label: "Integration", value: "42" },
      { label: "Regression", value: "38" },
      { label: "Latency", value: "21" },
      { label: "Security", value: "26" }
    ]
  },
  deploy: {
    eyebrow: "Deploy",
    title: "Deploy to Production",
    description: "Ship the approved release with clear rollout gates, observability, and fallback controls.",
    accentClass: "bg-sky-500",
    checklist: [
      { label: "Release build approved", complete: true },
      { label: "Observability connected", complete: true },
      { label: "Rollback policy synced", complete: true },
      { label: "Traffic window open", complete: false }
    ],
    ctaLabel: "Deploy Now"
  }
};

export const stellarLogos = [
  "INTERSCOPE",
  "SPOTIFY",
  "Nexera",
  "M3",
  "LAURA COLE",
  "vertex"
] as const;
