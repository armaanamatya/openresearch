import type { LucideIcon } from "lucide-react";
import { BarChart3, BookOpen, Rocket, Users } from "lucide-react";

export type StellarTabId = "analyse" | "train" | "testing" | "deploy";

export type StellarTab = {
  id: StellarTabId;
  label: string;
  icon: LucideIcon;
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
