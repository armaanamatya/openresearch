import { FileText, BookOpen, Settings, Headphones, type LucideIcon } from "lucide-react";
import styles from "./PaperTabs.module.css";

export const PAPER_TABS = ["Paper", "Blog", "Autoresearch", "Audio"] as const;

export type PaperTab = (typeof PAPER_TABS)[number];

interface PaperTabsProps {
  active: PaperTab;
  onTabChange: (tab: PaperTab) => void;
  className?: string;
}

const TAB_ICONS: Record<PaperTab, LucideIcon> = {
  Paper: FileText,
  Blog: BookOpen,
  Autoresearch: Settings,
  Audio: Headphones,
};

/** Paper/Blog/Autoresearch/Audio tab bar (alphaXiv screen `_autoresearc`):
 * a maroon active-underline sits under whichever tab is selected. Pure
 * presentational — the caller owns which panel `active` renders. */
export function PaperTabs({ active, onTabChange, className }: PaperTabsProps) {
  const classes = [styles.tabs, className].filter(Boolean).join(" ");

  return (
    <div className={classes} role="tablist" aria-label="Paper views">
      {PAPER_TABS.map((tab) => {
        const Icon = TAB_ICONS[tab];
        const isActive = tab === active;
        const tabClasses = [styles.tab, isActive ? styles.active : ""]
          .filter(Boolean)
          .join(" ");

        return (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={tabClasses}
            data-active={isActive ? "true" : "false"}
            onClick={() => onTabChange(tab)}
          >
            <Icon size={14} aria-hidden="true" />
            <span>{tab}</span>
          </button>
        );
      })}
    </div>
  );
}
