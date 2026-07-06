import { useId } from "react";
import { Github, ExternalLink } from "lucide-react";
import styles from "./RepoField.module.css";

interface RepoFieldProps {
  /** Current "owner/repo" value (or whatever text the caller is editing). */
  value: string;
  onChange: (value: string) => void;
  label: string;
  placeholder?: string;
  className?: string;
}

/** Labelled repo-confirm input (alphaXiv screen `_4`): a bordered text field
 * plus a GitHub-link affordance rendered under it once there's a value to
 * link to. Controlled + presentational only — the caller owns the value and
 * decides what, if anything, "confirm" does with it. */
export function RepoField({ value, onChange, label, placeholder, className }: RepoFieldProps) {
  const inputId = useId();
  const trimmed = value.trim();
  const classes = [styles.field, className].filter(Boolean).join(" ");

  return (
    <div className={classes}>
      <label htmlFor={inputId} className={styles.label}>
        {label}
      </label>
      <input
        id={inputId}
        type="text"
        className={styles.input}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
      {trimmed && (
        <a
          className={styles.repoLink}
          href={`https://github.com/${trimmed}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Github size={14} aria-hidden="true" />
          <span>{trimmed}</span>
          <ExternalLink size={12} aria-hidden="true" />
        </a>
      )}
    </div>
  );
}
