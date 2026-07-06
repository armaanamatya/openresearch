import styles from "./ReasoningChip.module.css";

export type ReasoningChipStatus = "pending" | "active" | "done";

interface ReasoningChipProps {
  label: string;
  /** Defaults to "done" — a settled tool-call/reasoning row (the common
   * case once a primitive_call has finished). "active" pulses the dot for
   * a call that is still in flight. */
  status?: ReasoningChipStatus;
  className?: string;
}

/** Gray tool-call chip row (alphaXiv screen `_5`): a status dot + a label,
 * e.g. "Loaded skill: orx". Reuses the same pending/active/done vocabulary
 * as Stepper so the two read as one system. Pure presentational — the
 * caller (T13) maps primitive_call events onto `label`/`status`. */
export function ReasoningChip({ label, status = "done", className }: ReasoningChipProps) {
  const classes = [styles.chip, styles[status], className].filter(Boolean).join(" ");

  return (
    <div className={classes} data-status={status}>
      <span className={styles.dot} aria-hidden="true" />
      <span className={styles.label}>{label}</span>
    </div>
  );
}
