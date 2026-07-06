import { Check } from "lucide-react";
import styles from "./Stepper.module.css";

export type StepStatus = "pending" | "active" | "done";

export interface StepperStep {
  label: string;
  status: StepStatus;
}

interface StepperProps {
  steps: StepperStep[];
  className?: string;
}

/** Ordered step list (alphaXiv screen D, "Run Autoresearch"): a real `<ol>`
 * of numbered markers that fill in as steps complete, with a small pulsing
 * dot on whichever step is currently active. Numbering is real ordinal
 * information here (a genuine sequence), not decoration. */
export function Stepper({ steps, className }: StepperProps) {
  const classes = [styles.stepper, className].filter(Boolean).join(" ");

  return (
    <ol className={classes}>
      {steps.map((step, index) => {
        const statusClass = styles[step.status];
        const itemClasses = [styles.step, statusClass].filter(Boolean).join(" ");

        return (
          <li
            key={`${index}-${step.label}`}
            className={itemClasses}
            data-status={step.status}
            aria-current={step.status === "active" ? "step" : undefined}
          >
            <span className={styles.marker}>
              {step.status === "done" ? (
                <Check size={14} aria-hidden="true" />
              ) : (
                index + 1
              )}
              {step.status === "active" && (
                <span className={styles.liveDot} aria-hidden="true" />
              )}
            </span>
            <span className={styles.label}>{step.label}</span>
          </li>
        );
      })}
    </ol>
  );
}
