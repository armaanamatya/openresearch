import type { MouseEventHandler, ReactNode } from "react";
import styles from "./PrimaryButton.module.css";

interface PrimaryButtonProps {
  children: ReactNode;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
  /** Defaults to "button" so a CTA never accidentally submits a form
   * unless the caller opts in with type="submit". */
  type?: "button" | "submit" | "reset";
  className?: string;
}

export function PrimaryButton({
  children,
  onClick,
  disabled = false,
  type = "button",
  className,
}: PrimaryButtonProps) {
  const classes = [styles.button, className].filter(Boolean).join(" ");

  return (
    <button type={type} onClick={onClick} disabled={disabled} className={classes}>
      {children}
    </button>
  );
}
