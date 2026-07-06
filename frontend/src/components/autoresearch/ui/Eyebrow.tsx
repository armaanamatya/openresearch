import type { ReactNode } from "react";
import styles from "./Eyebrow.module.css";

interface EyebrowProps {
  children: ReactNode;
  className?: string;
}

/** Small tracked uppercase label sitting above a headline (e.g. "ARXIV
 * 2605.15155"). Uppercasing is done in CSS (text-transform), not by
 * mutating the string, so the DOM/a11y tree keeps the text as authored. */
export function Eyebrow({ children, className }: EyebrowProps) {
  const classes = [styles.eyebrow, className].filter(Boolean).join(" ");

  return <p className={classes}>{children}</p>;
}
