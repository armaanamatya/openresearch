import type { ReactNode } from "react";
import styles from "./Card.module.css";

export type CardVariant = "default" | "panel";

interface CardProps {
  children: ReactNode;
  /** "default" = primary surface (bold border + full shadow). "panel" =
   * softer secondary surface (subtle border + shallow shadow). */
  variant?: CardVariant;
  className?: string;
}

export function Card({ children, variant = "default", className }: CardProps) {
  const variantClass = variant === "panel" ? styles.panel : styles.default;
  const classes = [styles.card, variantClass, className].filter(Boolean).join(" ");

  return (
    <div className={classes} data-variant={variant}>
      {children}
    </div>
  );
}
