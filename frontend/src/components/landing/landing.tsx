import styles from "./landing.module.css";
import "./landing.global.css";

export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      {/* Phase 2 Tasks 3–7 fill this in section by section */}
      <div
        style={{
          padding: "200px 40px",
          textAlign: "center",
          color: "var(--landing-ink-3)",
          fontFamily: "var(--font-jetbrains-mono), monospace"
        }}
      >
        § Landing scaffolded · sections forthcoming.
      </div>
    </div>
  );
}
