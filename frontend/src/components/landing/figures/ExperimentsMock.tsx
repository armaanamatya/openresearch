/**
 * §4.0 Experiments mock — two-column layout: a reproduction-trajectory
 * line chart (chart-col) on the left with baseline + RLM curves, a band
 * envelope, and the paper-target dashed reference at 0.631; and an
 * ACTIVE RUNS list (runs-col) on the right with 6 run rows (2 with the
 * `match` class).
 *
 * Sourced from docs/design/landing-source/index.html lines 1192-1316.
 * Maps the `.mock-exp`, `.chart-col`, `.chart-head`, `.legend`,
 * `.runs-col`, `.runs-h`, `.run-row`, `.run-row.match`, `.id`,
 * `.name`, `.score`, `.status`, `.sw` descendant selectors.
 *
 * The inline `<svg>` uses an `.nlbl` axis-label class scoped via a
 * local `<style>` block inside `<defs>` — same pattern as HeroTree /
 * ComprehensionMock — since `.nlbl` is not in the module CSS. The
 * `<pattern id="grid">` is defined inline so the chart renders
 * standalone without depending on other figures' defs.
 *
 * Source HTML uses kebab-case SVG attributes (`stroke-width`,
 * `stroke-dasharray`, `text-anchor`); JSX requires camelCase.
 * Source `var(--ink)` / `var(--accent)` / `var(--ink-3)` /
 * `var(--ink-4)` / `var(--mono)` map to the `--landing-*` namespace
 * used by the ported design system.
 */
import styles from "../landing.module.css";

export function ExperimentsMock(): React.JSX.Element {
  return (
    <div className={`${styles.mock} ${styles["mock-exp"]}`}>
      {/* ----- left : chart with title + legend + SVG + bottom meta ----- */}
      <div className={styles["chart-col"]}>
        <div className={styles["chart-head"]}>
          <h5>Reproduction trajectory</h5>
          <div className={styles.legend}>
            <span><span className={styles.sw} style={{ background: "var(--landing-ink)" }} />baseline</span>
            <span><span className={styles.sw} style={{ background: "var(--landing-accent)" }} />RLM</span>
            <span><span className={styles.sw} style={{ background: "var(--landing-ink-4)", borderTop: "1px dashed", height: 0, borderColor: "var(--landing-ink-3)" }} />paper</span>
          </div>
        </div>
        <svg viewBox="0 0 700 280" width="100%" aria-hidden>
          <defs>
            <pattern id="grid" width="70" height="28" patternUnits="userSpaceOnUse">
              <path d="M70 0 L0 0 0 28" fill="none" stroke="oklch(0.235 0.004 270)" strokeWidth="1" />
            </pattern>
            <style>{`.nlbl { font-family: var(--font-jetbrains-mono), ui-monospace, monospace; font-size: 10px; fill: oklch(0.56 0.005 90); letter-spacing: 0.04em; }`}</style>
          </defs>
          <rect x="40" y="10" width="640" height="230" fill="url(#grid)" />

          {/* y axis labels */}
          <text x="32" y="20"  textAnchor="end" className="nlbl">0.70</text>
          <text x="32" y="77"  textAnchor="end" className="nlbl">0.65</text>
          <text x="32" y="134" textAnchor="end" className="nlbl">0.60</text>
          <text x="32" y="191" textAnchor="end" className="nlbl">0.55</text>
          <text x="32" y="244" textAnchor="end" className="nlbl">0.50</text>

          {/* x axis */}
          <text x="40"  y="266" className="nlbl">step 0</text>
          <text x="200" y="266" className="nlbl">2k</text>
          <text x="360" y="266" className="nlbl">4k</text>
          <text x="520" y="266" className="nlbl">6k</text>
          <text x="660" y="266" className="nlbl" textAnchor="end">8k</text>

          {/* paper target dashed line @ 0.631 */}
          <line x1="40" y1="92" x2="680" y2="92" stroke="oklch(0.56 0.005 90)" strokeDasharray="3 4" strokeWidth="1" />
          <text x="676" y="86" className="nlbl" textAnchor="end">PAPER · 0.631</text>

          {/* baseline curve (gray) */}
          <path d="M40,230 C 120,210 180,180 260,160 S 420,140 540,138 L 680,140"
                fill="none" stroke="oklch(0.78 0.005 90)" strokeWidth="1.5" />

          {/* RLM curve (accent) */}
          <path d="M40,235 C 120,210 180,170 260,140 S 420,108 540,98 L 680,96"
                fill="none" stroke="oklch(0.80 0.16 70)" strokeWidth="1.5" />

          {/* band */}
          <path d="M40,237 C 120,212 180,172 260,142 S 420,110 540,100 L 680,98 L 680,94 C 540,96 420,106 260,138 S 120,208 40,233 Z"
                fill="oklch(0.80 0.16 70 / .10)" />

          {/* dots last */}
          <circle cx="680" cy="96"  r="3" fill="oklch(0.80 0.16 70)" />
          <circle cx="680" cy="140" r="3" fill="oklch(0.78 0.005 90)" />
        </svg>

        <div
          style={{
            display: "flex",
            gap: "24px",
            marginTop: "14px",
            fontFamily: "var(--font-jetbrains-mono), ui-monospace, monospace",
            fontSize: "11px",
            color: "var(--landing-ink-3)",
          }}
        >
          <span>step <span style={{ color: "var(--landing-ink)" }}>8,000</span></span>
          <span>seed <span style={{ color: "var(--landing-ink)" }}>0…4</span></span>
          <span>budget·d <span style={{ color: "var(--landing-ink)" }}>3</span></span>
          <span style={{ marginLeft: "auto", color: "var(--landing-accent)" }}>▲ 0.627 ± 0.004</span>
        </div>
      </div>

      {/* ----- right : ACTIVE RUNS list ----- */}
      <div className={styles["runs-col"]}>
        <div className={styles["runs-h"]} style={{ paddingTop: "6px" }}>ACTIVE RUNS · 32</div>
        <div className={`${styles["run-row"]} ${styles.match}`}>
          <div>
            <div className={styles.id}>RUN-7e3a-s0 · §4.2</div>
            <div className={styles.name}>GSM8K · pass@1 · seed 0</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className={styles.score}>0.627</div>
            <div className={styles.status}>match</div>
          </div>
        </div>
        <div className={`${styles["run-row"]} ${styles.match}`}>
          <div>
            <div className={styles.id}>RUN-7e3a-s1 · §4.2</div>
            <div className={styles.name}>GSM8K · pass@1 · seed 1</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className={styles.score}>0.623</div>
            <div className={styles.status}>match</div>
          </div>
        </div>
        <div className={styles["run-row"]}>
          <div>
            <div className={styles.id}>RUN-7e3a-s2 · §4.2</div>
            <div className={styles.name}>GSM8K · pass@1 · seed 2</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className={styles.score}>…</div>
            <div className={styles.status}>running</div>
          </div>
        </div>
        <div className={`${styles["run-row"]} ${styles.match}`}>
          <div>
            <div className={styles.id}>RUN-7e3a · §4.3</div>
            <div className={styles.name}>MATH · pass@1</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className={styles.score}>0.412</div>
            <div className={styles.status}>match</div>
          </div>
        </div>
        <div className={styles["run-row"]}>
          <div>
            <div className={styles.id}>RUN-7e3a · §4.3</div>
            <div className={styles.name}>HumanEval+ · pass@1</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className={styles.score}>0.586</div>
            <div className={styles.status}>−0.018</div>
          </div>
        </div>
        <div className={styles["run-row"]}>
          <div>
            <div className={styles.id}>RUN-7e3a · §6.1</div>
            <div className={styles.name}>Ablation · no-verifier</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className={styles.score}>…</div>
            <div className={styles.status}>queued</div>
          </div>
        </div>
      </div>
    </div>
  );
}
