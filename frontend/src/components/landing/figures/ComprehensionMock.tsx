/**
 * §1.0 Comprehension mock — paper preview, extracted claim cards,
 * and a citation-graph/rubric column.
 *
 * Sourced from docs/design/landing-source/index.html lines 895-986.
 * Class names mirror the CSS module's `.mock-paper` descendant rules
 * (.col, .doc-meta, .paper-title, .authors, .abstract, .claim .cid,
 * .claim .ctxt, .claim .meta .badge, .graph-mini .title). Because the
 * module hashes every class independently, every nested element
 * needs its own `styles[...]` reference.
 *
 * The inline `<svg>` citation graph uses `.nlbl` / `.nlbl-accent`
 * classes scoped via a local `<style>` block — same approach as
 * HeroTree.tsx — since those classes are not in the module CSS.
 */
import styles from "../landing.module.css";

export function ComprehensionMock(): React.JSX.Element {
  return (
    <div className={`${styles.mock} ${styles["mock-paper"]}`}>
      {/* ----- col 1 : paper preview ----- */}
      <div className={styles.col}>
        <div className={styles["doc-meta"]}>PAPER&nbsp;&nbsp;·&nbsp;&nbsp;PG&nbsp;3 / 17</div>
        <h4 className={styles["paper-title"]}>Recursive Language Models: Self-Refining Reasoning via Bounded Sub-Calls</h4>
        <div className={styles.authors}>Anonymous · arXiv:2410.04265 · v2</div>
        <p className={styles.abstract}>
          We introduce <mark>Recursive Language Models (RLM)</mark>, a paradigm in which a model
          decomposes a task, invokes itself on sub-tasks under a bounded depth budget,
          and aggregates partial results. On a suite of reasoning benchmarks, RLM agents
          <span className={styles["hl-rule"]}>improve absolute accuracy by 7–14 points</span> over a flat
          baseline at matched inference compute. Improvements are largest on tasks with
          <mark>verifiable intermediate state</mark>, suggesting RLM is best suited to
          domains where each sub-call can be checked against a rubric…
        </p>
      </div>

      {/* ----- col 2 : extracted claims ----- */}
      <div className={styles.col}>
        <div className={styles["doc-meta"]}>EXTRACTED&nbsp;CLAIMS&nbsp;&nbsp;·&nbsp;&nbsp;47</div>

        <div className={styles.claim}>
          <div className={styles.cid}>C-014 · §4.2 · TABLE 1, ROW 3</div>
          <div className={styles.ctxt}>RLM-7B beats flat-7B by <b>+9.2 pts</b> on GSM8K (pass@1).</div>
          <div className={styles.meta}>
            <span className={`${styles.badge} ${styles.ok}`}>QUANTITATIVE</span>
            <span className={styles.badge}>REPRODUCIBLE</span>
          </div>
        </div>

        <div className={styles.claim}>
          <div className={styles.cid}>C-018 · §4.3 · FIG 4</div>
          <div className={styles.ctxt}>Depth budget d=3 dominates d≤2 across 4 of 5 benchmarks.</div>
          <div className={styles.meta}>
            <span className={`${styles.badge} ${styles.ok}`}>QUANTITATIVE</span>
            <span className={styles.badge}>REPRODUCIBLE</span>
          </div>
        </div>

        <div className={styles.claim}>
          <div className={styles.cid}>C-022 · §5.1 · INLINE</div>
          <div className={styles.ctxt}>Sub-call cost grows sub-linearly with task length.</div>
          <div className={styles.meta}>
            <span className={styles.badge}>QUALITATIVE</span>
          </div>
        </div>

        <div className={styles.claim}>
          <div className={styles.cid}>C-031 · §6 · TABLE 3</div>
          <div className={styles.ctxt}>Verifier ablation drops accuracy by <b>−4.1 pts</b>.</div>
          <div className={styles.meta}>
            <span className={`${styles.badge} ${styles.ok}`}>QUANTITATIVE</span>
            <span className={styles.badge}>REPRODUCIBLE</span>
          </div>
        </div>
      </div>

      {/* ----- col 3 : citation graph + rubric task coverage ----- */}
      <div className={styles.col}>
        <div className={styles["doc-meta"]}>RUBRIC&nbsp;&nbsp;·&nbsp;&nbsp;INDUCED</div>
        <div className={styles["graph-mini"]}>
          <div className={styles.title}>CITATION GRAPH · §4.2 ↔ TABLE 1</div>
          <svg viewBox="0 0 280 150" width="100%" aria-hidden>
            <defs>
              <style>{`
                .nlbl { font-family: var(--font-jetbrains-mono), ui-monospace, monospace; font-size: 10px; fill: oklch(0.56 0.005 90); letter-spacing: 0.04em; }
                .nlbl-accent { fill: oklch(0.80 0.16 70); }
              `}</style>
            </defs>
            <line x1="40"  y1="40"  x2="140" y2="40"  stroke="oklch(0.28 0.004 270)"/>
            <line x1="40"  y1="40"  x2="140" y2="80"  stroke="oklch(0.80 0.16 70 / .8)"/>
            <line x1="40"  y1="40"  x2="140" y2="120" stroke="oklch(0.28 0.004 270)"/>
            <line x1="140" y1="80"  x2="240" y2="60"  stroke="oklch(0.80 0.16 70 / .8)"/>
            <line x1="140" y1="80"  x2="240" y2="110" stroke="oklch(0.28 0.004 270)"/>
            <circle cx="40"  cy="40"  r="5" fill="oklch(0.205 0.004 270)" stroke="oklch(0.56 0.005 90)"/>
            <circle cx="140" cy="40"  r="5" fill="oklch(0.205 0.004 270)" stroke="oklch(0.56 0.005 90)"/>
            <circle cx="140" cy="80"  r="5" fill="oklch(0.80 0.16 70)" stroke="oklch(0.80 0.16 70)"/>
            <circle cx="140" cy="120" r="5" fill="oklch(0.205 0.004 270)" stroke="oklch(0.56 0.005 90)"/>
            <circle cx="240" cy="60"  r="5" fill="oklch(0.205 0.004 270)" stroke="oklch(0.56 0.005 90)"/>
            <circle cx="240" cy="110" r="5" fill="oklch(0.205 0.004 270)" stroke="oklch(0.56 0.005 90)"/>
            <text x="50"  y="36"  className="nlbl">§4.2</text>
            <text x="150" y="36"  className="nlbl">TBL 1</text>
            <text x="150" y="76"  className="nlbl nlbl-accent">ROW 3</text>
            <text x="150" y="116" className="nlbl">FIG 4</text>
            <text x="250" y="56"  className="nlbl">C-014</text>
            <text x="250" y="106" className="nlbl">C-018</text>
          </svg>
        </div>
        <div className={styles["graph-mini"]} style={{ marginTop: "10px" }}>
          <div className={styles.title}>RUBRIC · TASK COVERAGE</div>
          <div
            style={{
              fontFamily: "var(--font-jetbrains-mono), ui-monospace, monospace",
              fontSize: "11px",
              color: "var(--landing-ink-2)",
              lineHeight: 1.85,
            }}
          >
            <div><span style={{ color: "var(--landing-ink-4)" }}>→</span> GSM8K · pass@1</div>
            <div><span style={{ color: "var(--landing-ink-4)" }}>→</span> MATH · pass@1</div>
            <div><span style={{ color: "var(--landing-ink-4)" }}>→</span> ARC-Challenge</div>
            <div><span style={{ color: "var(--landing-ink-4)" }}>→</span> HumanEval+</div>
            <div><span style={{ color: "var(--landing-ink-4)" }}>→</span> StrategyQA</div>
            <div style={{ color: "var(--landing-ink-4)", marginTop: "4px" }}>+ 4 ablations · §6</div>
          </div>
        </div>
      </div>
    </div>
  );
}
