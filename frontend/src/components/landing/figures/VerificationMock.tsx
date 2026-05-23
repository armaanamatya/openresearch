/**
 * §5.0 Verification mock — a three-row composition: a `rubric-head`
 * (paper title + meta), a 6-column / 6-row `rubric-table` of scored
 * claims (C-014, C-014b, C-017, C-018, C-022, C-031), and a 4-cell
 * `score-strip` summarising Claims verified, Reproduction score,
 * Drift band, and a fixed Artifact descriptor.
 *
 * Sourced verbatim from docs/design/landing-source/index.html lines
 * 1353-1443. Maps the `.mock-verify`, `.rubric-head`, `.rubric-table`,
 * `.cid`, `.cl`, `.tgt`, `.obs`, `.delta`, `.delta.pos`, `.delta.neg`,
 * `.judge`, `.pill.match`, `.score-strip`, `.cell`, `.lbl`, `.v`,
 * `.v.placeholder`, `.sub`, and `.tok` descendant selectors.
 *
 * Row C-018 carries `.delta` with NO sign modifier (the cell is the
 * `·` middot, not numeric). Row C-022 carries plain text "deviation"
 * in `.judge` — NO `.pill` wrapper, unlike the five `match` rows.
 * Token placeholders use JSX brace-escape (`{"{{TOKEN_NAME}}"}`) to
 * pass the literal `{{TOKEN_NAME}}` through to the rendered DOM.
 */
import styles from "../landing.module.css";

export function VerificationMock(): React.JSX.Element {
  return (
    <div className={`${styles.mock} ${styles["mock-verify"]}`}>
      <div className={styles["rubric-head"]}>
        <h5>Reproduction scorecard · arxiv:2410.04265</h5>
        <span className={styles.meta}>induced rubric · 18 cells · 5 seeds</span>
      </div>

      <table className={styles["rubric-table"]}>
        <thead>
          <tr>
            <th>id</th>
            <th>claim</th>
            <th>paper</th>
            <th>observed</th>
            <th>Δ</th>
            <th>judge</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className={styles.cid}>C-014</td>
            <td className={styles.cl}>GSM8K · pass@1 · RLM-7B</td>
            <td className={styles.tgt}>0.631</td>
            <td className={styles.obs}>0.627 ± 0.004</td>
            <td className={`${styles.delta} ${styles.neg}`}>−0.004</td>
            <td className={styles.judge}><span className={`${styles.pill} ${styles.match}`}>match</span></td>
          </tr>
          <tr>
            <td className={styles.cid}>C-014b</td>
            <td className={styles.cl}>GSM8K · gain vs. flat-7B</td>
            <td className={styles.tgt}>+9.2 pts</td>
            <td className={styles.obs}>+8.9 pts</td>
            <td className={`${styles.delta} ${styles.neg}`}>−0.3</td>
            <td className={styles.judge}><span className={`${styles.pill} ${styles.match}`}>match</span></td>
          </tr>
          <tr>
            <td className={styles.cid}>C-017</td>
            <td className={styles.cl}>MATH · pass@1 · RLM-7B</td>
            <td className={styles.tgt}>0.418</td>
            <td className={styles.obs}>0.412 ± 0.006</td>
            <td className={`${styles.delta} ${styles.neg}`}>−0.006</td>
            <td className={styles.judge}><span className={`${styles.pill} ${styles.match}`}>match</span></td>
          </tr>
          <tr>
            <td className={styles.cid}>C-018</td>
            <td className={styles.cl}>depth d=3 dominates d≤2</td>
            <td className={styles.tgt}>4 of 5</td>
            <td className={styles.obs}>4 of 5</td>
            <td className={styles.delta}>·</td>
            <td className={styles.judge}><span className={`${styles.pill} ${styles.match}`}>match</span></td>
          </tr>
          <tr>
            <td className={styles.cid}>C-022</td>
            <td className={styles.cl}>HumanEval+ · pass@1</td>
            <td className={styles.tgt}>0.604</td>
            <td className={styles.obs}>0.586 ± 0.009</td>
            <td className={`${styles.delta} ${styles.neg}`}>−0.018</td>
            <td className={styles.judge}>deviation</td>
          </tr>
          <tr>
            <td className={styles.cid}>C-031</td>
            <td className={styles.cl}>verifier ablation drop</td>
            <td className={styles.tgt}>−4.1 pts</td>
            <td className={styles.obs}>−3.8 pts</td>
            <td className={`${styles.delta} ${styles.pos}`}>+0.3</td>
            <td className={styles.judge}><span className={`${styles.pill} ${styles.match}`}>match</span></td>
          </tr>
        </tbody>
      </table>

      <div className={styles["score-strip"]}>
        <div className={styles.cell}>
          <div className={styles.lbl}>Claims verified</div>
          <div className={`${styles.v} ${styles.placeholder}`}>
            <span className={styles.tok}>{"{{N_VERIFIED}}"}</span> / <span className={styles.tok}>{"{{N_CLAIMS}}"}</span>
          </div>
          <div className={styles.sub}>scorecard cells · 5 seeds</div>
        </div>
        <div className={styles.cell}>
          <div className={styles.lbl}>Reproduction score</div>
          <div className={`${styles.v} ${styles.placeholder}`}>
            <span className={styles.tok}>{"{{REPRODUCTION_SCORE}}"}</span>
          </div>
          <div className={styles.sub}>PaperBench · v0 protocol</div>
        </div>
        <div className={styles.cell}>
          <div className={styles.lbl}>Drift band</div>
          <div className={`${styles.v} ${styles.placeholder}`}>
            ± <span className={styles.tok}>{"{{DRIFT_BAND}}"}</span>
          </div>
          <div className={styles.sub}>median |Δ| across cells</div>
        </div>
        <div className={styles.cell}>
          <div className={styles.lbl}>Artifact</div>
          <div className={styles.v} style={{ fontSize: "14px", letterSpacing: 0 }}>report.md · run.json</div>
          <div className={styles.sub}>signed · seed-pinned · reproducible</div>
        </div>
      </div>
    </div>
  );
}
