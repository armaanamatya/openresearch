/**
 * §2.0 Environment mock — two-column layout: a file-tree + lockfile
 * on the left, and four spec cards + a build-log stream on the right.
 *
 * Sourced from docs/design/landing-source/index.html lines 1023-1081.
 * The CSS module's `.mock-env`, `.tree`, `.lockfile`, `.env-card`,
 * `.cards-row`, `.log` blocks all use descendant selectors, so every
 * nested element needs its own `styles[...]` reference.
 *
 * The tree's box-drawing characters (├─, │, └─) and non-breaking
 * spaces are preserved verbatim from the source HTML.
 */
import styles from "../landing.module.css";

export function EnvironmentMock(): React.JSX.Element {
  return (
    <div className={`${styles.mock} ${styles["mock-env"]}`}>
      {/* ----- left : tree + lockfile ----- */}
      <div className={styles["env-left"]}>
        <div className={styles.tree} aria-hidden>
          <span className={styles.root}>rlm-7e3a/</span><br />
          <span className={styles.dim}>├─</span> Dockerfile&nbsp;<span className={`${styles.ok} ${styles.pin}`}>pinned</span><br />
          <span className={styles.dim}>├─</span> environment.lock<br />
          <span className={styles.dim}>├─</span> data/<br />
          <span className={styles.dim}>│&nbsp;&nbsp;├─</span> gsm8k.parquet&nbsp;<span className={styles.dim}>·</span> <span className={styles.pin}>sha 4f8a…</span><br />
          <span className={styles.dim}>│&nbsp;&nbsp;├─</span> math.parquet&nbsp;&nbsp;<span className={styles.dim}>·</span> <span className={styles.pin}>sha c102…</span><br />
          <span className={styles.dim}>│&nbsp;&nbsp;└─</span> humaneval.json <span className={styles.dim}>·</span> <span className={styles.pin}>sha 71e4…</span><br />
          <span className={styles.dim}>├─</span> src/<br />
          <span className={styles.dim}>│&nbsp;&nbsp;├─</span> model.py<br />
          <span className={styles.dim}>│&nbsp;&nbsp;├─</span> recurse.py<br />
          <span className={styles.dim}>│&nbsp;&nbsp;└─</span> verify.py<br />
          <span className={styles.dim}>└─</span> runs/&nbsp;<span className={styles.dim}>(12)</span>
        </div>
        <div className={styles.lockfile}>
          <div><span className={styles.k}>cuda</span>&nbsp;&nbsp;<span className={styles.v}>12.4.1</span></div>
          <div><span className={styles.k}>python</span>&nbsp;<span className={styles.v}>3.11.9</span></div>
          <div><span className={styles.k}>torch</span>&nbsp;&nbsp;<span className={styles.v}>2.4.0+cu124</span></div>
          <div><span className={styles.k}>flash-attn</span>&nbsp;<span className={styles.v}>2.6.3</span></div>
          <div><span className={styles.k}>transformers</span>&nbsp;<span className={styles.v}>4.44.2</span></div>
          <div><span className={styles.k}>datasets</span>&nbsp;<span className={styles.v}>2.20.0</span></div>
        </div>
      </div>

      {/* ----- right : env cards + log ----- */}
      <div className={styles["env-right"]}>
        <div className={styles["cards-row"]}>
          <div className={styles["env-card"]}>
            <div className={styles.lbl}>GPU</div>
            <div className={styles.val}>H100 · 80GB</div>
            <div className={styles.sub}>×4 · interconnect NVLink</div>
          </div>
          <div className={styles["env-card"]}>
            <div className={styles.lbl}>Build</div>
            <div className={styles.val}>2m 47s</div>
            <div className={styles.sub}>cache hit · 11 / 14 layers</div>
          </div>
          <div className={styles["env-card"]}>
            <div className={styles.lbl}>Image</div>
            <div className={styles.val}>4.2 GB</div>
            <div className={styles.sub}>reproducible · bit-identical rebuild</div>
          </div>
          <div className={styles["env-card"]}>
            <div className={styles.lbl}>Dataset</div>
            <div className={styles.val}>3 / 3</div>
            <div className={styles.sub}>hashes match paper appendix</div>
          </div>
        </div>
        <div className={styles.log}>
          <div><span className={styles.ts}>[00:00.12]</span> pulling base &nbsp; nvidia/cuda:12.4.1-cudnn-runtime</div>
          <div><span className={styles.ts}>[00:14.81]</span> installing pinned wheels …</div>
          <div><span className={styles.ts}>[01:09.20]</span> verifying flash-attn @ 2.6.3 &nbsp; <span className={styles.ok}>ok</span></div>
          <div><span className={styles.ts}>[02:01.55]</span> hashing data/ &nbsp; gsm8k.parquet &nbsp; <span className={styles.ok}>match</span></div>
          <div><span className={styles.ts}>[02:18.30]</span> hashing data/ &nbsp; math.parquet &nbsp;&nbsp; <span className={styles.ok}>match</span></div>
          <div><span className={styles.ts}>[02:22.04]</span> hashing data/ &nbsp; humaneval.json <span className={styles.ok}>match</span></div>
          <div><span className={styles.ts}>[02:47.18]</span> <span className={styles.em}>image sealed</span> · sha256:9a4f…02e1</div>
        </div>
      </div>
    </div>
  );
}
