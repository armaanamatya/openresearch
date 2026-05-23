/**
 * §3.0 Implementation mock — two-column layout: a file-tree sidebar
 * (SRC + RUNS sections) on the left, and a tabbed code panel on the
 * right showing the recurse.py diff between baseline and branch 3.2a.
 *
 * Sourced from docs/design/landing-source/index.html lines 1118-1156.
 * Mirrors the `.mock-impl`, `.files-col`, `.files-h`, `.f`, `.f.active`,
 * `.code-col`, `.code-tabs`, `.tab`, `.code`, `.ln`, `.kw`, `.fn`,
 * `.st`, `.cm`, `.add`, `.del` descendant selectors in the CSS module.
 *
 * JSX whitespace caveat: the Python code inside `<pre>` requires
 * inter-line newlines via `{"\n"}` (JSX strips newlines adjacent to
 * tags) and preserved indentation via explicit string literals where
 * runs of spaces appear at line starts. `.add` / `.del` are
 * `display: block` so they self-terminate without trailing `\n`.
 */
import styles from "../landing.module.css";

export function ImplementationMock(): React.JSX.Element {
  return (
    <div className={`${styles.mock} ${styles["mock-impl"]}`}>
      {/* ----- left : SRC tree + RUNS list ----- */}
      <div className={styles["files-col"]}>
        <div className={styles["files-h"]}>SRC · rlm-7e3a</div>
        <div className={styles.f}><span className={styles.ic}>›</span> model.py</div>
        <div className={`${styles.f} ${styles.active}`}><span className={styles.ic}>›</span> recurse.py</div>
        <div className={styles.f}><span className={styles.ic}>›</span> verify.py</div>
        <div className={styles.f}><span className={styles.ic}>›</span> rubric.py</div>
        <div className={styles.f}><span className={styles.ic}>›</span> data.py</div>
        <div className={styles["files-h"]} style={{ marginTop: "18px" }}>RUNS</div>
        <div className={styles.f}><span className={styles.ic}>●</span> baseline</div>
        <div className={styles.f}><span className={styles.ic}>●</span> 3.2a · lr×2</div>
        <div className={styles.f}><span className={styles.ic}>●</span> 3.2b · bs 256</div>
        <div className={styles.f}><span className={styles.ic}>●</span> 3.2c · lion</div>
        <div className={styles.f}><span className={styles.ic}>●</span> 3.2d · cosine</div>
      </div>

      {/* ----- right : tabbed Python code with diff ----- */}
      <div className={styles["code-col"]}>
        <div className={styles["code-tabs"]}>
          <div className={`${styles.tab} ${styles.active}`}>recurse.py</div>
          <div className={styles.tab}>verify.py</div>
          <div className={styles.tab}>rubric.py</div>
          <div className={styles.tab} style={{ color: "var(--landing-ink-4)" }}>+ open</div>
        </div>
        <pre className={styles.code}>
<span className={styles.ln}>{" 1"}</span> <span className={styles.kw}>def</span> <span className={styles.fn}>recurse</span>(task, depth, budget):{"\n"}
<span className={styles.ln}>{" 2"}</span>     <span className={styles.kw}>if</span> depth {">"}= budget.max_depth:{"\n"}
<span className={styles.ln}>{" 3"}</span>         <span className={styles.kw}>return</span> <span className={styles.fn}>flat_solve</span>(task){"\n"}
<span className={styles.ln}>{" 4"}</span>     plan = <span className={styles.fn}>decompose</span>(task)              <span className={styles.cm}># §3 fig 2</span>{"\n"}
<span className={styles.ln}>{" 5"}</span>     subs = []{"\n"}
<span className={styles.ln}>{" 6"}</span>     <span className={styles.kw}>for</span> sub <span className={styles.kw}>in</span> plan.children:{"\n"}
<span className={styles.del}><span className={styles.ln}>{" 7"}</span>         partial = <span className={styles.fn}>recurse</span>(sub, depth+1, budget)</span>
<span className={styles.add}><span className={styles.ln}>{" 7"}</span>         partial = <span className={styles.fn}>recurse</span>(sub, depth+1, budget.tighten(depth))</span>
<span className={styles.add}><span className={styles.ln}>{" 8"}</span>         <span className={styles.kw}>if</span> <span className={styles.kw}>not</span> <span className={styles.fn}>verify_partial</span>(sub, partial):   <span className={styles.cm}># §6 ablation</span></span>
<span className={styles.add}><span className={styles.ln}>{" 9"}</span>             partial = <span className={styles.fn}>recurse</span>(sub, depth+1, budget.retry())</span>
<span className={styles.ln}>10</span>         subs.append(partial){"\n"}
<span className={styles.ln}>11</span>     <span className={styles.kw}>return</span> <span className={styles.fn}>aggregate</span>(plan, subs, rubric=<span className={styles.st}>&quot;task&quot;</span>){"\n"}
<span className={styles.ln}>12</span>{"\n"}
<span className={styles.ln}>13</span> <span className={styles.cm}># attribution: paper §3.1 alg 1 · diff 3.2a · 2 of 5 seeds verified</span>
        </pre>
      </div>
    </div>
  );
}
