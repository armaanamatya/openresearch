import Link from "next/link";
import styles from "./landing.module.css";
import "./landing.global.css";
import { HeroTree } from "./figures/HeroTree";
import { ComprehensionMock } from "./figures/ComprehensionMock";
import { EnvironmentMock } from "./figures/EnvironmentMock";
import { ImplementationMock } from "./figures/ImplementationMock";
import { ExperimentsMock } from "./figures/ExperimentsMock";
import { VerificationMock } from "./figures/VerificationMock";
import { NavScrollMount, RevealMount } from "./client-bits";

const GITHUB_URL = "https://github.com/armaanamatya/openresearch";

export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      <NavScrollMount targetId="nav" scrolledClass={styles.scrolled} />
      <RevealMount />
      <Nav />
      <Hero />
      <ComprehensionSection />
      <EnvironmentSection />
      <ImplementationSection />
      <ExperimentsSection />
      <VerificationSection />
      <BenchmarkSection />
      <CTAFooter />
      <Footer />
    </div>
  );
}

function Nav(): React.JSX.Element {
  return (
    <nav id="nav" className={styles.nav}>
      <div className={`${styles.wrap} ${styles["nav-inner"]}`}>
        <Link href="/" className={styles.brand}>
          <span className={styles["brand-mark"]} aria-hidden />
          <span className={styles["brand-name"]}>
            ReproLab<span className={styles.dot}>.</span>
          </span>
        </Link>
        <div className={styles["nav-links"]}>
          <a href="#pipeline">How it works</a>
          <a href="#benchmarks">Benchmarks</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a>
        </div>
        <Link href="/lab" className={styles["nav-cta"]}>
          <span>Open lab</span>
          <span className={styles.ar}>→</span>
        </Link>
      </div>
    </nav>
  );
}

function Hero(): React.JSX.Element {
  return (
    <header className={styles.hero} id="top">
      <div className={styles.wrap}>

        <div className={`${styles["hero-eyebrow"]} ${styles.mono} ${styles.reveal}`}>
          <span className={styles.pip} aria-hidden />
          <span>RLM&nbsp;&nbsp;·&nbsp;&nbsp;§ 0.0 — Reproducibility, automated.</span>
        </div>

        <div className={styles["hero-grid"]}>
          <h1 className={`${styles.display} ${styles.reveal}`}>
            Reproduce any ML paper, <span className={styles["accent-soft"]}>end&nbsp;to&nbsp;end.</span>
          </h1>
          <div className={styles.reveal}>
            <p className={styles.lede} style={{ margin: "0 0 24px" }}>
              ReproLab is a paper-reproduction agent built on the Recursive Language Model paradigm.
              It reads a paper, builds the environment, implements the method, runs the experiments,
              and grades itself against the paper&apos;s own claims.
            </p>
            <dl className={styles["hero-meta"]}>
              <div>
                <dt>Paradigm</dt>
                <dd>Recursive Language Models (RLM)</dd>
              </div>
              <div>
                <dt>Stages</dt>
                <dd>§ 1.0&nbsp;→&nbsp;5.0&nbsp;&nbsp;·&nbsp;&nbsp;closed-loop</dd>
              </div>
              <div>
                <dt>Modality</dt>
                <dd>Code, configs, container, claims</dd>
              </div>
              <div>
                <dt>Status</dt>
                <dd>Research preview · <span className={styles.tok}>{"{{REPRODUCTION_SCORE}}"}</span></dd>
              </div>
            </dl>
          </div>
        </div>

        <div className={`${styles["hero-actions"]} ${styles.reveal}`}>
          <Link href="/lab" className={`${styles.btn} ${styles["btn-primary"]}`}>
            <span>Open the lab</span>
            <span aria-hidden>→</span>
          </Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={`${styles.btn} ${styles["btn-ghost"]}`}>
            <GitHubIcon />
            <span>View on GitHub</span>
          </a>
        </div>

        <figure className={`${styles["hero-figure"]} ${styles.reveal}`}>
          <span className={styles["fig-label"]}>FIG&nbsp;§ 0.1&nbsp;&nbsp;—&nbsp;&nbsp;Lab UI / exploration tree</span>
          <div className={styles.frame}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.dots}>
                <span className={styles.dot} />
                <span className={styles.dot} />
                <span className={styles.dot} />
              </span>
              <span className={styles.crumbs}>
                lab&nbsp;/&nbsp;<b>arxiv:2410.04265 — Recursive Language Models</b>
              </span>
              <span className={styles.right}>run · rlm-7e3a · 02:14:08</span>
            </div>
            <div style={{ padding: "22px 24px 28px" }}>
              <HeroTree />
            </div>
          </div>
        </figure>
      </div>
    </header>
  );
}

function ComprehensionSection(): React.JSX.Element {
  return (
    <section className={styles.spec} id="pipeline">
      <div className={styles.wrap}>
        <div className={`${styles["spec-header"]} ${styles.reveal}`}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>1.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Comprehension</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>stage 01 / 05</span>
            </div>
            <h2 className={styles.head}>Read the paper like an author would.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              The agent ingests the PDF and source repo, extracts every quantitative claim,
              and builds a citation graph linking each claim to the table, figure, or
              equation that supports it.
            </p>
            <div className={styles["spec-substeps"]}>
              <span><span className={styles.num}>§ 1.1</span><span>Parsing</span></span>
              <span><span className={styles.num}>§ 1.2</span><span>Claim extraction</span></span>
              <span><span className={styles.num}>§ 1.3</span><span>Citation graph</span></span>
              <span><span className={styles.num}>§ 1.4</span><span>Rubric induction</span></span>
            </div>
          </div>
        </div>

        <figure className={`${styles["spec-visual"]} ${styles.reveal}`}>
          <span className={styles["fig-label"]}>FIG&nbsp;§ 1.2&nbsp;&nbsp;—&nbsp;&nbsp;Claim extraction · rubric induction</span>
          <div className={`${styles.frame} ${styles["frame-bleed-r"]}`}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.crumbs}>comprehension&nbsp;/&nbsp;<b>arxiv:2410.04265</b></span>
              <span className={styles.right}>47 claims · 23 figures · 9 tables</span>
            </div>
            <ComprehensionMock />
          </div>
        </figure>
      </div>
    </section>
  );
}

function EnvironmentSection(): React.JSX.Element {
  return (
    <section className={styles.spec}>
      <div className={styles.wrap}>
        <div className={`${styles["spec-header"]} ${styles.reveal}`}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>2.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Environment</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>stage 02 / 05</span>
            </div>
            <h2 className={styles.head}>Stand up exactly the machine the paper ran on.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              Pinned CUDA, pinned wheels, pinned data hashes. A reproducible container is
              built, verified against the paper&apos;s dependencies, and locked before a
              single training step runs.
            </p>
            <div className={styles["spec-substeps"]}>
              <span><span className={styles.num}>§ 2.1</span><span>Toolchain</span></span>
              <span><span className={styles.num}>§ 2.2</span><span>Lockfile</span></span>
              <span><span className={styles.num}>§ 2.3</span><span>Dataset hashes</span></span>
              <span><span className={styles.num}>§ 2.4</span><span>GPU provisioning</span></span>
            </div>
          </div>
        </div>

        <figure className={`${styles["spec-visual"]} ${styles.reveal}`}>
          <span className={styles["fig-label"]}>FIG&nbsp;§ 2.2&nbsp;&nbsp;—&nbsp;&nbsp;Lockfile · GPU provisioning</span>
          <div className={styles.frame}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.crumbs}>env&nbsp;/&nbsp;<b>rlm-7e3a / container</b></span>
              <span className={styles.right}>image · sha256:9a4f…02e1</span>
            </div>
            <EnvironmentMock />
          </div>
        </figure>
      </div>
    </section>
  );
}

function ImplementationSection(): React.JSX.Element {
  return (
    <section className={styles.spec}>
      <div className={styles.wrap}>
        <div className={`${styles["spec-header"]} ${styles.reveal}`}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>3.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Implementation</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>stage 03 / 05</span>
            </div>
            <h2 className={styles.head}>Write the method, then improve it.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              A baseline implementation is generated from the paper&apos;s pseudocode and
              figures. The RLM then explores bounded variations against the induced
              rubric — every edit attributed, every diff reviewable.
            </p>
            <div className={styles["spec-substeps"]}>
              <span><span className={styles.num}>§ 3.1</span><span>Baseline</span></span>
              <span><span className={styles.num}>§ 3.2</span><span>Improvement exploration</span></span>
              <span><span className={styles.num}>§ 3.3</span><span>Diff review</span></span>
              <span><span className={styles.num}>§ 3.4</span><span>Attribution</span></span>
            </div>
          </div>
        </div>

        <figure className={`${styles["spec-visual"]} ${styles.reveal}`}>
          <span className={styles["fig-label"]}>FIG&nbsp;§ 3.2&nbsp;&nbsp;—&nbsp;&nbsp;Baseline vs. improvement · diff view</span>
          <div className={`${styles.frame} ${styles["frame-bleed-l"]}`}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.crumbs}>implementation&nbsp;/&nbsp;<b>recurse.py</b></span>
              <span className={styles.right}>branch 3.2a · +18 / −6</span>
            </div>
            <ImplementationMock />
          </div>
        </figure>
      </div>
    </section>
  );
}

function ExperimentsSection(): React.JSX.Element {
  return (
    <section className={styles.spec}>
      <div className={styles.wrap}>
        <div className={`${styles["spec-header"]} ${styles.reveal}`}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>4.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Experiments</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>stage 04 / 05</span>
            </div>
            <h2 className={styles.head}>Run the table the paper claims.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              Every row, every column, every seed. Experiments are scheduled across the
              rubric, results stream into the lab UI, and intermediate state is logged
              so a failed run is debuggable, not opaque.
            </p>
            <div className={styles["spec-substeps"]}>
              <span><span className={styles.num}>§ 4.1</span><span>Run scheduler</span></span>
              <span><span className={styles.num}>§ 4.2</span><span>Seed sweeps</span></span>
              <span><span className={styles.num}>§ 4.3</span><span>Live metrics</span></span>
              <span><span className={styles.num}>§ 4.4</span><span>Run registry</span></span>
            </div>
          </div>
        </div>

        <figure className={`${styles["spec-visual"]} ${styles.reveal}`}>
          <span className={styles["fig-label"]}>FIG&nbsp;§ 4.3&nbsp;&nbsp;—&nbsp;&nbsp;Live metrics · run registry</span>
          <div className={styles.frame}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.crumbs}>experiments&nbsp;/&nbsp;<b>GSM8K · pass@1</b></span>
              <span className={styles.right}>5 seeds · 32 runs · 4h 12m</span>
            </div>
            <ExperimentsMock />
          </div>
        </figure>
      </div>
    </section>
  );
}

function VerificationSection(): React.JSX.Element {
  return (
    <section className={styles.spec}>
      <div className={styles.wrap}>
        <div className={`${styles["spec-header"]} ${styles.reveal}`}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>5.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Verification</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>stage 05 / 05</span>
            </div>
            <h2 className={styles.head}>Score the result against the paper&apos;s own claims.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              Each observed metric is graded against the rubric induced in §1.4 —
              match, deviation, or contradiction — with the supporting table cell
              and confidence interval surfaced for every cell.
            </p>
            <div className={styles["spec-substeps"]}>
              <span><span className={styles.num}>§ 5.1</span><span>Rubric scoring</span></span>
              <span><span className={styles.num}>§ 5.2</span><span>Confidence intervals</span></span>
              <span><span className={styles.num}>§ 5.3</span><span>Report artifact</span></span>
              <span><span className={styles.num}>§ 5.4</span><span>Audit trail</span></span>
            </div>
          </div>
        </div>

        <figure className={`${styles["spec-visual"]} ${styles.reveal}`}>
          <span className={styles["fig-label"]}>FIG&nbsp;§ 5.1&nbsp;&nbsp;—&nbsp;&nbsp;Reproduction scorecard</span>
          <div className={`${styles.frame} ${styles["frame-bleed-r"]}`}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.crumbs}>verification&nbsp;/&nbsp;<b>scorecard · rlm-7e3a</b></span>
              <span className={styles.right}>artifact · report.md · 14 KB</span>
            </div>
            <VerificationMock />
          </div>
        </figure>
      </div>
    </section>
  );
}

function BenchmarkSection(): React.JSX.Element {
  return (
    <section className={styles.bench} id="benchmarks">
      <div className={styles.wrap}>
        <div className={`${styles.row} ${styles.reveal}`}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>6.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Benchmarks</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>measured · not promised</span>
            </div>
            <h2 className={styles.head}>Numbers go here once real runs say so.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              ReproLab reports a reproduction score per paper on the PaperBench v0
              protocol. Every figure on this page is a placeholder until a sealed,
              seed-pinned run confirms it. We will not ship a number we cannot rerun.
            </p>
          </div>
        </div>

        <div className={`${styles["bench-numbers"]} ${styles.reveal}`}>
          <div className={styles["bench-cell"]}>
            <span className={styles["pl-note"]}>placeholder</span>
            <div className={styles.lbl}>PaperBench v0 · score</div>
            <div className={`${styles.v} ${styles.placeholder}`}>
              <span className={styles.tok}>{"{{REPRODUCTION_SCORE}}"}</span>
              <sup>↗ target {"{{TARGET_SCORE}}"}</sup>
            </div>
            <div className={styles.sub}>Median per-cell rubric match across the suite.</div>
          </div>
          <div className={styles["bench-cell"]}>
            <span className={styles["pl-note"]}>placeholder</span>
            <div className={styles.lbl}>Papers reproduced</div>
            <div className={`${styles.v} ${styles.placeholder}`}>
              <span className={styles.tok}>{"{{N_PAPERS}}"}</span>
            </div>
            <div className={styles.sub}>End-to-end. Comprehension through sealed scorecard.</div>
          </div>
          <div className={styles["bench-cell"]}>
            <span className={styles["pl-note"]}>placeholder</span>
            <div className={styles.lbl}>Median drift</div>
            <div className={`${styles.v} ${styles.placeholder}`}>
              ± <span className={styles.tok}>{"{{MEDIAN_DRIFT}}"}</span>
            </div>
            <div className={styles.sub}>Absolute deviation between observed and paper-reported metrics.</div>
          </div>
          <div className={styles["bench-cell"]}>
            <span className={styles["pl-note"]}>placeholder</span>
            <div className={styles.lbl}>Wall-clock per paper</div>
            <div className={`${styles.v} ${styles.placeholder}`}>
              <span className={styles.tok}>{"{{HOURS_PER_PAPER}}"}</span>
            </div>
            <div className={styles.sub}>Median, includes environment build and verification.</div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTAFooter(): React.JSX.Element {
  return (
    <section className={styles["cta-foot"]} id="github">
      <div className={`${styles.wrap} ${styles.reveal}`}>
        <div className={styles["h-eyebrow"]} style={{ marginBottom: 22 }}>§ 7.0 — Try it</div>
        <h2>An end-to-end reproduction,<br />in one command.</h2>
        <p className={styles.lede}>
          Point ReproLab at an arXiv ID. Get back a sealed environment, an implementation,
          a scorecard, and an audit trail. If a claim doesn&apos;t reproduce, you&apos;ll see exactly which one.
        </p>
        <div className={styles.actions}>
          <Link href="/lab" className={`${styles.btn} ${styles["btn-primary"]}`}>
            <span>Open the lab</span>
            <span aria-hidden>→</span>
          </Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={`${styles.btn} ${styles["btn-ghost"]}`}>
            <GitHubIcon />
            <span>github.com/armaanamatya/openresearch</span>
          </a>
        </div>
      </div>
    </section>
  );
}

function Footer(): React.JSX.Element {
  return (
    <footer className={styles.foot}>
      <div className={`${styles.wrap} ${styles["foot-inner"]}`}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className={styles["brand-mark"]} style={{ width: 18, height: 18 }} aria-hidden />
          <span>ReproLab · 2026</span>
        </div>
        <div className={styles.links}>
          <a href="#pipeline">How it works</a>
          <a href="#benchmarks">Benchmarks</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a>
        </div>
        <div className={styles.mono}>§ end · v0.1.0</div>
      </div>
    </footer>
  );
}

function GitHubIcon(): React.JSX.Element {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38v-1.34c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.72-.49.06-.48.06-.48.8.06 1.22.82 1.22.82.71 1.21 1.87.86 2.33.66.07-.52.28-.86.5-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 014 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}
