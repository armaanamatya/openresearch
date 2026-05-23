import Link from "next/link";
import styles from "./landing.module.css";
import "./landing.global.css";
import { HeroTree } from "./figures/HeroTree";
import { NavScrollMount } from "./client-bits";

const GITHUB_URL = "https://github.com/armaanamatya/openresearch";

export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      <NavScrollMount targetId="nav" scrolledClass={styles.scrolled} />
      <Nav />
      <Hero />
      {/* Tasks 4-7 add sections + footer below */}
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

function GitHubIcon(): React.JSX.Element {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38v-1.34c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.72-.49.06-.48.06-.48.8.06 1.22.82 1.22.82.71 1.21 1.87.86 2.33.66.07-.52.28-.86.5-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 014 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}
