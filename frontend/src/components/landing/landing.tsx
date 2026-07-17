import Link from "next/link";
import styles from "./landing.module.css";
import "./landing.global.css";
import { HeroTree } from "./figures/HeroTree";
import { ComprehensionMock } from "./figures/ComprehensionMock";
import { EnvironmentMock } from "./figures/EnvironmentMock";
import { ImplementationMock } from "./figures/ImplementationMock";
import { ExperimentsMock } from "./figures/ExperimentsMock";
import { VerificationMock } from "./figures/VerificationMock";
import { NavScrollMount } from "./client-bits";

const GITHUB_URL = "https://github.com/armaanamatya/openresearch";

const NAV_LINKS: ReadonlyArray<{ href: string; label: string; external?: boolean }> = [
  { href: "#different", label: "Why it's different" },
  { href: "#pipeline", label: "How it works" },
  { href: "#trust", label: "Open source" }
];

export function LandingPage(): React.JSX.Element {
  return (
    <div className={styles.shell}>
      <NavScrollMount targetId="nav" scrolledClass={styles.scrolled} />
      <Nav />
      <Hero />
      <DifferentSection />
      <PipelineSection />
      <TrustSection />
      <CTAFooter />
      <Footer />
    </div>
  );
}

/* ─────────────────────────────  NAV  ───────────────────────────── */

function Nav(): React.JSX.Element {
  return (
    <nav id="nav" className={styles.nav}>
      <div className={`${styles.wrap} ${styles["nav-inner"]}`}>
        <Link href="/" className={styles.brand}>
          <span className={styles["brand-mark"]} aria-hidden />
          <span className={styles["brand-name"]}>
            OpenResearch<span className={styles.dot}>.</span>
          </span>
        </Link>

        <div className={styles["nav-links"]}>
          {NAV_LINKS.map((l) => (
            <a key={l.href} href={l.href}>
              {l.label}
            </a>
          ))}
          <Link href="/leaderboard">Runs</Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </div>

        <div className={styles["nav-right"]}>
          <Link href="/lab" className={styles["nav-cta"]}>
            <span>Open lab</span>
            <span className={styles.ar} aria-hidden>
              →
            </span>
          </Link>

          {/* CSS-only mobile menu — desktop nav-links hide ≤720px, this shows */}
          <details className={styles["nav-mobile"]}>
            <summary className={styles["nav-burger"]} aria-label="Menu">
              <span aria-hidden />
              <span aria-hidden />
              <span aria-hidden />
            </summary>
            <div className={styles["nav-menu"]}>
              {NAV_LINKS.map((l) => (
                <a key={l.href} href={l.href}>
                  {l.label}
                </a>
              ))}
              <Link href="/leaderboard">Runs</Link>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer">
                GitHub
              </a>
            </div>
          </details>
        </div>
      </div>
    </nav>
  );
}

/* ─────────────────────────────  HERO  ──────────────────────────── */

const DELIVERABLES = ["Cited brief", "Source map", "Contradiction log", "Reproducible code"] as const;

function Hero(): React.JSX.Element {
  return (
    <header className={styles.hero} id="top">
      <div className={styles.wrap}>
        <div className={`${styles["hero-eyebrow"]} ${styles.mono}`}>
          <span className={styles.pip} aria-hidden />
          <span>Verified research — not summaries.</span>
        </div>

        <div className={styles["hero-grid"]}>
          <h1 className={styles.display}>
            Other tools summarize papers.{" "}
            <span className={styles.accent}>We re&#8209;run&nbsp;them.</span>
          </h1>
          <div>
            <p className={styles.lede} style={{ margin: "0 0 26px" }}>
              OpenResearch reads the literature, rebuilds the experiments, and runs them — then hands
              you a cited brief where every number is checked against a result it produced, and every
              contradiction is flagged.
            </p>
            <div className={styles["hero-deliverables"]}>
              <span className={styles["deliv-label"]}>What comes back</span>
              <div className={styles["deliv-chips"]}>
                {DELIVERABLES.map((d) => (
                  <span key={d} className={styles.chip}>
                    {d}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className={styles["hero-actions"]}>
          <Link href="/lab" className={`${styles.btn} ${styles["btn-primary"]}`}>
            <span>Open the lab</span>
            <span aria-hidden>→</span>
          </Link>
          <Link href="/leaderboard" className={`${styles.btn} ${styles["btn-ghost"]}`}>
            <span>Browse live runs</span>
            <span aria-hidden>→</span>
          </Link>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className={`${styles.btn} ${styles["btn-ghost"]}`}
          >
            <GitHubIcon />
            <span>View source</span>
          </a>
        </div>

        <figure className={styles["hero-figure"]}>
          <span className={styles["fig-label"]}>FIG 0.1 — The lab, mid-run (illustrative)</span>
          <div className={styles.frame}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.dots}>
                <span className={styles.dot} />
                <span className={styles.dot} />
                <span className={styles.dot} />
              </span>
              <span className={styles.crumbs}>
                lab&nbsp;/&nbsp;<b>reproduction · exploration tree</b>
              </span>
              <span className={styles.right}>live run</span>
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

/* ───────────────────────  WHY IT'S DIFFERENT  ──────────────────── */

const COMPARE_ROWS: ReadonlyArray<{ them: string; us: string }> = [
  { them: "Summarize the abstract", us: "Rebuild and run the method" },
  { them: "Quote the reported number", us: "Re-run it and measure the real one" },
  { them: "Tell you what the paper says", us: "Flag where reported ≠ reproduced" },
  { them: "Ask you to trust the text", us: "Link every claim to an executed run" }
];

function DifferentSection(): React.JSX.Element {
  return (
    <section className={styles.different} id="different">
      <div className={styles.wrap}>
        <div className={styles["section-lead"]}>
          <span className={styles["h-eyebrow"]}>Why it&apos;s different</span>
          <h2 className={styles.headline}>
            Summarizers read the paper.
            <br />
            <span className={styles.accent}>OpenResearch runs it.</span>
          </h2>
          <p className={styles.lede}>
            Elicit, Consensus, Perplexity and the rest tell you what a paper claims. None of them
            check whether the claim holds. OpenResearch rebuilds the experiment, executes it, and
            grades every number against what actually came out.
          </p>
        </div>

        <div className={styles.compare}>
          <div className={`${styles["compare-col"]} ${styles.them}`}>
            <div className={styles["compare-head"]}>
              <span className={styles["compare-tag"]}>Read-only research tools</span>
            </div>
            {COMPARE_ROWS.map((r) => (
              <div key={r.them} className={styles["compare-row"]}>
                <span className={styles["compare-mark"]} aria-hidden>
                  ·
                </span>
                <span>{r.them}</span>
              </div>
            ))}
          </div>

          <div className={`${styles["compare-col"]} ${styles.us}`}>
            <div className={styles["compare-head"]}>
              <span className={`${styles["compare-tag"]} ${styles.accent}`}>OpenResearch</span>
            </div>
            {COMPARE_ROWS.map((r) => (
              <div key={r.us} className={styles["compare-row"]}>
                <span className={`${styles["compare-mark"]} ${styles.ok}`} aria-hidden>
                  ✓
                </span>
                <span>{r.us}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ─────────────────────────  PIPELINE  ──────────────────────────── */

type Stage = {
  n: string;
  word: string;
  stage: string;
  head: string;
  blurb: string;
  subs: ReadonlyArray<[string, string]>;
  figLabel: string;
  crumbs: string;
  crumbsBold: string;
  crumbsRight: string;
  Figure: () => React.JSX.Element;
  bleed?: "l" | "r";
};

const STAGES: ReadonlyArray<Stage> = [
  {
    n: "1.0",
    word: "Comprehension",
    stage: "stage 01 / 05",
    head: "Read the paper like an author would.",
    blurb:
      "It ingests the PDF and source repo, extracts every quantitative claim, and links each one to the table, figure, or equation that supports it.",
    subs: [
      ["1.1", "Parsing"],
      ["1.2", "Claim extraction"],
      ["1.3", "Citation graph"],
      ["1.4", "Rubric induction"]
    ],
    figLabel: "FIG 1.2 — Claim extraction · rubric induction",
    crumbs: "comprehension",
    crumbsBold: "claims + citation graph",
    crumbsRight: "illustrative",
    Figure: ComprehensionMock,
    bleed: "r"
  },
  {
    n: "2.0",
    word: "Environment",
    stage: "stage 02 / 05",
    head: "Stand up exactly the machine the paper ran on.",
    blurb:
      "Pinned CUDA, pinned wheels, pinned data hashes. A reproducible container is built and locked before a single training step runs.",
    subs: [
      ["2.1", "Toolchain"],
      ["2.2", "Lockfile"],
      ["2.3", "Dataset hashes"],
      ["2.4", "GPU provisioning"]
    ],
    figLabel: "FIG 2.2 — Lockfile · GPU provisioning",
    crumbs: "env",
    crumbsBold: "container",
    crumbsRight: "illustrative",
    Figure: EnvironmentMock
  },
  {
    n: "3.0",
    word: "Implementation",
    stage: "stage 03 / 05",
    head: "Write the method, then improve it.",
    blurb:
      "A baseline is generated from the paper's pseudocode and figures. The agent then explores bounded variations against the induced rubric — every edit attributed, every diff reviewable.",
    subs: [
      ["3.1", "Baseline"],
      ["3.2", "Improvement search"],
      ["3.3", "Diff review"],
      ["3.4", "Attribution"]
    ],
    figLabel: "FIG 3.2 — Baseline vs. improvement · diff view",
    crumbs: "implementation",
    crumbsBold: "recurse.py",
    crumbsRight: "illustrative",
    Figure: ImplementationMock,
    bleed: "l"
  },
  {
    n: "4.0",
    word: "Experiments",
    stage: "stage 04 / 05",
    head: "Run the table the paper claims.",
    blurb:
      "Every row, every column, every seed. Runs are scheduled across the rubric, results stream into the lab live, and intermediate state is logged so a failed run is debuggable, not opaque.",
    subs: [
      ["4.1", "Run scheduler"],
      ["4.2", "Seed sweeps"],
      ["4.3", "Live metrics"],
      ["4.4", "Run registry"]
    ],
    figLabel: "FIG 4.3 — Live metrics · run registry",
    crumbs: "experiments",
    crumbsBold: "results matrix",
    crumbsRight: "illustrative",
    Figure: ExperimentsMock
  },
  {
    n: "5.0",
    word: "Verification",
    stage: "stage 05 / 05",
    head: "Score the result against the paper's own claims.",
    blurb:
      "Each observed metric is graded against the rubric — match, deviation, or contradiction — with the supporting table cell surfaced for every line. This is the brief you take away.",
    subs: [
      ["5.1", "Rubric scoring"],
      ["5.2", "Deltas + intervals"],
      ["5.3", "Cited brief"],
      ["5.4", "Audit trail"]
    ],
    figLabel: "FIG 5.1 — Reproduction scorecard · the cited brief",
    crumbs: "verification",
    crumbsBold: "scorecard",
    crumbsRight: "illustrative",
    Figure: VerificationMock,
    bleed: "r"
  }
];

function PipelineSection(): React.JSX.Element {
  return (
    <div id="pipeline">
      <section className={`${styles.spec} ${styles["pipeline-lead"]}`}>
        <div className={styles.wrap}>
          <div className={styles["section-lead"]}>
            <span className={styles["h-eyebrow"]}>How it works</span>
            <h2 className={styles.headline}>From question to verified brief.</h2>
            <p className={styles.lede}>
              Five stages, one closed loop. Point it at an arXiv ID; it works through comprehension to
              a sealed scorecard — and if a claim doesn&apos;t reproduce, you see exactly which one.
            </p>
          </div>
        </div>
      </section>

      {STAGES.map((s) => (
        <PipelineStage key={s.n} stage={s} />
      ))}
    </div>
  );
}

function PipelineStage({ stage: s }: { stage: Stage }): React.JSX.Element {
  const Figure = s.Figure;
  const frameClass =
    s.bleed === "r"
      ? `${styles.frame} ${styles["frame-bleed-r"]}`
      : s.bleed === "l"
        ? `${styles.frame} ${styles["frame-bleed-l"]}`
        : styles.frame;

  return (
    <section className={styles.spec}>
      <div className={styles.wrap}>
        <div className={styles["spec-header"]}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>{s.n}</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>{s.word}</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>{s.stage}</span>
            </div>
            <h3 className={styles.head}>{s.head}</h3>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>{s.blurb}</p>
            <div className={styles["spec-substeps"]}>
              {s.subs.map(([num, label]) => (
                <span key={num}>
                  <span className={styles.num}>§ {num}</span>
                  <span>{label}</span>
                </span>
              ))}
            </div>
          </div>
        </div>

        <figure className={styles["spec-visual"]}>
          <span className={styles["fig-label"]}>{s.figLabel}</span>
          <div className={frameClass}>
            <div className={styles["mock-chrome"]}>
              <span className={styles.crumbs}>
                {s.crumbs}&nbsp;/&nbsp;<b>{s.crumbsBold}</b>
              </span>
              <span className={styles.right}>{s.crumbsRight}</span>
            </div>
            <Figure />
          </div>
        </figure>
      </div>
    </section>
  );
}

/* ──────────────────────────  TRUST BAND  ───────────────────────── */

const TRUST_CELLS: ReadonlyArray<{ lbl: string; v: string; sub: string }> = [
  { lbl: "Open source", v: "On GitHub", sub: "Read every line of the agent — no black box." },
  { lbl: "Audit trail", v: "Per-claim", sub: "Every score links to the run and the source it checked." },
  { lbl: "Reproducible", v: "Pinned + seeded", sub: "Same environment, same seeds — re-run any result yourself." },
  { lbl: "Research preview", v: "Watch it work", sub: "Stream the agent reasoning and executing, step by step." }
];

function TrustSection(): React.JSX.Element {
  return (
    <section className={styles.trust} id="trust">
      <div className={styles.wrap}>
        <div className={styles["bench-row"]}>
          <div className={styles["spec-num"]}>
            <span className={styles.glyph}>§</span>
            <span className={styles.n}>6.0</span>
          </div>
          <div className={styles["spec-title"]}>
            <div className={styles.name}>
              <span className={styles.word}>Trust</span>
              <span className={styles.rule} aria-hidden />
              <span className={styles.stage}>shown, not claimed</span>
            </div>
            <h2 className={styles.head}>Open-source. Every result is re-runnable.</h2>
          </div>
          <div>
            <p className={styles["spec-blurb"]}>
              We&apos;re in research preview, so the page carries no scoreboard we can&apos;t stand
              behind. Instead, the proof is the product: read the code, watch a run stream live, and
              re-execute any result on the same pinned environment.
            </p>
          </div>
        </div>

        <div className={styles["trust-grid"]}>
          {TRUST_CELLS.map((c) => (
            <div key={c.lbl} className={styles["trust-cell"]}>
              <div className={styles.lbl}>{c.lbl}</div>
              <div className={styles.v}>{c.v}</div>
              <div className={styles.sub}>{c.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ────────────────────────────  CTA  ────────────────────────────── */

function CTAFooter(): React.JSX.Element {
  return (
    <section className={styles["cta-foot"]} id="github">
      <div className={styles.wrap}>
        <div className={styles["h-eyebrow"]} style={{ marginBottom: 22 }}>
          § 7.0 — Try it
        </div>
        <h2>
          Point it at a paper.
          <br />
          Get back a brief you can check.
        </h2>
        <p className={styles.lede}>
          Drop in an arXiv ID. OpenResearch returns a sealed environment, an implementation, a
          scorecard, and an audit trail — and tells you exactly which claims held and which
          didn&apos;t.
        </p>
        <div className={styles.actions}>
          <Link href="/lab" className={`${styles.btn} ${styles["btn-primary"]}`}>
            <span>Open the lab</span>
            <span aria-hidden>→</span>
          </Link>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className={`${styles.btn} ${styles["btn-ghost"]}`}
          >
            <GitHubIcon />
            <span>View source on GitHub</span>
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
          <span>OpenResearch · 2026</span>
        </div>
        <div className={styles.links}>
          <a href="#different">Why it&apos;s different</a>
          <a href="#pipeline">How it works</a>
          <Link href="/leaderboard">Runs</Link>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
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
