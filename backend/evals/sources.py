"""Literature sources for OpenResearch evaluation methodology.

All papers, benchmarks, and tools referenced in our eval design.
Organized by category for citation in papers and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Source:
    key: str
    title: str
    authors: str
    year: int
    url: str
    category: str
    relevance: str


EVAL_SOURCES: dict[str, Source] = {
    # --- Agent Evaluation Frameworks ---
    "inspect_ai": Source(
        key="inspect_ai",
        title="Inspect: A Framework for Large Language Model Evaluations",
        authors="UK AI Safety Institute (AISI)",
        year=2024,
        url="https://inspect.aisi.org.uk/",
        category="framework",
        relevance="Open-source eval framework with Docker sandboxing. "
                  "Dataset→Task→Solver→Scorer primitives map to our pipeline.",
    ),
    "deepeval": Source(
        key="deepeval",
        title="DeepEval: Open-Source LLM Evaluation Framework",
        authors="Confident AI",
        year=2024,
        url="https://github.com/confident-ai/deepeval",
        category="framework",
        relevance="Pytest-native eval with 50+ metrics. Used for CI-level reproduction scoring.",
    ),
    "braintrust": Source(
        key="braintrust",
        title="Braintrust: AI Product Evaluation Platform",
        authors="Braintrust",
        year=2024,
        url="https://www.braintrust.dev/",
        category="framework",
        relevance="Production observability + eval. Relevant for future production deployment.",
    ),
    "langsmith": Source(
        key="langsmith",
        title="LangSmith: Full-Lifecycle LLM Application Development Platform",
        authors="LangChain",
        year=2024,
        url="https://www.langchain.com/langsmith",
        category="framework",
        relevance="Multi-turn evals and trace analysis. Insights Agent for auto-discovery.",
    ),

    # --- Reproduction Benchmarks ---
    "paperbench": Source(
        key="paperbench",
        title="PaperBench: Evaluating AI's Ability to Replicate AI Research",
        authors="OpenAI",
        year=2025,
        url="https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf",
        category="benchmark",
        relevance="20 ML papers, 8316 sub-tasks. Claude 3.5 achieves 21% vs human 41%. "
                  "Our primary comparison benchmark for reproduction fidelity.",
    ),
    "researchcodebench": Source(
        key="researchcodebench",
        title="ResearchCodeBench: Evaluating LLMs on ML Code Implementation",
        authors="Various",
        year=2025,
        url="https://arxiv.org/html/2506.02314v1",
        category="benchmark",
        relevance="212 coding tasks from ML papers. Best models <40% correct. "
                  "Comparison for our code generation quality.",
    ),
    "core_bench": Source(
        key="core_bench",
        title="CORE-Bench: Computational Reproducibility Benchmark",
        authors="Various",
        year=2025,
        url="https://arxiv.org/abs/2409.11363",
        category="benchmark",
        relevance="Tests full pipeline: code→execution→debugging without human. "
                  "Direct comparison for our end-to-end pipeline.",
    ),
    "mle_bench": Source(
        key="mle_bench",
        title="MLE-bench: AI Research Agents for Machine Learning",
        authors="OpenAI",
        year=2024,
        url="https://arxiv.org/html/2507.02554v1",
        category="benchmark",
        relevance="75 Kaggle tasks for ML engineering. Measures engineering competence.",
    ),

    # --- Scientific Discovery Evaluation ---
    "ai_scientist_v1": Source(
        key="ai_scientist_v1",
        title="The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
        authors="Sakana AI",
        year=2024,
        url="https://sakana.ai/ai-scientist/",
        category="discovery",
        relevance="Automated reviewer achieving 69% balanced accuracy. "
                  "Independent audit found 42% execution failures, 57% hallucinated results.",
    ),
    "ai_scientist_v2": Source(
        key="ai_scientist_v2",
        title="The AI Scientist v2: Agentic Tree Search for Scientific Discovery",
        authors="Sakana AI",
        year=2025,
        url="https://arxiv.org/abs/2504.08066",
        category="discovery",
        relevance="Best-first tree search for hypotheses. Novelty-annotated hypothesis banks.",
    ),
    "ari_evaluation": Source(
        key="ari_evaluation",
        title="Evaluating Sakana's AI Scientist: Independent Assessment",
        authors="ARI",
        year=2025,
        url="https://arxiv.org/abs/2502.14297",
        category="discovery",
        relevance="Critical audit: 57% hallucinated results, novelty misclassification, "
                  "only 14.7% recent citations. Motivates our integrity checks.",
    ),
    "google_co_scientist": Source(
        key="google_co_scientist",
        title="AI Co-Scientist: Accelerating Scientific Breakthroughs",
        authors="Google Research",
        year=2025,
        url="https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/",
        category="discovery",
        relevance="Elo ratings from self-play tournaments. Validated with wet-lab experiments. "
                  "Motivates our Elo tournament system.",
    ),
    "codescientist": Source(
        key="codescientist",
        title="CodeScientist: Genetic Search for AI Research Ideas",
        authors="AI2 (Allen Institute)",
        year=2025,
        url="https://arxiv.org/abs/2503.22708",
        category="discovery",
        relevance="19 automated discoveries, 6 judged novel+sound. Genetic search approach.",
    ),
    "discoveryworld": Source(
        key="discoveryworld",
        title="DiscoveryWorld: Evaluating Scientific Discovery Agents",
        authors="AI2",
        year=2024,
        url="https://arxiv.org/abs/2406.06769",
        category="benchmark",
        relevance="120 tasks, 3 dimensions (completion, process, understanding). "
                  "Humans 70%, AI 20% on hard tasks.",
    ),
    "scienceagentbench": Source(
        key="scienceagentbench",
        title="ScienceAgentBench: Toward Rigorous Assessment of Language Agents",
        authors="Various (ICLR 2025)",
        year=2025,
        url="https://arxiv.org/abs/2410.05080",
        category="benchmark",
        relevance="102 tasks from 44 papers. Best agent 34.3%. Target output is Python program.",
    ),
    "mlgym": Source(
        key="mlgym",
        title="MLGym: Open-Ended AI Research Tasks",
        authors="Meta",
        year=2025,
        url="https://arxiv.org/abs/2502.14499",
        category="benchmark",
        relevance="Key finding: agents find hyperparameters but don't generate novel algorithms. "
                  "Motivates our novelty scoring.",
    ),

    # --- Hypothesis Quality ---
    "rnd_novelty": Source(
        key="rnd_novelty",
        title="Enabling AI Scientists to Recognize Innovation: Relative Neighbor Density",
        authors="Various",
        year=2025,
        url="https://arxiv.org/html/2503.01508",
        category="methodology",
        relevance="RND algorithm: embed ideas, measure neighbor density. AUROC 0.82. "
                  "Basis for our novelty scoring.",
    ),
    "idea_novelty_checker": Source(
        key="idea_novelty_checker",
        title="Literature-Grounded Novelty Assessment via RAG",
        authors="Various",
        year=2025,
        url="https://arxiv.org/abs/2506.22026",
        category="methodology",
        relevance="RAG pipeline for novelty checking. 13% higher human agreement.",
    ),
    "liveideabench": Source(
        key="liveideabench",
        title="LiveIdeaBench: Evaluating LLMs' Scientific Creativity with Minimal Context",
        authors="Various",
        year=2025,
        url="https://liveideabench.com/",
        category="benchmark",
        relevance="1180 keywords, 22 domains, dynamic LLM judge panel. "
                  "Scientific creativity poorly predicted by general intelligence.",
    ),
    "ai_idea_bench": Source(
        key="ai_idea_bench",
        title="AI Idea Bench: Quantitative Evaluation of LLM-Generated Research Ideas",
        authors="Various",
        year=2025,
        url="https://arxiv.org/abs/2504.14191",
        category="benchmark",
        relevance="3495 papers from AI conferences. Quantitative idea comparison.",
    ),

    # --- Integrity & P-Hacking Detection ---
    "hidden_pitfalls": Source(
        key="hidden_pitfalls",
        title="The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems",
        authors="Various",
        year=2025,
        url="https://arxiv.org/html/2509.08713",
        category="methodology",
        relevance="4 failure modes with detection tests: post-hoc selection, data leakage, "
                  "metric misuse, benchmark selection bias. Detection 55%→82% with traces. "
                  "Directly motivates our integrity checking.",
    ),

    # --- Research Synthesis Evaluation ---
    "researchrubrics": Source(
        key="researchrubrics",
        title="ResearchRubrics: Benchmarking Research Synthesis Quality",
        authors="Scale AI (ICLR 2026)",
        year=2025,
        url="https://arxiv.org/html/2511.07685v1",
        category="methodology",
        relevance="6-dimension rubric, 2593 expert criteria, ternary grading. "
                  "Best agents 67.7%. Basis for our Research Map scoring.",
    ),

    # --- A/B Testing ---
    "parloa_bayesian": Source(
        key="parloa_bayesian",
        title="How to A/B Test AI Agents With a Bayesian Model",
        authors="Parloa",
        year=2025,
        url="https://www.parloa.com/labs/research/ai-agent-testing/",
        category="methodology",
        relevance="Hierarchical Bayesian model combining binary+continuous metrics. "
                  "Reaches significance faster. Basis for our A/B testing.",
    ),

    # --- Meta / Surveys ---
    "metr_time_horizon": Source(
        key="metr_time_horizon",
        title="Measuring AI Ability to Complete Long Tasks",
        authors="METR",
        year=2025,
        url="https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/",
        category="methodology",
        relevance="Time horizon doubling every 7 months. HCAST 189 tasks with auto-eval.",
    ),
    "eval_survey": Source(
        key="eval_survey",
        title="Survey on Evaluation of LLM-based Agents",
        authors="Various",
        year=2025,
        url="https://arxiv.org/abs/2503.16416",
        category="survey",
        relevance="Comprehensive survey of agent evaluation approaches.",
    ),
    "verina": Source(
        key="verina",
        title="VERINA: Benchmarking Verifiable Code Generation",
        authors="Various",
        year=2025,
        url="https://arxiv.org/html/2505.23135",
        category="benchmark",
        relevance="61.4% correct code, 51% sound specs, 3.6% successful proofs. "
                  "Formal verification still too hard; prefer execution-based.",
    ),
    "novelseek": Source(
        key="novelseek",
        title="NovelSeek: Closed-Loop Multi-Agent Framework for Scientific Discovery",
        authors="Various",
        year=2025,
        url="https://arxiv.org/abs/2505.16938",
        category="discovery",
        relevance="Idea bank in embeddings to avoid dead ends. Concrete yield improvements.",
    ),
    "re_bench": Source(
        key="re_bench",
        title="RE-Bench: AI R&D Evaluation (7 ML Engineering Tasks)",
        authors="METR",
        year=2024,
        url="https://metr.org/AI_R_D_Evaluation_Report.pdf",
        category="benchmark",
        relevance="AI 4x humans at 2h but plateaus at longer horizons. "
                  "Motivates measuring wall-time efficiency.",
    ),
}


def get_sources_by_category(category: str) -> list[Source]:
    return [s for s in EVAL_SOURCES.values() if s.category == category]


def get_comparison_benchmarks() -> list[Source]:
    """Get benchmarks we compare our agent against."""
    return [
        EVAL_SOURCES["paperbench"],
        EVAL_SOURCES["researchcodebench"],
        EVAL_SOURCES["core_bench"],
        EVAL_SOURCES["scienceagentbench"],
        EVAL_SOURCES["discoveryworld"],
    ]


def format_citation(key: str) -> str:
    """Format a source as a citation string."""
    s = EVAL_SOURCES[key]
    return f"{s.authors}. \"{s.title}.\" ({s.year}). {s.url}"


def format_all_citations() -> str:
    """Format all sources for a references section."""
    lines = []
    categories = sorted(set(s.category for s in EVAL_SOURCES.values()))
    for cat in categories:
        lines.append(f"\n## {cat.title()}\n")
        for s in get_sources_by_category(cat):
            lines.append(f"- [{s.key}] {s.authors}. \"{s.title}.\" ({s.year}). {s.url}")
            lines.append(f"  Relevance: {s.relevance}")
    return "\n".join(lines)
