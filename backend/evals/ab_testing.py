"""Bayesian A/B testing for comparing agent versions.

Uses Beta-Binomial model for binary metrics (build success, run success)
and Normal model for continuous metrics (fidelity score, metric match).

Reaches statistical significance faster than frequentist methods with
small sample sizes — ideal for expensive agent evaluations.

References:
- Parloa: "How to A/B Test AI Agents With a Bayesian Model" (2025)
- Beta-Binomial conjugate model for binary outcomes
- Monte Carlo estimation of P(A > B)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from backend.evals.schemas import ABTestResult


@dataclass
class BetaPosterior:
    """Beta distribution posterior for binary outcomes."""
    alpha: float = 1.0  # prior successes + 1
    beta: float = 1.0   # prior failures + 1
    n: int = 0

    def update(self, success: bool) -> None:
        self.n += 1
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self, rng: random.Random) -> float:
        """Sample from Beta distribution using Jöhnk's algorithm."""
        return _beta_sample(self.alpha, self.beta, rng)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Approximate credible interval using normal approximation."""
        z = 1.96 if level == 0.95 else 2.576  # 95% or 99%
        std = math.sqrt(self.variance)
        return (
            max(0.0, self.mean - z * std),
            min(1.0, self.mean + z * std),
        )


@dataclass
class NormalPosterior:
    """Normal distribution posterior for continuous outcomes."""
    mean: float = 0.0
    variance: float = 1.0
    n: int = 0
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        self._sum += value
        self._sum_sq += value * value
        self.mean = self._sum / self.n
        if self.n > 1:
            self.variance = (self._sum_sq / self.n - self.mean ** 2) * self.n / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 1e-10))

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std / math.sqrt(max(self.n, 1)))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        z = 1.96 if level == 0.95 else 2.576
        se = self.std / math.sqrt(max(self.n, 1))
        return (self.mean - z * se, self.mean + z * se)


class BayesianABTest:
    """Bayesian A/B test comparing two agent versions on a single metric.

    Usage:
        test = BayesianABTest("v1.0", "v1.1", "build_success", is_binary=True)
        test.add_observation_a(True)
        test.add_observation_a(True)
        test.add_observation_b(False)
        test.add_observation_b(True)
        result = test.result()
        print(f"P(A > B) = {result.p_a_better:.3f}")
    """

    def __init__(
        self,
        version_a: str,
        version_b: str,
        metric: str,
        *,
        is_binary: bool = True,
        significance_threshold: float = 0.95,
        n_samples: int = 10_000,
        seed: int | None = None,
    ):
        self.version_a = version_a
        self.version_b = version_b
        self.metric = metric
        self.is_binary = is_binary
        self.significance_threshold = significance_threshold
        self.n_samples = n_samples
        self._rng = random.Random(seed)

        if is_binary:
            self._posterior_a: BetaPosterior | NormalPosterior = BetaPosterior()
            self._posterior_b: BetaPosterior | NormalPosterior = BetaPosterior()
        else:
            self._posterior_a = NormalPosterior()
            self._posterior_b = NormalPosterior()

    def add_observation_a(self, value: float | bool) -> None:
        if self.is_binary:
            assert isinstance(self._posterior_a, BetaPosterior)
            self._posterior_a.update(bool(value))
        else:
            assert isinstance(self._posterior_a, NormalPosterior)
            self._posterior_a.update(float(value))

    def add_observation_b(self, value: float | bool) -> None:
        if self.is_binary:
            assert isinstance(self._posterior_b, BetaPosterior)
            self._posterior_b.update(bool(value))
        else:
            assert isinstance(self._posterior_b, NormalPosterior)
            self._posterior_b.update(float(value))

    def probability_a_better(self) -> float:
        """Monte Carlo estimate of P(A > B)."""
        a_wins = 0
        for _ in range(self.n_samples):
            sample_a = self._posterior_a.sample(self._rng)
            sample_b = self._posterior_b.sample(self._rng)
            if sample_a > sample_b:
                a_wins += 1
        return a_wins / self.n_samples

    def is_significant(self) -> bool:
        """True if we have high confidence one version is better."""
        p = self.probability_a_better()
        return p > self.significance_threshold or p < (1 - self.significance_threshold)

    def result(self) -> ABTestResult:
        """Compute full A/B test result."""
        p_a = self.probability_a_better()
        p_b = 1.0 - p_a
        significant = (
            p_a > self.significance_threshold or
            p_b > self.significance_threshold
        )

        winner = None
        if significant:
            winner = self.version_a if p_a > p_b else self.version_b

        return ABTestResult(
            version_a=self.version_a,
            version_b=self.version_b,
            metric=self.metric,
            n_a=self._posterior_a.n,
            n_b=self._posterior_b.n,
            mean_a=self._posterior_a.mean,
            mean_b=self._posterior_b.mean,
            p_a_better=p_a,
            p_b_better=p_b,
            credible_interval_a=self._posterior_a.credible_interval(),
            credible_interval_b=self._posterior_b.credible_interval(),
            is_significant=significant,
            winner=winner,
            details=(
                f"Bayesian {'Beta-Binomial' if self.is_binary else 'Normal'} model, "
                f"{self.n_samples} MC samples, threshold={self.significance_threshold}"
            ),
        )


class MultiMetricABTest:
    """Run A/B tests across multiple metrics simultaneously.

    Combines binary and continuous metrics into a single comparison.
    """

    def __init__(
        self,
        version_a: str,
        version_b: str,
        *,
        significance_threshold: float = 0.95,
        seed: int | None = None,
    ):
        self.version_a = version_a
        self.version_b = version_b
        self.significance_threshold = significance_threshold
        self._tests: dict[str, BayesianABTest] = {}
        self._seed = seed

    def register_metric(self, metric: str, is_binary: bool = True) -> None:
        self._tests[metric] = BayesianABTest(
            self.version_a, self.version_b, metric,
            is_binary=is_binary,
            significance_threshold=self.significance_threshold,
            seed=self._seed,
        )

    def add_observation_a(self, metrics: dict[str, float | bool]) -> None:
        for metric, value in metrics.items():
            if metric in self._tests:
                self._tests[metric].add_observation_a(value)

    def add_observation_b(self, metrics: dict[str, float | bool]) -> None:
        for metric, value in metrics.items():
            if metric in self._tests:
                self._tests[metric].add_observation_b(value)

    def results(self) -> dict[str, ABTestResult]:
        return {name: test.result() for name, test in self._tests.items()}

    def overall_winner(self) -> str | None:
        """Determine overall winner based on majority of significant metrics."""
        results = self.results()
        a_wins = sum(1 for r in results.values() if r.winner == self.version_a)
        b_wins = sum(1 for r in results.values() if r.winner == self.version_b)
        if a_wins > b_wins and a_wins > 0:
            return self.version_a
        elif b_wins > a_wins and b_wins > 0:
            return self.version_b
        return None


# --- Helpers ---


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Sample from Beta(alpha, beta) using gamma variates."""
    x = _gamma_sample(alpha, rng)
    y = _gamma_sample(beta, rng)
    if x + y == 0:
        return 0.5
    return x / (x + y)


def _gamma_sample(shape: float, rng: random.Random) -> float:
    """Sample from Gamma(shape, 1) using Marsaglia and Tsang's method."""
    if shape < 1:
        return _gamma_sample(shape + 1, rng) * rng.random() ** (1.0 / shape)

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    while True:
        x = rng.gauss(0, 1)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if u < 1.0 - 0.0331 * x ** 4:
            return d * v
        if math.log(u) < 0.5 * x ** 2 + d * (1.0 - v + math.log(v)):
            return d * v
