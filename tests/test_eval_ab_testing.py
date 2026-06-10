"""Tests for Bayesian A/B testing system."""


from backend.evals.ab_testing import (
    BayesianABTest,
    BetaPosterior,
    MultiMetricABTest,
    NormalPosterior,
)


class TestBetaPosterior:
    def test_prior_is_uniform(self):
        p = BetaPosterior()
        assert p.mean == 0.5
        assert p.n == 0

    def test_update_success(self):
        p = BetaPosterior()
        p.update(True)
        assert p.alpha == 2
        assert p.beta == 1
        assert p.mean > 0.5

    def test_update_failure(self):
        p = BetaPosterior()
        p.update(False)
        assert p.alpha == 1
        assert p.beta == 2
        assert p.mean < 0.5

    def test_many_successes(self):
        p = BetaPosterior()
        for _ in range(100):
            p.update(True)
        assert p.mean > 0.95
        assert p.n == 100

    def test_credible_interval(self):
        p = BetaPosterior()
        for _ in range(50):
            p.update(True)
        for _ in range(50):
            p.update(False)
        lo, hi = p.credible_interval()
        assert lo < 0.5 < hi
        assert hi - lo < 0.2  # reasonably tight with 100 obs


class TestNormalPosterior:
    def test_initial_state(self):
        p = NormalPosterior()
        assert p.n == 0

    def test_update_mean(self):
        p = NormalPosterior()
        p.update(10.0)
        p.update(20.0)
        assert p.mean == 15.0
        assert p.n == 2

    def test_variance_computation(self):
        p = NormalPosterior()
        for v in [10, 20, 30, 40, 50]:
            p.update(v)
        assert p.mean == 30.0
        assert p.variance > 0

    def test_credible_interval(self):
        p = NormalPosterior()
        for v in range(100):
            p.update(float(v))
        lo, hi = p.credible_interval()
        assert lo < p.mean < hi


class TestBayesianABTestBinary:
    def test_a_clearly_better(self):
        test = BayesianABTest("v1", "v2", "build_success", is_binary=True, seed=42)
        # v1: 90% success rate
        for _ in range(9):
            test.add_observation_a(True)
        test.add_observation_a(False)
        # v2: 50% success rate
        for _ in range(5):
            test.add_observation_b(True)
        for _ in range(5):
            test.add_observation_b(False)

        result = test.result()
        assert result.p_a_better > 0.9
        assert result.mean_a > result.mean_b
        assert result.winner == "v1"

    def test_b_clearly_better(self):
        test = BayesianABTest("v1", "v2", "run_success", is_binary=True, seed=42)
        # v1: 30% success
        for _ in range(3):
            test.add_observation_a(True)
        for _ in range(7):
            test.add_observation_a(False)
        # v2: 95% success
        for _ in range(19):
            test.add_observation_b(True)
        test.add_observation_b(False)

        result = test.result()
        assert result.p_b_better > 0.95
        assert result.winner == "v2"

    def test_inconclusive_with_equal_results(self):
        test = BayesianABTest("v1", "v2", "metric", is_binary=True, seed=42)
        for _ in range(5):
            test.add_observation_a(True)
            test.add_observation_a(False)
            test.add_observation_b(True)
            test.add_observation_b(False)

        result = test.result()
        assert not result.is_significant
        assert result.winner is None
        assert 0.3 < result.p_a_better < 0.7

    def test_small_sample_not_significant(self):
        test = BayesianABTest("v1", "v2", "metric", is_binary=True, seed=42)
        test.add_observation_a(True)
        test.add_observation_b(False)
        result = test.result()
        # With just 1 observation each, shouldn't be significant
        assert not result.is_significant


class TestBayesianABTestContinuous:
    def test_higher_mean_wins(self):
        test = BayesianABTest("v1", "v2", "fidelity", is_binary=False, seed=42)
        # v1: mean ~0.9
        for v in [0.85, 0.90, 0.92, 0.88, 0.91, 0.93, 0.89, 0.90, 0.87, 0.91]:
            test.add_observation_a(v)
        # v2: mean ~0.6
        for v in [0.55, 0.60, 0.62, 0.58, 0.61, 0.63, 0.59, 0.60, 0.57, 0.61]:
            test.add_observation_b(v)

        result = test.result()
        assert result.p_a_better > 0.95
        assert result.is_significant
        assert result.winner == "v1"

    def test_result_has_credible_intervals(self):
        test = BayesianABTest("v1", "v2", "score", is_binary=False, seed=42)
        for v in [0.8, 0.9, 0.85]:
            test.add_observation_a(v)
            test.add_observation_b(v - 0.2)

        result = test.result()
        assert result.credible_interval_a[0] < result.credible_interval_a[1]
        assert result.credible_interval_b[0] < result.credible_interval_b[1]


class TestMultiMetricABTest:
    def test_overall_winner(self):
        ab = MultiMetricABTest("v1", "v2", seed=42)
        ab.register_metric("build_success", is_binary=True)
        ab.register_metric("fidelity", is_binary=False)

        # v1 wins both
        for _ in range(10):
            ab.add_observation_a({"build_success": True, "fidelity": 0.9})
            ab.add_observation_b({"build_success": False, "fidelity": 0.4})

        assert ab.overall_winner() == "v1"

    def test_mixed_results_no_winner(self):
        ab = MultiMetricABTest("v1", "v2", seed=42)
        ab.register_metric("build_success", is_binary=True)
        ab.register_metric("fidelity", is_binary=False)

        # v1 wins build, v2 wins fidelity
        for _ in range(10):
            ab.add_observation_a({"build_success": True, "fidelity": 0.3})
            ab.add_observation_b({"build_success": False, "fidelity": 0.9})

        results = ab.results()
        assert len(results) == 2

    def test_all_metrics_returned(self):
        ab = MultiMetricABTest("v1", "v2", seed=42)
        ab.register_metric("m1", is_binary=True)
        ab.register_metric("m2", is_binary=True)
        ab.register_metric("m3", is_binary=False)

        ab.add_observation_a({"m1": True, "m2": False, "m3": 0.5})
        ab.add_observation_b({"m1": True, "m2": True, "m3": 0.7})

        results = ab.results()
        assert "m1" in results
        assert "m2" in results
        assert "m3" in results
