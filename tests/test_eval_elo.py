"""Tests for Elo tournament system."""


from backend.evals.elo import (
    EloTournament,
    ELO_START_RATING,
    compute_confidence,
)


class TestEloTournament:
    def test_add_competitor(self):
        t = EloTournament()
        t.add_competitor("v1.0")
        assert "v1.0" in t.competitors
        rating = t.get_rating("v1.0")
        assert rating is not None
        assert rating.rating == ELO_START_RATING

    def test_record_match_a_wins(self):
        t = EloTournament()
        t.add_competitor("v1")
        t.add_competitor("v2")
        t.record_match("v1", "v2", winner="v1", paper_id="ppo")

        r1 = t.get_rating("v1")
        r2 = t.get_rating("v2")
        assert r1.rating > ELO_START_RATING
        assert r2.rating < ELO_START_RATING
        assert r1.wins == 1
        assert r2.losses == 1

    def test_record_match_draw(self):
        t = EloTournament()
        t.add_competitor("v1")
        t.add_competitor("v2")
        t.record_match("v1", "v2", winner=None)

        r1 = t.get_rating("v1")
        r2 = t.get_rating("v2")
        # Draw between equal ratings -> no change
        assert r1.rating == ELO_START_RATING
        assert r2.rating == ELO_START_RATING
        assert r1.draws == 1
        assert r2.draws == 1

    def test_strong_player_gains_less(self):
        t = EloTournament()
        t.add_competitor("strong", starting_rating=1700)
        t.add_competitor("weak", starting_rating=1300)

        # Strong wins (expected) -> small gain
        t.record_match("strong", "weak", winner="strong")
        strong = t.get_rating("strong")
        # Gain should be small since win was expected
        assert strong.rating - 1700 < 10

    def test_upset_gives_big_rating_change(self):
        t = EloTournament()
        t.add_competitor("strong", starting_rating=1700)
        t.add_competitor("weak", starting_rating=1300)

        # Weak wins (upset) -> big gain
        t.record_match("strong", "weak", winner="weak")
        weak = t.get_rating("weak")
        # Gain should be large since win was unexpected
        assert weak.rating - 1300 > 20

    def test_expected_score_symmetric(self):
        t = EloTournament()
        e_a = t.expected_score(1500, 1500)
        assert abs(e_a - 0.5) < 0.001

    def test_expected_score_400_diff(self):
        t = EloTournament()
        # 400 point difference -> expected ~91% win
        e_strong = t.expected_score(1900, 1500)
        assert e_strong > 0.9
        e_weak = t.expected_score(1500, 1900)
        assert e_weak < 0.1

    def test_rankings_sorted(self):
        t = EloTournament()
        t.add_competitor("v1")
        t.add_competitor("v2")
        t.add_competitor("v3")

        # v3 beats everyone
        t.record_match("v3", "v1", winner="v3")
        t.record_match("v3", "v2", winner="v3")
        t.record_match("v1", "v2", winner="v1")

        rankings = t.get_rankings()
        assert rankings[0].version == "v3"
        assert rankings[1].version == "v1"
        assert rankings[2].version == "v2"

    def test_matches_stored(self):
        t = EloTournament()
        t.add_competitor("v1")
        t.add_competitor("v2")
        t.record_match("v1", "v2", winner="v1", paper_id="ppo",
                       judge_rationale="A was better")
        assert len(t.matches) == 1
        assert t.matches[0].judge_rationale == "A was better"

    def test_round_robin(self):
        t = EloTournament()
        t.add_competitor("v1")
        t.add_competitor("v2")
        t.add_competitor("v3")

        # v1 always wins
        def match_fn(a, b):
            return "v1" if "v1" in (a, b) else a

        t.run_round_robin(match_fn, n_rounds=2, seed=42)
        rankings = t.get_rankings()
        assert rankings[0].version == "v1"
        # 3 competitors, round-robin = 3 matches per round, 2 rounds = 6 matches
        total_matches = sum(r.matches_played for r in rankings) // 2
        assert total_matches == 6

    def test_round_robin_with_papers(self):
        t = EloTournament()
        t.add_competitor("v1")
        t.add_competitor("v2")

        call_count = [0]
        def match_fn(a, b):
            call_count[0] += 1
            return a

        t.run_round_robin(match_fn, paper_ids=["ppo", "mixmatch"], n_rounds=1, seed=42)
        # 1 pair * 2 papers = 2 matches
        assert call_count[0] == 2

    def test_auto_add_competitor(self):
        t = EloTournament()
        t.record_match("new_a", "new_b", winner="new_a")
        assert "new_a" in t.competitors
        assert "new_b" in t.competitors


class TestJudgePrompt:
    def test_generate_prompt(self):
        t = EloTournament()
        prompt = t.generate_judge_prompt(
            "v1", "v2",
            "Research map A content",
            "Research map B content",
            paper_context="PPO CartPole",
        )
        assert "v1" in prompt
        assert "v2" in prompt
        assert "Research map A content" in prompt
        assert "PPO CartPole" in prompt
        assert "WINNER:" in prompt

    def test_parse_response_a_wins(self):
        t = EloTournament()
        winner = t.parse_judge_response("WINNER: A\nA was better because...", "v1", "v2")
        assert winner == "v1"

    def test_parse_response_b_wins(self):
        t = EloTournament()
        winner = t.parse_judge_response("WINNER: B\nB had better synthesis", "v1", "v2")
        assert winner == "v2"

    def test_parse_response_draw(self):
        t = EloTournament()
        winner = t.parse_judge_response("WINNER: DRAW\nBoth were equal", "v1", "v2")
        assert winner is None


class TestConfidence:
    def test_zero_matches(self):
        assert compute_confidence(0) < 0.1

    def test_ten_matches(self):
        c = compute_confidence(10)
        assert 0.3 < c < 0.5

    def test_thirty_matches(self):
        c = compute_confidence(30)
        assert c > 0.75

    def test_fifty_matches(self):
        c = compute_confidence(50)
        assert c > 0.9

    def test_monotonic(self):
        prev = 0
        for n in range(0, 100, 10):
            c = compute_confidence(n)
            assert c >= prev
            prev = c
