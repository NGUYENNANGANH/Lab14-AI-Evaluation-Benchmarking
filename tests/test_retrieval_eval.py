"""
Unit tests for engine/retrieval_eval.py
Run with: python -m pytest tests/test_retrieval_eval.py -v
"""
import sys
import os
import asyncio
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.retrieval_eval import RetrievalEvaluator


@pytest.fixture
def evaluator():
    return RetrievalEvaluator()


# ============================================================================
# Tests for calculate_hit_rate
# ============================================================================

class TestHitRate:
    """Test cases for Hit Rate calculation."""

    def test_hit_rate_basic_hit(self, evaluator):
        """Required test: expected ID found in retrieved list."""
        assert evaluator.calculate_hit_rate(['A'], ['B', 'A']) == 1.0

    def test_hit_rate_basic_miss(self, evaluator):
        """Expected ID NOT in retrieved list → 0.0."""
        assert evaluator.calculate_hit_rate(['A'], ['B', 'C', 'D']) == 0.0

    def test_hit_rate_first_position(self, evaluator):
        """Expected ID at first position."""
        assert evaluator.calculate_hit_rate(['A'], ['A', 'B', 'C']) == 1.0

    def test_hit_rate_multiple_expected(self, evaluator):
        """At least one expected ID is present → hit."""
        assert evaluator.calculate_hit_rate(['A', 'B'], ['C', 'B', 'D']) == 1.0

    def test_hit_rate_no_match_multiple(self, evaluator):
        """No expected IDs found in retrieved."""
        assert evaluator.calculate_hit_rate(['A', 'B'], ['C', 'D', 'E']) == 0.0

    def test_hit_rate_top_k_boundary(self, evaluator):
        """Match at exactly top_k boundary."""
        # top_k=3, match at index 2 (3rd position) → hit
        assert evaluator.calculate_hit_rate(['A'], ['B', 'C', 'A', 'D'], top_k=3) == 1.0

    def test_hit_rate_beyond_top_k(self, evaluator):
        """Match beyond top_k → miss."""
        # top_k=2, 'A' at index 2 (3rd position) → miss
        assert evaluator.calculate_hit_rate(['A'], ['B', 'C', 'A', 'D'], top_k=2) == 0.0

    def test_hit_rate_empty_retrieved(self, evaluator):
        """Empty retrieved list → miss."""
        assert evaluator.calculate_hit_rate(['A'], []) == 0.0

    def test_hit_rate_top_k_1(self, evaluator):
        """top_k=1: only first result matters."""
        assert evaluator.calculate_hit_rate(['B'], ['A', 'B'], top_k=1) == 0.0
        assert evaluator.calculate_hit_rate(['A'], ['A', 'B'], top_k=1) == 1.0

    def test_hit_rate_large_top_k(self, evaluator):
        """top_k larger than retrieved list."""
        assert evaluator.calculate_hit_rate(['C'], ['A', 'B', 'C'], top_k=10) == 1.0


# ============================================================================
# Tests for calculate_mrr
# ============================================================================

class TestMRR:
    """Test cases for Mean Reciprocal Rank calculation."""

    def test_mrr_basic(self, evaluator):
        """Required test: match at 2nd position → 1/2 = 0.5."""
        assert evaluator.calculate_mrr(['A'], ['B', 'A', 'C']) == 0.5

    def test_mrr_first_position(self, evaluator):
        """Match at first position → 1/1 = 1.0."""
        assert evaluator.calculate_mrr(['A'], ['A', 'B', 'C']) == 1.0

    def test_mrr_third_position(self, evaluator):
        """Match at 3rd position → 1/3 ≈ 0.333."""
        assert abs(evaluator.calculate_mrr(['A'], ['B', 'C', 'A']) - 1/3) < 1e-9

    def test_mrr_no_match(self, evaluator):
        """No match → 0.0."""
        assert evaluator.calculate_mrr(['A'], ['B', 'C', 'D']) == 0.0

    def test_mrr_multiple_expected_first_match_wins(self, evaluator):
        """Multiple expected IDs: returns RR for earliest match."""
        # 'B' appears at index 1 (position 2), 'A' at index 3 (position 4)
        # First match should be 'B' at position 2 → 1/2
        assert evaluator.calculate_mrr(['A', 'B'], ['C', 'B', 'D', 'A']) == 0.5

    def test_mrr_empty_retrieved(self, evaluator):
        """Empty retrieved list → 0.0."""
        assert evaluator.calculate_mrr(['A'], []) == 0.0

    def test_mrr_last_position(self, evaluator):
        """Match at last position."""
        assert evaluator.calculate_mrr(['A'], ['B', 'C', 'D', 'E', 'A']) == 0.2

    def test_mrr_duplicate_in_retrieved(self, evaluator):
        """Duplicate expected in retrieved: first occurrence counts."""
        assert evaluator.calculate_mrr(['A'], ['B', 'A', 'A', 'C']) == 0.5


# ============================================================================
# Tests for evaluate_batch (async)
# ============================================================================

class TestEvaluateBatch:
    """Test cases for batch evaluation."""

    def test_evaluate_batch_basic(self, evaluator):
        """Batch eval should return valid structure."""
        dataset = [
            {
                "question": "Test Q1",
                "expected_retrieval_ids": ["doc_001"],
                "retrieved_ids": ["doc_001", "doc_002"],
                "metadata": {"type": "fact-check"}
            },
            {
                "question": "Test Q2",
                "expected_retrieval_ids": ["doc_003"],
                "retrieved_ids": ["doc_002", "doc_003"],
                "metadata": {"type": "fact-check"}
            }
        ]
        result = asyncio.run(evaluator.evaluate_batch(dataset))

        assert "avg_hit_rate" in result
        assert "avg_mrr" in result
        assert "total_cases" in result
        assert result["total_cases"] == 2
        assert result["avg_hit_rate"] == 1.0  # Both have hits
        assert result["avg_mrr"] == 0.75  # (1.0 + 0.5) / 2

    def test_evaluate_batch_with_miss(self, evaluator):
        """One hit, one miss → avg_hit_rate = 0.5."""
        dataset = [
            {
                "question": "Q1",
                "expected_retrieval_ids": ["A"],
                "retrieved_ids": ["A", "B"],
                "metadata": {"type": "test"}
            },
            {
                "question": "Q2",
                "expected_retrieval_ids": ["C"],
                "retrieved_ids": ["A", "B"],
                "metadata": {"type": "test"}
            }
        ]
        result = asyncio.run(evaluator.evaluate_batch(dataset))
        assert result["avg_hit_rate"] == 0.5
        assert result["avg_mrr"] == 0.5  # (1.0 + 0.0) / 2

    def test_evaluate_batch_has_breakdown(self, evaluator):
        """Batch eval should include breakdown by type."""
        dataset = [
            {
                "question": "Q1",
                "expected_retrieval_ids": ["A"],
                "retrieved_ids": ["A"],
                "metadata": {"type": "easy"}
            },
            {
                "question": "Q2",
                "expected_retrieval_ids": ["B"],
                "retrieved_ids": ["B"],
                "metadata": {"type": "hard"}
            }
        ]
        result = asyncio.run(evaluator.evaluate_batch(dataset))
        assert "breakdown_by_type" in result
        assert "easy" in result["breakdown_by_type"]
        assert "hard" in result["breakdown_by_type"]


# ============================================================================
# Quick runnable check (without pytest)
# ============================================================================

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()

    # Required assertions from lab instructions
    assert evaluator.calculate_hit_rate(['A'], ['B', 'A']) == 1.0, "Hit Rate test FAILED"
    print("✅ calculate_hit_rate(['A'], ['B', 'A']) == 1.0")

    assert evaluator.calculate_mrr(['A'], ['B', 'A', 'C']) == 0.5, "MRR test FAILED"
    print("✅ calculate_mrr(['A'], ['B', 'A', 'C']) == 0.5")

    # Extra assertions
    assert evaluator.calculate_hit_rate(['A'], ['B', 'C', 'D']) == 0.0
    print("✅ calculate_hit_rate(['A'], ['B', 'C', 'D']) == 0.0")

    assert evaluator.calculate_mrr(['A'], ['A', 'B', 'C']) == 1.0
    print("✅ calculate_mrr(['A'], ['A', 'B', 'C']) == 1.0")

    assert evaluator.calculate_mrr(['A'], ['B', 'C', 'D']) == 0.0
    print("✅ calculate_mrr(['A'], ['B', 'C', 'D']) == 0.0")

    print("\n🎉 All assertions passed!")
