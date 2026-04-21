"""
Unit tests for engine/llm_judge.py
Run with: python -m pytest tests/test_llm_judge.py -v
Or quick check: python tests/test_llm_judge.py
"""
import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.llm_judge import LLMJudge


# ============================================================================
# Test: Conflict Detection (score_A=5, score_B=2 → trigger tie-breaker)
# ============================================================================

async def test_conflict_detection():
    """
    Cố tình mock score_A=5, score_B=2 để đảm bảo hệ thống
    nhận diện được xung đột và gọi tie-breaker.
    """
    judge = LLMJudge()

    # Mock OpenAI trả accuracy=5
    async def mock_openai(prompt):
        return {
            "accuracy": 5, "tone": 4,
            "reasoning": "Câu trả lời rất chính xác.",
            "tokens_used": 100, "model": "gpt-4o-mini"
        }

    # Mock Anthropic trả accuracy=2 (chênh lệch = 3 > threshold=1)
    async def mock_anthropic(prompt):
        return {
            "accuracy": 2, "tone": 3,
            "reasoning": "Câu trả lời thiếu nhiều thông tin.",
            "tokens_used": 80, "model": "claude-3-5-haiku-20241022"
        }

    judge._call_openai_judge = mock_openai
    judge._call_anthropic_judge = mock_anthropic

    result = await judge.evaluate_multi_judge(
        question="Gói Premium giá bao nhiêu?",
        answer="Gói Premium giá 499.000 VNĐ/tháng.",
        ground_truth="Gói Premium bao gồm lưu trữ không giới hạn, giá 499.000 VNĐ/tháng."
    )

    print(f"\n--- Test Conflict Detection ---")
    print(f"Final Score: {result['final_score']}")
    print(f"Agreement Rate: {result['agreement_rate']}")
    print(f"Conflict Resolved: {result['conflict_resolved']}")
    print(f"Individual Scores: {json.dumps(result['individual_scores'], indent=2)}")
    if result['tie_breaker']:
        print(f"Tie-breaker: {json.dumps(result['tie_breaker'], indent=2)}")

    # Assertions
    assert result["conflict_resolved"] == True, "❌ Conflict should be detected!"
    assert result["tie_breaker"] is not None, "❌ Tie-breaker should have been called!"
    assert "agreement_rate" in result, "❌ Missing agreement_rate!"
    assert result["agreement_rate"] == 0.25, f"❌ Agreement rate should be 0.25 (diff=3), got {result['agreement_rate']}"

    print("✅ Conflict detection test PASSED!")
    return result


# ============================================================================
# Test: Agreement (score_A=4, score_B=4 → no conflict)
# ============================================================================

async def test_agreement():
    """
    Mock 2 Judge cho điểm giống nhau → không trigger tie-breaker.
    """
    judge = LLMJudge()

    async def mock_openai(prompt):
        return {
            "accuracy": 4, "tone": 4,
            "reasoning": "Tốt.", "tokens_used": 100, "model": "gpt-4o-mini"
        }

    async def mock_anthropic(prompt):
        return {
            "accuracy": 4, "tone": 5,
            "reasoning": "Rất tốt.", "tokens_used": 80, "model": "claude-3-5-haiku-20241022"
        }

    judge._call_openai_judge = mock_openai
    judge._call_anthropic_judge = mock_anthropic

    result = await judge.evaluate_multi_judge(
        question="Test?", answer="Answer.", ground_truth="Answer."
    )

    print(f"\n--- Test Agreement ---")
    print(f"Final Score: {result['final_score']}")
    print(f"Agreement Rate: {result['agreement_rate']}")
    print(f"Conflict Resolved: {result['conflict_resolved']}")

    assert result["conflict_resolved"] == False, "❌ Should NOT trigger conflict!"
    assert result["tie_breaker"] is None, "❌ Tie-breaker should NOT be called!"
    assert result["agreement_rate"] == 1.0, f"❌ Agreement should be 1.0, got {result['agreement_rate']}"
    assert result["final_score"] == 4.25, f"❌ Score should be (4+4+4+5)/4=4.25, got {result['final_score']}"

    print("✅ Agreement test PASSED!")


# ============================================================================
# Test: Result structure has all required fields
# ============================================================================

async def test_result_structure():
    """Kiểm tra kết quả có đủ 6 trường bắt buộc."""
    judge = LLMJudge()

    async def mock_judge(prompt):
        return {
            "accuracy": 3, "tone": 3,
            "reasoning": "OK.", "tokens_used": 50, "model": "mock"
        }

    judge._call_openai_judge = mock_judge
    judge._call_anthropic_judge = mock_judge

    result = await judge.evaluate_multi_judge("Q?", "A.", "GT.")

    required_fields = ["final_score", "agreement_rate", "individual_scores",
                       "total_tokens", "conflict_resolved", "tie_breaker"]

    for field in required_fields:
        assert field in result, f"❌ Missing required field: {field}"

    print(f"\n--- Test Result Structure ---")
    print(f"All 6 required fields present: {required_fields}")
    print("✅ Structure test PASSED!")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("🧪 Running LLM Judge Tests...\n")

    asyncio.run(test_result_structure())
    asyncio.run(test_agreement())
    asyncio.run(test_conflict_detection())

    print("\n🎉 All LLM Judge tests PASSED!")
