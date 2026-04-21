"""
Unit tests for engine/llm_judge.py
Run with: python -m pytest tests/test_llm_judge.py -v
Or quick check: python tests/test_llm_judge.py
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.llm_judge import LLMJudge


# ============================================================================
# Test: Conflict Detection (score_A=5, score_B=2 -> trigger tie-breaker)
# ============================================================================

async def test_conflict_detection():
    """
    Mock score_A=5, score_B=2 de dam bao he thong
    nhan dien duoc xung dot va goi tie-breaker.
    """
    judge = LLMJudge()

    # Mock gpt-4o-mini tra accuracy=5
    async def mock_model_a(prompt, model):
        return {
            "accuracy": 5, "tone": 4,
            "reasoning": "Cau tra loi rat chinh xac.",
            "tokens_used": 100, "model": "gpt-4o-mini"
        }

    # Mock gpt-4o tra accuracy=2 (chenh = 3 > threshold=1)
    call_count = 0
    async def mock_judge(prompt, model):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # First call: gpt-4o-mini
            return {
                "accuracy": 5, "tone": 4,
                "reasoning": "Cau tra loi rat chinh xac.",
                "tokens_used": 100, "model": "gpt-4o-mini"
            }
        elif call_count == 2:  # Second call: gpt-4o
            return {
                "accuracy": 2, "tone": 3,
                "reasoning": "Cau tra loi thieu nhieu thong tin.",
                "tokens_used": 80, "model": "gpt-4o"
            }
        else:  # Tie-breaker call
            return {
                "accuracy": 4, "tone": 4,
                "reasoning": "Tie-breaker: cau tra loi kha tot.",
                "tokens_used": 120, "model": "gpt-4o"
            }

    judge._call_judge = mock_judge

    result = await judge.evaluate_multi_judge(
        question="Goi Premium gia bao nhieu?",
        answer="Goi Premium gia 499.000 VND/thang.",
        ground_truth="Goi Premium bao gom luu tru khong gioi han, gia 499.000 VND/thang."
    )

    print("\n--- Test Conflict Detection ---")
    print(f"Final Score: {result['final_score']}")
    print(f"Agreement Rate: {result['agreement_rate']}")
    print(f"Conflict Resolved: {result['conflict_resolved']}")
    print(f"Individual Scores: {json.dumps(result['individual_scores'], indent=2, ensure_ascii=False)}")
    if result['tie_breaker']:
        print(f"Tie-breaker: {json.dumps(result['tie_breaker'], indent=2, ensure_ascii=False)}")

    assert result["conflict_resolved"] == True, "FAIL: Conflict should be detected!"
    assert result["tie_breaker"] is not None, "FAIL: Tie-breaker should have been called!"
    assert result["agreement_rate"] == 0.25, f"FAIL: Agreement rate should be 0.25, got {result['agreement_rate']}"

    print("[PASSED] Conflict detection test!")
    return result


# ============================================================================
# Test: Agreement (score_A=4, score_B=4 -> no conflict)
# ============================================================================

async def test_agreement():
    """Mock 2 Judge cho diem giong nhau -> khong trigger tie-breaker."""
    judge = LLMJudge()

    async def mock_judge(prompt, model):
        if model == "gpt-4o-mini":
            return {
                "accuracy": 4, "tone": 4,
                "reasoning": "Tot.", "tokens_used": 100, "model": "gpt-4o-mini"
            }
        else:
            return {
                "accuracy": 4, "tone": 5,
                "reasoning": "Rat tot.", "tokens_used": 80, "model": "gpt-4o"
            }

    judge._call_judge = mock_judge

    result = await judge.evaluate_multi_judge(
        question="Test?", answer="Answer.", ground_truth="Answer."
    )

    print("\n--- Test Agreement ---")
    print(f"Final Score: {result['final_score']}")
    print(f"Agreement Rate: {result['agreement_rate']}")
    print(f"Conflict Resolved: {result['conflict_resolved']}")

    assert result["conflict_resolved"] == False, "FAIL: Should NOT trigger conflict!"
    assert result["tie_breaker"] is None, "FAIL: Tie-breaker should NOT be called!"
    assert result["agreement_rate"] == 1.0, f"FAIL: Agreement should be 1.0, got {result['agreement_rate']}"
    assert result["final_score"] == 4.25, f"FAIL: Score should be 4.25, got {result['final_score']}"

    print("[PASSED] Agreement test!")


# ============================================================================
# Test: Result structure has all required fields
# ============================================================================

async def test_result_structure():
    """Kiem tra ket qua co du 6 truong bat buoc."""
    judge = LLMJudge()

    async def mock_judge(prompt, model):
        return {
            "accuracy": 3, "tone": 3,
            "reasoning": "OK.", "tokens_used": 50, "model": model
        }

    judge._call_judge = mock_judge

    result = await judge.evaluate_multi_judge("Q?", "A.", "GT.")

    required_fields = ["final_score", "agreement_rate", "individual_scores",
                       "total_tokens", "conflict_resolved", "tie_breaker"]

    for field in required_fields:
        assert field in result, f"FAIL: Missing required field: {field}"

    print("\n--- Test Result Structure ---")
    print(f"All 6 required fields present: {required_fields}")
    print("[PASSED] Structure test!")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("Running LLM Judge Tests...\n")

    asyncio.run(test_result_structure())
    asyncio.run(test_agreement())
    asyncio.run(test_conflict_detection())

    print("\nAll LLM Judge tests PASSED!")
