import asyncio
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Judge Prompt Template
# ============================================================================
JUDGE_PROMPT = """Bạn là một AI Judge chuyên nghiệp, đánh giá chất lượng câu trả lời của một AI Agent hỗ trợ khách hàng.

**Câu hỏi của người dùng:**
{question}

**Câu trả lời của Agent:**
{answer}

**Đáp án chuẩn (Ground Truth):**
{ground_truth}

Hãy chấm điểm theo 2 tiêu chí (thang 1-5):

1. **accuracy** (Độ chính xác):
   - 5: Hoàn toàn chính xác, đầy đủ thông tin so với Ground Truth
   - 4: Chính xác nhưng thiếu một vài chi tiết nhỏ
   - 3: Đúng ý chính nhưng thiếu nhiều chi tiết hoặc có sai sót nhỏ
   - 2: Chỉ đúng một phần, có sai sót đáng kể
   - 1: Hoàn toàn sai hoặc không liên quan

2. **tone** (Sự chuyên nghiệp):
   - 5: Cực kỳ chuyên nghiệp, lịch sự, rõ ràng
   - 4: Chuyên nghiệp, phù hợp
   - 3: Chấp nhận được nhưng có thể cải thiện
   - 2: Thiếu chuyên nghiệp hoặc quá cứng nhắc
   - 1: Không phù hợp, thô lỗ hoặc gây nhầm lẫn

Trả về KẾT QUẢ duy nhất theo định dạng JSON (KHÔNG kèm markdown):
{{"accuracy": <1-5>, "tone": <1-5>, "reasoning": "<giải thích ngắn gọn>"}}"""


class LLMJudge:
    """
    Multi-Judge Consensus Engine.
    Sử dụng ít nhất 2 model LLM để chấm điểm, tính toán độ đồng thuận,
    và tự động xử lý xung đột khi 2 Judge cho điểm chênh lệch lớn.
    """

    def __init__(self, model_a: str = "gpt-4o-mini", model_b: str = "claude-3-5-haiku-20241022"):
        self.model_a = model_a
        self.model_b = model_b
        self.conflict_threshold = 1  # Ngưỡng chênh lệch kích hoạt tie-breaker
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5 dựa trên độ chính xác so với Ground Truth.",
            "tone": "Chấm điểm từ 1-5 dựa trên sự chuyên nghiệp của ngôn ngữ."
        }
        self.total_tokens = 0

    # ========================================================================
    # Individual Judge Calls
    # ========================================================================

    async def _call_openai_judge(self, prompt: str) -> Dict[str, Any]:
        """Gọi OpenAI model để chấm điểm."""
        import openai

        client = openai.AsyncOpenAI()
        try:
            response = await client.chat.completions.create(
                model=self.model_a,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=300
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            tokens = response.usage.total_tokens if response.usage else 0

            result["accuracy"] = max(1, min(5, int(result.get("accuracy", 3))))
            result["tone"] = max(1, min(5, int(result.get("tone", 3))))
            result["tokens_used"] = tokens
            result["model"] = self.model_a
            return result

        except Exception as e:
            print(f"  ⚠️ OpenAI Judge error: {e}")
            return {
                "accuracy": 3, "tone": 3,
                "reasoning": f"Error: {str(e)}",
                "tokens_used": 0, "model": self.model_a
            }

    async def _call_anthropic_judge(self, prompt: str) -> Dict[str, Any]:
        """Gọi Anthropic Claude model để chấm điểm."""
        import anthropic

        client = anthropic.AsyncAnthropic()
        try:
            response = await client.messages.create(
                model=self.model_b,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text

            # Claude đôi khi wrap JSON trong markdown code blocks
            if "```" in content:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                result = json.loads(json_str.strip())
            else:
                result = json.loads(content)

            tokens = response.usage.input_tokens + response.usage.output_tokens

            result["accuracy"] = max(1, min(5, int(result.get("accuracy", 3))))
            result["tone"] = max(1, min(5, int(result.get("tone", 3))))
            result["tokens_used"] = tokens
            result["model"] = self.model_b
            return result

        except Exception as e:
            print(f"  ⚠️ Anthropic Judge error: {e}")
            return {
                "accuracy": 3, "tone": 3,
                "reasoning": f"Error: {str(e)}",
                "tokens_used": 0, "model": self.model_b
            }

    # ========================================================================
    # Conflict Resolution (Tie-Breaker)
    # ========================================================================

    async def _resolve_conflict(self, prompt: str, score_a: Dict, score_b: Dict) -> Dict[str, Any]:
        """
        Tie-breaker: Khi 2 Judge xung đột (chênh > 1 điểm accuracy),
        gọi Judge thứ 3 với context của 2 Judge trước, lấy median.
        """
        diff = abs(score_a["accuracy"] - score_b["accuracy"])
        print(f"  ⚠️ XUNG ĐỘT: {score_a['model']}={score_a['accuracy']}, "
              f"{score_b['model']}={score_b['accuracy']} (chênh={diff})")

        tie_prompt = f"""{prompt}

LƯU Ý: Hai Judge trước đã chấm điểm như sau:
- Judge A ({score_a['model']}): accuracy={score_a['accuracy']}, lý do: {score_a.get('reasoning', 'N/A')}
- Judge B ({score_b['model']}): accuracy={score_b['accuracy']}, lý do: {score_b.get('reasoning', 'N/A')}

Hãy phân tích cả hai quan điểm và đưa ra đánh giá công tâm cuối cùng."""

        tie_result = await self._call_openai_judge(tie_prompt)

        # Lấy median của 3 điểm
        acc_scores = sorted([score_a["accuracy"], score_b["accuracy"], tie_result["accuracy"]])
        tone_scores = sorted([score_a["tone"], score_b["tone"], tie_result["tone"]])

        print(f"  ✅ TIE-BREAKER: chấm accuracy={tie_result['accuracy']} "
              f"→ Median={acc_scores[1]}")

        return {
            "resolved_accuracy": acc_scores[1],
            "resolved_tone": tone_scores[1],
            "tie_breaker_score": tie_result,
            "tie_breaker_tokens": tie_result["tokens_used"]
        }

    # ========================================================================
    # Main Evaluation Method
    # ========================================================================

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Multi-Judge Consensus: Gọi 2 model Judge song song,
        tính Agreement Rate, và xử lý xung đột tự động.
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )

        # 1. Gọi 2 Judge song song
        score_a, score_b = await asyncio.gather(
            self._call_openai_judge(prompt),
            self._call_anthropic_judge(prompt)
        )

        # 2. Tính Agreement Rate (1.0 = đồng ý hoàn toàn, 0.0 = bất đồng hoàn toàn)
        accuracy_diff = abs(score_a["accuracy"] - score_b["accuracy"])
        agreement_rate = 1.0 - (accuracy_diff / 4.0)

        total_tokens = score_a["tokens_used"] + score_b["tokens_used"]
        conflict_resolved = False
        tie_breaker_info = None

        # 3. Tính final scores (trung bình)
        final_accuracy = (score_a["accuracy"] + score_b["accuracy"]) / 2
        final_tone = (score_a["tone"] + score_b["tone"]) / 2

        # 4. Xử lý xung đột nếu chênh accuracy > threshold
        if accuracy_diff > self.conflict_threshold:
            conflict = await self._resolve_conflict(prompt, score_a, score_b)
            final_accuracy = conflict["resolved_accuracy"]
            final_tone = conflict["resolved_tone"]
            total_tokens += conflict["tie_breaker_tokens"]
            conflict_resolved = True
            tie_breaker_info = {
                "model": conflict["tie_breaker_score"]["model"],
                "accuracy": conflict["tie_breaker_score"]["accuracy"],
                "tone": conflict["tie_breaker_score"]["tone"],
                "reasoning": conflict["tie_breaker_score"].get("reasoning", "")
            }

        final_score = (final_accuracy + final_tone) / 2
        self.total_tokens += total_tokens

        return {
            "final_score": final_score,
            "agreement_rate": round(agreement_rate, 4),
            "individual_scores": {
                score_a["model"]: {
                    "accuracy": score_a["accuracy"],
                    "tone": score_a["tone"],
                    "reasoning": score_a.get("reasoning", "")
                },
                score_b["model"]: {
                    "accuracy": score_b["accuracy"],
                    "tone": score_b["tone"],
                    "reasoning": score_b.get("reasoning", "")
                }
            },
            "total_tokens": total_tokens,
            "conflict_resolved": conflict_resolved,
            "tie_breaker": tie_breaker_info
        }

    # ========================================================================
    # Position Bias Check (Nâng cao)
    # ========================================================================

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        """
        pass
