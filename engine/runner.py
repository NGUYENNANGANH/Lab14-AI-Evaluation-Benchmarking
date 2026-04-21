import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        # Cost & Performance tracking
        self.total_tokens = 0
        self.total_cost = 0.0

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time
        
        # 2. Chạy RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"], 
            response["answer"], 
            test_case["expected_answer"]
        )

        # 4. Extract token & cost từ Agent response
        metadata = response.get("metadata", {})
        tokens_used = metadata.get("tokens_used", 0)
        cost = metadata.get("cost_usd", 0.0)
        self.total_tokens += tokens_used
        self.total_cost += cost
        
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
            "cost_info": {
                "tokens_used": tokens_used,
                "cost_usd": cost
            }
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Dùng asyncio.Semaphore thay vì batch loop.
        Semaphore giới hạn tối đa batch_size task chạy đồng thời,
        nhưng khi 1 task xong, task mới chạy NGAY (không chờ cả batch).
        """
        # Reset tracking mỗi lần chạy
        self.total_tokens = 0
        self.total_cost = 0.0

        sem = asyncio.Semaphore(batch_size)

        async def run_with_sem(case):
            async with sem:
                return await self.run_single_test(case)

        tasks = [run_with_sem(case) for case in dataset]
        return list(await asyncio.gather(*tasks))

    def get_performance_summary(self, total_cases: int, total_time: float) -> dict:
        """Trả về dict performance/cost để nhúng vào summary.json."""
        return {
            "total_time_seconds": round(total_time, 2),
            "avg_latency_per_case": round(total_time / max(total_cases, 1), 3),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_per_eval_usd": round(self.total_cost / max(total_cases, 1), 6),
            "under_2_minutes": total_time < 120
        }


# ============================================================================
# Test truc tiep: python engine/runner.py
# ============================================================================
if __name__ == "__main__":
    import json
    import os
    import sys
    import io

    # Fix Unicode output on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Them thu muc goc vao path de import duoc agent, engine
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from dotenv import load_dotenv
    from agent.main_agent import MainAgent
    from engine.llm_judge import LLMJudge

    load_dotenv()

    # Evaluator gia lap (Buoc 1 se thay the bang RetrievalEvaluator that)
    class SimpleEvaluator:
        async def score(self, case, resp):
            return {
                "faithfulness": 0.9,
                "relevancy": 0.8,
                "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
            }

    # 3 test cases mau de kiem tra
    test_data = [
        {
            "question": "Lam the nao de doi mat khau tai khoan?",
            "expected_answer": "Vao Cai dat > Bao mat > Doi mat khau, nhap mat khau cu va moi.",
        },
        {
            "question": "Chinh sach hoan tien nhu the nao?",
            "expected_answer": "Hoan tien trong vong 7 ngay ke tu ngay mua, san pham chua su dung.",
        },
        {
            "question": "Hay viet mot bai tho ve tinh yeu.",
            "expected_answer": "Xin loi, toi chi ho tro cac cau hoi lien quan den dich vu.",
        }
    ]

    async def test_runner():
        print("[>>] Khoi dong test BenchmarkRunner voi 3 cases...\n")

        runner = BenchmarkRunner(MainAgent(), SimpleEvaluator(), LLMJudge())

        pipeline_start = time.perf_counter()
        results = await runner.run_all(test_data, batch_size=3)
        pipeline_time = time.perf_counter() - pipeline_start

        # In bang ket qua
        print("\n" + "=" * 90)
        print(f"{'BANG KET QUA BENCHMARK':^90}")
        print("=" * 90)
        print(f"{'#':<4} {'Cau hoi':<40} {'Score':>8} {'Hit Rate':>10} {'Cost($)':>12} {'Latency':>10}")
        print("-" * 90)

        for i, r in enumerate(results, 1):
            question = r["test_case"][:37] + "..." if len(r["test_case"]) > 37 else r["test_case"]
            score = r["judge"]["final_score"]
            hit_rate = r["ragas"]["retrieval"]["hit_rate"]
            cost = r["cost_info"]["cost_usd"]
            latency = r["latency"]
            status = "PASS" if r["status"] == "pass" else "FAIL"
            print(f"{i:<4} {question:<40} {score:>5.2f} {status:<4} {hit_rate:>9.1f} {cost:>12.6f} {latency:>9.2f}s")

        print("-" * 90)

        # In tong ket Performance & Cost
        perf = runner.get_performance_summary(len(results), pipeline_time)
        print(f"\n[TIME]   Tong thoi gian pipeline:  {perf['total_time_seconds']}s")
        print(f"[SPEED]  Avg latency/case:         {perf['avg_latency_per_case']}s")
        print(f"[TOKEN]  Tong tokens su dung:      {perf['total_tokens']:,}")
        print(f"[COST]   Tong chi phi:             ${perf['total_cost_usd']:.6f}")
        print(f"[EVAL]   Chi phi moi eval:         ${perf['cost_per_eval_usd']:.6f}")
        under2 = "[OK]" if perf['under_2_minutes'] else "[WARN]"
        print(f"{under2}    Hoan thanh duoi 2 phut:  {perf['under_2_minutes']}")

    asyncio.run(test_runner())
