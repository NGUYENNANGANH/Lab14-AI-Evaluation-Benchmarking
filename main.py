import asyncio
import json
import os
import time
from dotenv import load_dotenv
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent

load_dotenv()

# Giả lập các components Expert
class ExpertEvaluator:
    async def score(self, case, resp): 
        # Giả lập tính toán Hit Rate và MRR
        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
        }

# MultiModelJudge đã được thay thế bằng LLMJudge thật (engine/llm_judge.py)


def decide_release(v1_summary, v2_summary):
    """
    Auto-gate cho regression testing.
    Điều kiện bắt buộc để APPROVE:
    1) Delta Judge score > 0
    2) V2 hit_rate > 0.8
    3) V2 agreement_rate > 0.7
    """
    v1_metrics = v1_summary["metrics"]
    v2_metrics = v2_summary["metrics"]

    delta_score = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    hit_rate_ok = v2_metrics["hit_rate"] > 0.8
    agreement_ok = v2_metrics["agreement_rate"] > 0.7

    approved = delta_score > 0 and hit_rate_ok and agreement_ok

    return {
        "approved": approved,
        "delta_score": delta_score,
        "hit_rate_ok": hit_rate_ok,
        "agreement_ok": agreement_ok,
    }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), LLMJudge())
    pipeline_start = time.perf_counter()
    results = await runner.run_all(dataset)
    pipeline_time = time.perf_counter() - pipeline_start

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        },
        "performance": runner.get_performance_summary(total, pipeline_time)
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    # Cờ mock phục vụ test gate: score tăng nhưng retrieval suy thoái.
    # Bật bằng: $env:MOCK_V2_GATE_TEST='1'; python main.py
    if os.getenv("MOCK_V2_GATE_TEST", "0") == "1":
        v2_summary["metrics"]["avg_score"] = v1_summary["metrics"]["avg_score"] + 0.2
        v2_summary["metrics"]["hit_rate"] = min(v1_summary["metrics"]["hit_rate"] - 0.2, 0.79)
        v2_summary["metrics"]["agreement_rate"] = max(v2_summary["metrics"]["agreement_rate"], 0.8)
        print("🧪 Đang dùng MOCK_V2_GATE_TEST: V2 score tăng nhưng hit_rate thấp để test gate.")

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    gate_result = decide_release(v1_summary, v2_summary)
    delta = gate_result["delta_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"V2 Hit Rate: {v2_summary['metrics']['hit_rate']:.2f} (yêu cầu > 0.8: {'✅' if gate_result['hit_rate_ok'] else '❌'})")
    print(f"V2 Agreement Rate: {v2_summary['metrics']['agreement_rate']:.2f} (yêu cầu > 0.7: {'✅' if gate_result['agreement_ok'] else '❌'})")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
