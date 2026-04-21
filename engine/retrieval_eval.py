from typing import List, Dict
import random


class RetrievalEvaluator:
    """
    Evaluator for Retrieval stage of a RAG pipeline.
    Computes Hit Rate and Mean Reciprocal Rank (MRR) to measure
    how well the retrieval component surfaces the correct documents.
    """

    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.

        Args:
            expected_ids: Danh sách ID tài liệu chứa đáp án đúng.
            retrieved_ids: Danh sách ID tài liệu mà retriever trả về (đã xếp hạng).
            top_k: Số lượng kết quả top-k để kiểm tra.

        Returns:
            1.0 nếu có ít nhất 1 expected_id nằm trong top_k retrieved_ids, ngược lại 0.0.
        """
        if not expected_ids:
            # Nếu không có expected_ids (out-of-context), coi là hit nếu retriever cũng không trả về gì
            return 1.0 if not retrieved_ids else 0.0

        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.

        Args:
            expected_ids: Danh sách ID tài liệu chứa đáp án đúng.
            retrieved_ids: Danh sách ID tài liệu mà retriever trả về (đã xếp hạng).

        Returns:
            Reciprocal rank (1/position) của kết quả khớp đầu tiên, hoặc 0.0 nếu không có.
        """
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _simulate_retrieval(self, test_case: Dict) -> List[str]:
        """
        Simulate a retrieval engine returning document IDs for a given question.
        In production, this would call your actual vector DB / retriever.

        The simulation adds realistic noise:
        - Correct docs appear with high probability
        - Some irrelevant docs are mixed in
        - Order is partially randomized
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])

        if not expected_ids:
            # Out-of-context / adversarial: retriever may return noise
            noise_ids = [f"doc_{random.randint(1, 22):03d}" for _ in range(random.randint(0, 3))]
            return noise_ids

        # Build a realistic retrieved list
        all_doc_ids = [f"doc_{i:03d}" for i in range(1, 23)]
        noise_pool = [d for d in all_doc_ids if d not in expected_ids]

        retrieved = []

        # Add expected docs with ~80% chance each, at varying positions
        for doc_id in expected_ids:
            if random.random() < 0.85:
                retrieved.append(doc_id)

        # Add 2-5 noise documents
        num_noise = random.randint(2, 5)
        noise = random.sample(noise_pool, min(num_noise, len(noise_pool)))

        # Interleave: expected docs tend to be ranked higher
        combined = retrieved + noise
        # Slight shuffle to add realism but keep expected docs biased toward top
        for i in range(len(combined)):
            if combined[i] in expected_ids and random.random() < 0.7:
                # Keep expected docs near the top
                continue
            swap_idx = random.randint(0, len(combined) - 1)
            combined[i], combined[swap_idx] = combined[swap_idx], combined[i]

        return combined[:7]  # Return top-7 results

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids'.
        Nếu có trường 'retrieved_ids', sẽ dùng trực tiếp.
        Nếu không, sẽ simulate retrieval.

        Returns:
            Dict chứa avg_hit_rate, avg_mrr, và breakdown theo type.
        """
        random.seed(42)  # Reproducibility

        hit_rates = []
        mrrs = []
        results_by_type: Dict[str, Dict[str, list]] = {}

        for case in dataset:
            expected_ids = case.get("expected_retrieval_ids", [])

            # Use provided retrieved_ids or simulate
            retrieved_ids = case.get("retrieved_ids", self._simulate_retrieval(case))

            # Calculate metrics
            hr = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            hit_rates.append(hr)
            mrrs.append(mrr)

            # Track by type
            case_type = case.get("metadata", {}).get("type", "unknown")
            if case_type not in results_by_type:
                results_by_type[case_type] = {"hit_rates": [], "mrrs": []}
            results_by_type[case_type]["hit_rates"].append(hr)
            results_by_type[case_type]["mrrs"].append(mrr)

        # Compute averages
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

        # Breakdown by type
        breakdown = {}
        for case_type, data in results_by_type.items():
            breakdown[case_type] = {
                "count": len(data["hit_rates"]),
                "avg_hit_rate": sum(data["hit_rates"]) / len(data["hit_rates"]),
                "avg_mrr": sum(data["mrrs"]) / len(data["mrrs"]),
            }

        return {
            "total_cases": len(dataset),
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "breakdown_by_type": breakdown,
            "all_hit_rates": hit_rates,
            "all_mrrs": mrrs,
        }
