import unittest

from eval.metrics import aggregate_scores, score_query
from eval.rag_eval import score_dataset


class RetrievalMetricsTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "qid": "q1",
            "category": "cpu",
            "relevant": [
                {
                    "source": "aiops-docs/cpu_high_usage.md",
                    "h2": "告警名称",
                    "grade": 2,
                    "evidence_quote": "CPU usage above 80 percent",
                },
                {
                    "source": "aiops-docs/cpu_high_usage.md",
                    "h2": "紧急处理措施",
                    "grade": 1,
                    "evidence_quote": "restart the affected instance",
                },
            ],
        }

    def test_multi_positive_recall_mrr_and_ndcg(self):
        candidates = [
            {
                "rank": 1,
                "content": "unrelated",
                "metadata": {"_source": "aiops-docs/disk_high_usage.md"},
            },
            {
                "rank": 2,
                "content": "CPU usage above 80 percent",
                "metadata": {"_source": "aiops-docs/cpu_high_usage.md"},
            },
            {
                "rank": 3,
                "content": "restart the affected instance",
                "metadata": {"_source": "aiops-docs/cpu_high_usage.md"},
            },
        ]
        result = score_query(self.row, candidates, k=3)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)
        self.assertEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["ndcg"], 0.659, places=3)
        self.assertEqual(result["doc_recall"], 1.0)

    def test_ndcg_uses_the_ideal_grades_within_k(self):
        row = {
            "qid": "q-ndcg",
            "category": "cpu",
            "relevant": [
                {"source": "doc.md", "h2": "a", "grade": 3, "evidence_quote": "gold-a"},
                {"source": "doc.md", "h2": "b", "grade": 2, "evidence_quote": "gold-b"},
                {"source": "doc.md", "h2": "c", "grade": 1, "evidence_quote": "gold-c"},
            ],
        }
        result = score_query(
            row,
            [{"rank": 1, "content": "gold-a", "metadata": {"_source": "doc.md"}}],
            k=1,
        )
        self.assertEqual(result["ndcg"], 1.0)

    def test_evidence_from_the_same_section_is_one_ranking_unit(self):
        row = {
            "qid": "q-unit",
            "category": "service_unavailable",
            "relevant": [
                {
                    "source": "doc.md",
                    "h2": "相关告警",
                    "grade": 2,
                    "evidence_quote": "HealthCheckFailed",
                },
                {
                    "source": "doc.md",
                    "h2": "相关告警",
                    "grade": 2,
                    "evidence_quote": "AllInstancesDown",
                },
            ],
        }
        result = score_query(
            row,
            [
                {"rank": 1, "content": "HealthCheckFailed", "metadata": {"_source": "doc.md"}},
                {"rank": 2, "content": "AllInstancesDown", "metadata": {"_source": "doc.md"}},
            ],
            k=3,
        )
        self.assertEqual(result["relevant_unit_count"], 1)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["ndcg"], 1.0)

    def test_negative_query_counts_returned_documents(self):
        row = {"qid": "n1", "category": "negative", "relevant": []}
        result = score_query(
            row,
            [{"rank": 1, "content": "anything", "metadata": {}}],
            k=3,
        )
        self.assertEqual(result["no_answer_return"], 1.0)
        self.assertEqual(result["empty_result"], 0.0)

    def test_document_recall_does_not_require_evidence_match(self):
        result = score_query(
            self.row,
            [
                {
                    "rank": 1,
                    "content": "same source but different section",
                    "metadata": {"_source": "aiops-docs/cpu_high_usage.md"},
                }
            ],
            k=3,
        )
        self.assertEqual(result["doc_recall"], 1.0)
        self.assertEqual(result["recall"], 0.0)

    def test_aggregate_ignores_missing_negative_metrics(self):
        positive = score_query(self.row, [], k=3)
        negative = score_query({"qid": "n1", "category": "negative", "relevant": []}, [], k=3)
        summary = aggregate_scores([positive, negative])
        self.assertEqual(summary["positive_queries"], 1)
        self.assertEqual(summary["negative_queries"], 1)
        self.assertEqual(summary["chunk_recall"], 0.0)
        self.assertEqual(summary["no_answer_return_rate"], 0.0)

    def test_missing_retrieval_results_fail_closed(self):
        with self.assertRaises(ValueError):
            score_dataset([self.row], {}, k=3)

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            score_query(self.row, [], k=0)


if __name__ == "__main__":
    unittest.main()
