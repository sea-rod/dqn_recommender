# utils/metrics.py
import numpy as np


def precision_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]
    hits = sum([1 for item in recommended_k if item in relevant_items])
    return hits / k


def dcg_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]
    return sum(
        [
            1 / np.log2(i + 2) if item in relevant_items else 0
            for i, item in enumerate(recommended_k)
        ]
    )


def ndcg_at_k(recommended_items, relevant_items, k):
    dcg = dcg_at_k(recommended_items, relevant_items, k)
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_items), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
