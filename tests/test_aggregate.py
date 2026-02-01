from src.aggregate import cons_at_k, filter_top_eta, measure_at_k, measure_top_eta_at_k


def test_majority_vote_clear_winner() -> None:
    samples = [
        {"sample_id": 1, "answer": 5, "confidence": 1.0},
        {"sample_id": 2, "answer": 5, "confidence": 0.5},
        {"sample_id": 3, "answer": 7, "confidence": 10.0},
    ]
    result = cons_at_k(samples)
    assert result.answer == 5
    assert not result.tie


def test_majority_vote_tie_breaks_smallest() -> None:
    samples = [
        {"sample_id": 1, "answer": 5, "confidence": 1.0},
        {"sample_id": 2, "answer": 7, "confidence": 1.0},
    ]
    result = cons_at_k(samples)
    assert result.answer == 5
    assert result.tie


def test_majority_vote_ignores_invalid() -> None:
    samples = [
        {"sample_id": 1, "answer": None, "confidence": 1.0},
        {"sample_id": 2, "answer": 3, "confidence": 1.0},
    ]
    result = cons_at_k(samples)
    assert result.answer == 3
    assert result.valid_count == 1


def test_majority_vote_zero_valid_answers() -> None:
    samples = [
        {"sample_id": 1, "answer": None, "confidence": 1.0},
        {"sample_id": 2, "answer": None, "confidence": 2.0},
    ]
    result = cons_at_k(samples)
    assert result.answer is None
    assert result.valid_count == 0


def test_weighted_vote_prefers_high_confidence() -> None:
    samples = [
        {"sample_id": 1, "answer": 5, "confidence": 1.0},
        {"sample_id": 2, "answer": 7, "confidence": 3.0},
        {"sample_id": 3, "answer": 5, "confidence": 1.0},
    ]
    result = measure_at_k(samples)
    assert result.answer == 7
    assert result.votes[7] == 3.0
    assert result.votes[5] == 2.0


def test_weighted_vote_tie_breaks_smallest() -> None:
    samples = [
        {"sample_id": 1, "answer": 5, "confidence": 2.0},
        {"sample_id": 2, "answer": 7, "confidence": 2.0},
    ]
    result = measure_at_k(samples)
    assert result.answer == 5
    assert result.tie


def test_filter_top_eta_keeps_ceil_fraction() -> None:
    samples = [
        {"sample_id": i, "answer": i, "confidence": float(i)} for i in range(1, 6)
    ]
    kept = filter_top_eta(samples, eta=0.4)
    # ceil(0.4 * 5) == 2
    assert len(kept) == 2
    assert [item["sample_id"] for item in kept] == [5, 4]


def test_filter_top_eta_handles_small_valid_counts() -> None:
    samples = [{"sample_id": 1, "answer": 9, "confidence": 0.1}]
    kept = filter_top_eta(samples, eta=0.1)
    assert len(kept) == 1


def test_filter_top_eta_handles_zero_valid() -> None:
    samples = [{"sample_id": 1, "answer": None, "confidence": 0.1}]
    kept = filter_top_eta(samples, eta=0.5)
    assert kept == []


def test_measure_top_eta_runs_weighted_vote_after_filtering() -> None:
    samples = [
        {"sample_id": 1, "answer": 5, "confidence": 0.1},
        {"sample_id": 2, "answer": 7, "confidence": 10.0},
        {"sample_id": 3, "answer": 5, "confidence": 0.2},
    ]
    result = measure_top_eta_at_k(samples, eta=1 / 3)
    assert result.answer == 7