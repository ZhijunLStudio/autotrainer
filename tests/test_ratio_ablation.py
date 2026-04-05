from autotrainer.skills.data_ratio_ablation.handler import DataRatioAblationHandler


def test_generate_equal_ratios():
    handler = DataRatioAblationHandler()
    ratios = handler.generate_equal_ratios(3)
    assert len(ratios) == 3
    assert abs(sum(ratios) - 1.0) < 0.001


def test_generate_proportional_ratios():
    handler = DataRatioAblationHandler()
    ratios = handler.generate_proportional_ratios([10000, 5000, 1000])
    assert abs(sum(ratios) - 1.0) < 0.001
    assert ratios[0] > ratios[1] > ratios[2]


def test_generate_leave_one_out():
    handler = DataRatioAblationHandler()
    base_ratios = [0.5, 0.3, 0.2]
    names = ["ds1", "ds2", "ds3"]
    loo_configs = handler.generate_leave_one_out(base_ratios, names)
    assert len(loo_configs) == 3
    assert loo_configs[0]["excluded"] == "ds1"
    assert abs(sum(loo_configs[0]["ratios"]) - 1.0) < 0.001
    assert loo_configs[0]["ratios"][0] == 0.0  # excluded ds1


def test_generate_ratio_sweep():
    handler = DataRatioAblationHandler()
    sweep = handler.generate_ratio_sweep(top_n=2)
    assert len(sweep) >= 5
    for combo in sweep:
        assert abs(sum(combo) - 1.0) < 0.001


def test_generate_ratio_sweep_three_datasets():
    handler = DataRatioAblationHandler()
    sweep = handler.generate_ratio_sweep(top_n=3)
    assert len(sweep) >= 5
    for combo in sweep:
        assert len(combo) == 3
        assert abs(sum(combo) - 1.0) < 0.001


def test_compute_score():
    handler = DataRatioAblationHandler()
    baseline = {"eval_loss": 1.0, "throughput": 1000}
    result = {"eval_loss": 0.5, "throughput": 1200, "has_nan": False}
    score = handler.compute_score(result, baseline)
    assert score > 1.0  # better than baseline


def test_compute_score_with_nan():
    handler = DataRatioAblationHandler()
    baseline = {"eval_loss": 1.0, "throughput": 1000}
    result = {"eval_loss": 0.5, "throughput": 1200, "has_nan": True}
    score_nan = handler.compute_score(result, baseline)
    result_no_nan = {"eval_loss": 0.5, "throughput": 1200, "has_nan": False}
    score_no_nan = handler.compute_score(result_no_nan, baseline)
    assert score_nan < score_no_nan  # NaN penalty


def test_rank_datasets_by_importance():
    handler = DataRatioAblationHandler()
    baseline_score = 1.0
    loo_scores = {"ds1": 0.7, "ds2": 0.95, "ds3": 1.05}
    rankings = handler.rank_datasets_by_importance(baseline_score, loo_scores)
    # ds1: delta=0.30 → important
    # ds2: delta=0.05 → important (boundary)
    # ds3: delta=-0.05 → neutral (boundary)
    assert rankings[0][0] == "ds1"
    assert rankings[0][2] == "important"
    assert rankings[-1][0] == "ds3"


def test_rank_datasets_noise():
    handler = DataRatioAblationHandler()
    baseline_score = 1.0
    loo_scores = {"ds1": 0.5, "ds2": 1.1}
    rankings = handler.rank_datasets_by_importance(baseline_score, loo_scores)
    # ds2: delta=-0.1 → noise (removing it improved score)
    ds2 = [r for r in rankings if r[0] == "ds2"][0]
    assert ds2[2] == "noise"


def test_generate_saturation_configs():
    handler = DataRatioAblationHandler()
    configs = handler.generate_saturation_configs()
    assert configs == [0.25, 0.50, 0.75, 1.00]
