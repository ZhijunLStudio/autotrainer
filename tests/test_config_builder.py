# tests/test_config_builder.py
from autotrainer.pf_integration.config_builder import ConfigBuilder


def test_build_multi_dataset_config():
    cb = ConfigBuilder()
    datasets = [
        ("/data/ds1/train.jsonl", 0.5),
        ("/data/ds2/train.jsonl", 0.3),
        ("/data/ds3/train.jsonl", 0.2),
    ]
    config = cb.build_multi_dataset_config(
        model_path="/models/paddleocr-vl",
        datasets=datasets,
    )
    assert config["data"]["train_dataset_path"] == "/data/ds1/train.jsonl,/data/ds2/train.jsonl,/data/ds3/train.jsonl"
    assert config["data"]["train_dataset_prob"] == "0.5,0.3,0.2"


def test_build_multi_dataset_config_with_overrides():
    cb = ConfigBuilder()
    datasets = [("/data/ds1/train.jsonl", 0.5), ("/data/ds2/train.jsonl", 0.5)]
    config = cb.build_multi_dataset_config(
        model_path="/models/paddleocr-vl",
        datasets=datasets,
        overrides={"finetuning": {"learning_rate": 0.0001}},
    )
    assert config["finetuning"]["learning_rate"] == 0.0001
    assert config["data"]["train_dataset_prob"] == "0.5,0.5"
