# tests/test_config_builder.py
from autotrainer.core.interfaces import TaskSpec
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


def test_build_task_config_vl_model():
    """build_task_config should add freeze_config for VL models."""
    spec = TaskSpec(
        name="test-vl", model_name_or_path="Org/VLModel",
        model_family="VL", stage="VL-SFT", template="qwen3_vl",
        freeze_config="freeze_vision freeze_aligner", max_seq_len=32768,
    )
    cb = ConfigBuilder()
    config = cb.build_task_config(spec, "/data/train.jsonl", eval_data_path="/data/eval.jsonl")
    assert config["model"]["model_name_or_path"] == "Org/VLModel"
    assert config["model"]["stage"] == "VL-SFT"
    assert config["data"]["template"] == "qwen3_vl"
    assert config["data"]["max_seq_len"] == 32768
    assert config["data"]["train_dataset_path"] == "/data/train.jsonl"
    assert config["data"]["eval_dataset_path"] == "/data/eval.jsonl"
    assert config["finetuning"]["freeze_config"] == "freeze_vision freeze_aligner"


def test_build_task_config_llm_model():
    """build_task_config should NOT add freeze_config for LLM models."""
    spec = TaskSpec(
        name="test-llm", model_name_or_path="Org/LLMModel",
        model_family="LLM", stage="SFT", template="qwen3_nothink",
        max_seq_len=32768,
    )
    cb = ConfigBuilder()
    config = cb.build_task_config(spec, "/data/train.jsonl")
    assert config["model"]["stage"] == "SFT"
    assert config["data"]["template"] == "qwen3_nothink"
    assert "freeze_config" not in config["finetuning"]


def test_build_task_config_with_overrides():
    """Overrides should be merged on top of task defaults."""
    spec = TaskSpec(
        name="test", model_name_or_path="Org/Model", stage="SFT",
        model_family="LLM", template="llama3",
    )
    cb = ConfigBuilder()
    config = cb.build_task_config(
        spec, "/data/train.jsonl",
        overrides={"model": {"use_lora": True}, "finetuning": {"learning_rate": 5e-5}},
    )
    assert config["model"]["use_lora"] is True
    assert config["finetuning"]["learning_rate"] == 5e-5


def test_build_defaults_respects_finetuning_defaults():
    """TaskSpec.finetuning_defaults should override base defaults."""
    spec = TaskSpec(
        name="test", model_name_or_path="Org/Model", stage="SFT",
        model_family="LLM", template="qwen3_nothink",
        finetuning_defaults={"learning_rate": 3e-5, "sharding": "stage2"},
    )
    cb = ConfigBuilder()
    defaults = cb.build_defaults(spec)
    assert defaults["finetuning"]["learning_rate"] == 3e-5
    assert defaults["finetuning"]["sharding"] == "stage2"


def test_build_paddleocr_vl_config_deprecated():
    """The deprecated wrapper should still work and produce valid config."""
    cb = ConfigBuilder()
    config = cb.build_paddleocr_vl_config(
        model_path="/models/paddleocr-vl",
        train_data="/data/train.jsonl",
        lora=True, lora_rank=16,
    )
    assert config["model"]["use_lora"] is True
    assert config["model"]["lora_rank"] == 16
    assert config["data"]["train_dataset_path"] == "/data/train.jsonl"
    assert config["finetuning"]["freeze_config"] == "freeze_vision freeze_aligner"
