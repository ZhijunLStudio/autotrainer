# tests/test_log_parser.py
from autotrainer.pf_integration.log_parser import LogParser, LogMetrics

def test_parse_eval_ppl():
    parser = LogParser()
    m = parser.parse_line("eval_loss: 1.234, eval_ppl: 3.435")
    assert m is not None
    assert m.eval_loss == 1.234
    assert m.eval_ppl == 3.435

def test_parse_generic_eval_metrics():
    parser = LogParser()
    m = parser.parse_line("eval_loss: 0.5, eval_accuracy: 0.92, eval_bleu: 45.6, global_step: 100")
    assert m is not None
    assert m.extra_metrics == {"eval_accuracy": 0.92, "eval_bleu": 45.6}

def test_parse_no_eval_metrics():
    parser = LogParser()
    m = parser.parse_line("some random log line")
    assert m is None

def test_has_nan_detection():
    parser = LogParser()
    m = parser.parse_line("global_step: 50, loss: nan")
    assert m is not None
    assert m.has_nan is True

def test_has_nan_detection_inf():
    parser = LogParser()
    m = parser.parse_line("global_step: 50, loss: inf")
    assert m is not None
    assert m.has_nan is True
