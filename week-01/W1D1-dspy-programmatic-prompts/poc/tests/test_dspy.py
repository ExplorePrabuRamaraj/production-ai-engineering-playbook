"""
Tests for W1D1 — DSPy & Programmatic Prompts

Covers: QASignature, Prediction, Predict, ChainOfThought,
        BootstrapFewShot, accuracy_metric, and the demo pipeline.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dspy_core import (
    QASignature,
    Prediction,
    Predict,
    ChainOfThought,
    BootstrapFewShot,
    Example,
    DEMO_RESULTS,
    TRAINSET,
    accuracy_metric,
    run_demo_pipeline,
)


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class TestQASignature:
    def test_has_question_input_field(self) -> None:
        assert "question" in QASignature.input_fields

    def test_has_rationale_output_field(self) -> None:
        assert "rationale" in QASignature.output_fields

    def test_has_answer_output_field(self) -> None:
        assert "answer" in QASignature.output_fields

    def test_instructions_are_non_empty(self) -> None:
        assert len(QASignature.instructions) > 0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_default_fields_are_empty_strings(self) -> None:
        pred = Prediction()
        assert pred.rationale == ""
        assert pred.answer == ""

    def test_fields_set_via_constructor(self) -> None:
        pred = Prediction(rationale="Some reasoning", answer="Final answer")
        assert pred.rationale == "Some reasoning"
        assert pred.answer == "Final answer"


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_prediction_instance(self) -> None:
        predictor = Predict(QASignature)
        result = predictor(question=DEMO_RESULTS[0]["question"])
        assert isinstance(result, Prediction)

    def test_answer_is_non_empty(self) -> None:
        predictor = Predict(QASignature)
        result = predictor(question=DEMO_RESULTS[1]["question"])
        assert len(result.answer) > 0

    def test_demos_default_to_empty_list(self) -> None:
        predictor = Predict(QASignature)
        assert predictor.demos == []

    def test_custom_lm_fn_is_used(self) -> None:
        def fake_lm(question: str, demos: list) -> Prediction:
            return Prediction(rationale="custom reasoning", answer="custom answer")

        predictor = Predict(QASignature)
        result = predictor(question="anything", _lm_fn=fake_lm)
        assert result.answer == "custom answer"


# ---------------------------------------------------------------------------
# ChainOfThought
# ---------------------------------------------------------------------------

class TestChainOfThought:
    def test_returns_prediction_with_rationale(self) -> None:
        cot = ChainOfThought(QASignature)
        result = cot(question=DEMO_RESULTS[0]["question"])
        assert isinstance(result, Prediction)
        assert len(result.rationale) > 0

    def test_is_subclass_of_predict(self) -> None:
        assert issubclass(ChainOfThought, Predict)


# ---------------------------------------------------------------------------
# BootstrapFewShot
# ---------------------------------------------------------------------------

class TestBootstrapFewShot:
    def test_returns_compiled_predictor(self) -> None:
        cot = ChainOfThought(QASignature)
        teleprompter = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=2)
        compiled = teleprompter.compile(cot, trainset=TRAINSET)
        assert isinstance(compiled, Predict)

    def test_compiled_demos_do_not_exceed_max(self) -> None:
        cot = ChainOfThought(QASignature)
        teleprompter = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=2)
        compiled = teleprompter.compile(cot, trainset=TRAINSET)
        assert len(compiled.demos) <= 2

    def test_demos_have_required_keys(self) -> None:
        cot = ChainOfThought(QASignature)
        teleprompter = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=3)
        compiled = teleprompter.compile(cot, trainset=TRAINSET)
        for demo in compiled.demos:
            assert "question" in demo
            assert "rationale" in demo
            assert "answer" in demo

    def test_zero_max_demos_produces_empty_list(self) -> None:
        cot = ChainOfThought(QASignature)
        teleprompter = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=0)
        compiled = teleprompter.compile(cot, trainset=TRAINSET)
        assert len(compiled.demos) == 0


# ---------------------------------------------------------------------------
# Metric function
# ---------------------------------------------------------------------------

class TestAccuracyMetric:
    @pytest.mark.parametrize("expected,predicted,should_pass", [
        (
            "Zero-shot uses no examples.",
            "Zero-shot uses no examples and relies on pre-trained knowledge.",
            True,
        ),
        (
            "Something totally different.",
            "A completely unrelated answer about something else.",
            False,
        ),
        (
            "Signatures decouple program logic from prompt text.",
            "Signatures decouple program logic from prompt text, enabling auto-optimization.",
            True,
        ),
    ])
    def test_metric_scoring(self, expected: str, predicted: str, should_pass: bool) -> None:
        example = Example(question="Q", answer=expected)
        pred = Prediction(rationale="", answer=predicted)
        assert accuracy_metric(example, pred) == should_pass


# ---------------------------------------------------------------------------
# Demo pipeline integration
# ---------------------------------------------------------------------------

class TestDemoPipeline:
    def test_returns_correct_number_of_results(self) -> None:
        results = run_demo_pipeline()
        assert len(results) == len(DEMO_RESULTS)

    def test_each_result_has_required_keys(self) -> None:
        results = run_demo_pipeline()
        for r in results:
            assert "question" in r
            assert "rationale" in r
            assert "answer" in r
            assert "num_demos" in r

    def test_answers_are_non_empty(self) -> None:
        results = run_demo_pipeline()
        for r in results:
            assert len(r["answer"]) > 0
