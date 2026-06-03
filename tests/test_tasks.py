from unittest.mock import MagicMock, patch

import torch


def _make_mocks():
    mock_tokenizer = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tokenizer.return_value = {"input_ids": mock_tensor, "attention_mask": mock_tensor}
    mock_tokenizer.decode.return_value = "decoded output"
    mock_tokenizer.batch_decode.return_value = ["decoded output"]

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    return mock_model, mock_tokenizer


def test_transform_text_prepends_task_prefix():
    from libs.tasks import transform_text

    model, tokenizer = _make_mocks()
    result = transform_text("hello world", model, tokenizer, task="summarize")

    tokenizer.assert_called_once()
    call_args = tokenizer.call_args
    assert call_args[0][0] == ["summarize: hello world"]
    assert result == "decoded output"


def test_transform_text_uses_accelerator():
    from libs.tasks import transform_text

    model, tokenizer = _make_mocks()
    transform_text("test", model, tokenizer, accelerator="cuda")

    tensor = tokenizer.return_value["input_ids"]
    tensor.to.assert_called_with("cuda")


def test_batch_transform_text_batches_inputs():
    from libs.tasks import batch_transform_text

    model, tokenizer = _make_mocks()
    tokenizer.batch_decode.return_value = ["out1", "out2", "out3"]
    model.generate.return_value = torch.tensor([[1], [2], [3]])

    pairs = [
        ("anonymize", "text1"),
        ("translate English to Italian", "text2"),
        ("summarize", "text3"),
    ]
    results = batch_transform_text(pairs, model, tokenizer)

    tokenizer.assert_called_once()
    call_args = tokenizer.call_args
    assert call_args[0][0] == [
        "anonymize: text1",
        "translate English to Italian: text2",
        "summarize: text3",
    ]
    assert call_args[1]["padding"] is True
    model.generate.assert_called_once()
    assert results == ["out1", "out2", "out3"]


def test_batch_transform_text_empty_input():
    from libs.tasks import batch_transform_text

    model, tokenizer = _make_mocks()
    results = batch_transform_text([], model, tokenizer)

    assert results == []
    model.generate.assert_not_called()


def test_anonymize_text_uses_correct_prefix():
    from libs.tasks import anonymize_text

    model, tokenizer = _make_mocks()
    anonymize_text("sensitive data", model, tokenizer)

    call_args = tokenizer.call_args
    assert call_args[0][0] == ["anonymize: sensitive data"]


def test_translate_text_uses_correct_prefix():
    from libs.tasks import translate_text

    model, tokenizer = _make_mocks()
    translate_text("hello", model, tokenizer)

    call_args = tokenizer.call_args
    assert call_args[0][0] == ["translate English to Italian: hello"]


def test_summarize_text_uses_correct_prefix():
    from libs.tasks import summarize_text

    model, tokenizer = _make_mocks()
    summarize_text("long text", model, tokenizer)

    call_args = tokenizer.call_args
    assert call_args[0][0] == ["summarize: long text"]


def test_generation_params_from_env():
    with patch.dict("os.environ", {"NUM_BEAMS": "8", "EARLY_STOPPING": "false", "MAX_LENGTH": "256"}):
        import importlib
        import libs.tasks
        importlib.reload(libs.tasks)

        assert libs.tasks.NUM_BEAMS == 8
        assert libs.tasks.EARLY_STOPPING is False
        assert libs.tasks.MAX_LENGTH == 256

        model, tokenizer = _make_mocks()
        libs.tasks.transform_text("test", model, tokenizer)

        generate_kwargs = model.generate.call_args[1]
        assert generate_kwargs["num_beams"] == 8
        assert generate_kwargs["early_stopping"] is False
        assert generate_kwargs["max_length"] == 256

    # restore defaults
    import importlib
    import libs.tasks
    importlib.reload(libs.tasks)
