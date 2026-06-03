from unittest.mock import MagicMock, patch

import pytest
from kserve import InferRequest, InferInput
from kserve.errors import InvalidInput

from libs.transformer_class import Seq2SeqModel


@pytest.fixture
def seq2seq_model():
    with patch.object(Seq2SeqModel, "load"):
        model = Seq2SeqModel("test-model")
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.device = "cpu"
        model.dtype = None
        model.ready = True
        return model


async def test_preprocess_v1_valid(seq2seq_model):
    payload = {"instances": [{"task": "anonymize", "source": "test text"}]}
    headers = {}
    result = await seq2seq_model.preprocess(payload, headers)
    assert result == payload
    assert headers["request-type"] == "v1"


async def test_preprocess_v1_invalid_task(seq2seq_model):
    payload = {"instances": [{"task": "nonexistent", "source": "text"}]}
    with pytest.raises(InvalidInput):
        await seq2seq_model.preprocess(payload, {})


async def test_preprocess_v2_valid(seq2seq_model):
    infer_input = InferInput(name="summarize", shape=[1], datatype="BYTES", data=["text"])
    payload = InferRequest(model_name="test", infer_inputs=[infer_input])
    headers = {}
    result = await seq2seq_model.preprocess(payload, headers)
    assert headers["request-type"] == "v2"
    assert result is payload


async def test_preprocess_v2_invalid_task(seq2seq_model):
    infer_input = InferInput(name="nonexistent", shape=[1], datatype="BYTES", data=["text"])
    payload = InferRequest(model_name="test", infer_inputs=[infer_input])
    with pytest.raises(InvalidInput):
        await seq2seq_model.preprocess(payload, {})


async def test_preprocess_invalid_payload(seq2seq_model):
    with pytest.raises(InvalidInput):
        await seq2seq_model.preprocess({}, {})


@patch("libs.transformer_class.batch_transform_text", return_value=["anonymized result"])
async def test_predict_v1(mock_batch, seq2seq_model):
    payload = {"instances": [{"task": "anonymize", "source": "my name is Marco"}]}
    result = await seq2seq_model.predict(payload, {})

    assert "request_id" in result
    assert len(result["predictions"]) == 1
    pred = result["predictions"][0]
    assert pred["task"] == "anonymize"
    assert pred["source"] == "my name is Marco"
    assert pred["target"] == "anonymized result"


@patch("libs.transformer_class.batch_transform_text", return_value=["translated result"])
async def test_predict_v2(mock_batch, seq2seq_model):
    infer_input = InferInput(name="translate", shape=[1], datatype="BYTES", data=["hello world"])
    payload = InferRequest(model_name="test", infer_inputs=[infer_input])
    result = await seq2seq_model.predict(payload, {})

    assert result.model_name == "test-model"
    assert len(result.outputs) == 1
    assert result.outputs[0].data == ["translated result"]


@patch("libs.transformer_class.batch_transform_text", return_value=["anon result", "trans result", "summ result"])
async def test_predict_v1_multiple_tasks(mock_batch, seq2seq_model):
    payload = {
        "instances": [
            {"task": "anonymize", "source": "text1"},
            {"task": "translate", "source": "text2"},
            {"task": "summarize", "source": "text3"},
        ]
    }
    result = await seq2seq_model.predict(payload, {})

    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["task"] == "anonymize"
    assert result["predictions"][0]["target"] == "anon result"
    assert result["predictions"][1]["task"] == "translate"
    assert result["predictions"][1]["target"] == "trans result"
    assert result["predictions"][2]["task"] == "summarize"
    assert result["predictions"][2]["target"] == "summ result"

    pairs = mock_batch.call_args[0][0]
    assert pairs[0] == ("anonymize", "text1")
    assert pairs[1] == ("translate English to Italian", "text2")
    assert pairs[2] == ("summarize", "text3")
