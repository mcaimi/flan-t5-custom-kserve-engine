#!/usr/bin/env python

import asyncio
import logging
import os
from typing import Dict, Union
from uuid import uuid4 as get_uuid

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from kserve import Model, InferRequest, InferResponse, InferOutput
from kserve.errors import InvalidInput
from .utils import get_accelerator_device
from .tasks import batch_transform_text, TASK_PREFIX_MAP

logger = logging.getLogger(__name__)

TASK_MAP: dict = set(TASK_PREFIX_MAP.keys())


class Seq2SeqModel(Model):
    def __init__(self, name: str, return_response_headers: bool = False):
        super().__init__(name)
        self.model_id = os.environ.get("MODEL_ID", default="/mnt/models")
        self.model = None
        self.tokenizer = None
        self.ready = False
        self.device = None
        self.dtype = None
        self.load()

    def load(self):
        self.device, self.dtype = get_accelerator_device()
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, torch_dtype=self.dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            raise RuntimeError(f"Failed loading model from {self.model_id}") from e

        self.model.to(self.device)
        self.ready = True

    # An example v1 JSON payload:
    #  {
    #    "instances": [
    #      {
    #        "task": "anonymize",
    #        "source": "text to be translated by the seq2seq model",
    #      }
    #    ]
    #  }
    #
    # An example v2 JSON payload:
    # {
    #  "inputs": [
    #    {
    #      "name": "anonymize",
    #      "shape": [1],
    #      "datatype": "BYTES",
    #      "data": ["Mi chiamo Marco e vivo a Milano"]
    #    }
    #  ]
    # }
    async def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"

            for pl in payload["instances"]:
                task_to_perform: str = pl.get("task")
                if task_to_perform not in TASK_MAP:
                    raise InvalidInput("Unsupported Task.")

        elif isinstance(payload, InferRequest):
            headers["request-type"] = "v2"

            for tsk in payload.inputs:
                if tsk.name not in TASK_MAP:
                    raise InvalidInput("Unsupported Task.")

        else:
            raise InvalidInput("invalid payload")

        return payload

    async def predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:
        req_uuid: str = f"{get_uuid()}"

        if isinstance(payload, Dict):
            pld = payload.get("instances")

            task_text_pairs = []
            metadata = []
            for task in pld:
                requested_task: str = task.get("task")
                source_text: str = task.get("source")
                task_text_pairs.append((TASK_PREFIX_MAP[requested_task], source_text))
                metadata.append((requested_task, source_text))

            target_texts = await asyncio.to_thread(
                batch_transform_text,
                task_text_pairs,
                model=self.model,
                tokenizer=self.tokenizer,
                accelerator=self.device,
            )

            results = [
                {
                    "task": meta[0],
                    "model_name": self.model_id,
                    "source": meta[1],
                    "target": target,
                }
                for meta, target in zip(metadata, target_texts)
            ]

            return {"request_id": req_uuid, "predictions": results}

        elif isinstance(payload, InferRequest):
            pld = payload.inputs

            task_text_pairs = []
            for item in pld:
                task_text_pairs.append((TASK_PREFIX_MAP[item.name], item.data[0]))

            target_texts = await asyncio.to_thread(
                batch_transform_text,
                task_text_pairs,
                model=self.model,
                tokenizer=self.tokenizer,
                accelerator=self.device,
            )

            results = [
                InferOutput(
                    name="result",
                    shape=[1],
                    datatype="BYTES",
                    data=[target],
                )
                for target in target_texts
            ]

            return InferResponse(
                response_id=req_uuid,
                model_name=self.name,
                infer_outputs=results,
            )
