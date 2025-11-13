#!/usr/bin/env python

# base libs
import os
from typing import Dict, Union

# import libraries
try:
    from uuid import uuid4 as get_uuid
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer
    )
    from kserve import Model, InferRequest, InferResponse, InferOutput
    from kserve.errors import InvalidInput
    from .utils import get_accelerator_device
    from .tasks import anonymize_text, translate_text, summarize_text
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    raise e

# task map
TASK_MAP: dict = {
    "anonymize": anonymize_text,
    "translate": translate_text,
    "summarize": summarize_text,
}

# Seq2Seq Model Serving Class
# instantiate this to perform text translation
class Seq2SeqModel(Model):
    # initialize class
    def __init__(self, name: str, return_response_headers: bool = False):
        super().__init__(name)
        self.model_id = os.environ.get("MODEL_ID", default="/mnt/models")
        # model checkpoint and tokenizer
        self.model = None
        self.tokenizer = None

        # health check
        self.ready = False
        # accelerator device
        self.device = None
        self.dtype = None
        
        # load model
        self.load()

    # load weights and instantiate pipeline
    def load(self):
        # detect accelerator
        self.device, self.dtype = get_accelerator_device()
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            raise Exception(f"Failed loading Model: {e}")

        # model model weights to accelerator if one is found
        self.model.to(self.device)

        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    # process incoming request payload.
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
    #   "inputs": [
    #       {
    #         "name": "anonymize",
    #         "shape": [1],
    #         "datatype": "BYTES",
    #         "data": ["Mi chiamo Marco Caimi e vivo a Ceriano Laghetto"]
    #       },
    #    ]
    # }
    # validate input request: requested task must be supported
    async def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Union[Dict, InferRequest]:
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"

            # extract payload
            payloads = payload["instances"]
            for pl in payloads:
                # get task
                task_to_perform: str = pl.get("task")

                # check available tasks..
                if task_to_perform not in TASK_MAP.keys():
                    raise InvalidInput("Unsupported Task.")
        # KServe InferRequest (v2)
        elif isinstance(payload, InferRequest):
            headers["request-type"] = "v2"
            
            # verify task
            for tsk in payload.inputs:
                parms: dict = { "task": tsk.name, "data": tsk.data[0] }

                print(parms)
                # check tasks
                if parms.get("task") not in TASK_MAP.keys():
                    raise InvalidInput("Unsupported Task.")

        else:
            # malformed or missing input payload
            raise InvalidInput("invalid payload")

        # return generation data
        return payload

    # perform a forward pass (inference) and return generated data
    async def predict(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        # run prediction on v1 and v2 api endpoints
        results: list = []
        # request uuid
        req_uuid: str = f"{get_uuid()}"
        if isinstance(payload, Dict): # v1 api call
            try:
                # extract instances
                pld = payload.get("instances")

                # iterate over tasks
                for task in pld:
                    # get source from request
                    requested_task: str = task.get("task")
                    source_text: str = task.get("source")

                    # run inference
                    target_text: str = TASK_MAP.get(requested_task)(
                        text = source_text,
                        model = self.model,
                        tokenizer = self.tokenizer,
                        accelerator = self.device
                    )

                    # push result
                    results.append({
                        "task": requested_task,
                        "model_name": self.model_id,
                        "source": source_text,
                        "target": target_text
                    })

            except Exception as e:  # error during generation
                results.append({
                    "task": requested_task,
                    "model_name": self.model_id,
                    "source": source_text,
                    "target": f"Error in inference: {e}"
                })

            # return payload
            return {
                "request_id": req_uuid,
                "predictions": results
            }
        elif isinstance(payload, InferRequest): # v2 api call
            try:
                # extract payloads
                pld = payload.inputs

                # run inference
                for item in pld:
                    requested_task: str = item.name
                    source_text: str = item.data[0]
                    
                    # run inference
                    target_text: str = TASK_MAP.get(requested_task)(
                        text = source_text,
                        model = self.model,
                        tokenizer = self.tokenizer,
                        accelerator = self.device
                    )

                    # append results
                    results.append(
                                InferOutput(
                                    name="result",
                                    shape=[1],
                                    datatype="BYTES",
                                    data=[f"{target_text}"]
                                )
                            )
                
                # return
                return InferResponse(
                            response_id=req_uuid,
                            model_name=self.name,
                            infer_outputs=results,
                        )
            except Exception as e:
                output = InferOutput(name="result",
                                     shape=[1],
                                     datatype="BYTES",
                                     data=[f"{e}"])
                return InferResponse(response_id=req_uuid, model_name=self.name, infer_outputs=[output])
                