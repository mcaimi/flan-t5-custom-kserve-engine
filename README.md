# KServe Compatible Demo Model Server For Seq2Seq Models

This is a demo model server that can run inside Openshift AI as a KServe model server.
It exposes a way to perform inference with a Sequence to Sequence model (such as T5) running in the backend.

It is a companion project to the finetuning example found [here](https://github.com/mcaimi/flan-t5-finetune-ita)

Of the many tasks that can be implemented using such a model, we only support now the "anonymize" task (which is described in the companion repo)

## Parameters

The server expects a JSON-encoded payload to start inference:

```json
 // example payload:
 {
   "instances": [
     {
       "task": "anonymize",
       "source": "text string to be anonymized",
     }
   ]
 }
```

## How to run

```bash
$ uv sync
$ MODEL_ID="path to the finetuned checkpoint" uv run model.py
```

