# KServe Compatible Demo Model Server For Seq2Seq Models

This is a demo model server that can run inside Openshift AI as a KServe model server.
It exposes a way to perform inference with a Sequence to Sequence model (such as T5) running in the backend.

It is a companion project to the finetuning example found [here](https://github.com/mcaimi/flan-t5-finetune-ita)

## Parameters

The server expects a JSON-encoded payload to start inference. 

- For v1 compatible requests:

```json
 // example v1 payload:
 {
   "instances": [
     {
       "task": "anonymize",
       "source": "text string to be anonymized",
     }
   ]
 }
```


- For v2 compatible requests:
```json
{
  "inputs": [
    {
      "name": "anonymize",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["text string to be anonymized"]
    },
  ]
}
```

## How to run

```bash
$ uv sync
$ MODEL_ID="path to the finetuned checkpoint" uv run model.py --model_name flant5-finetuned 
```

