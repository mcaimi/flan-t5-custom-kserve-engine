# FLAN-T5 Custom KServe Engine

A custom [KServe](https://kserve.github.io/website/) model server for sequence-to-sequence (seq2seq) models such as FLAN-T5. It is designed to run on [OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai) as a custom serving runtime and exposes REST endpoints for text transformation tasks.

This project is a companion to the [FLAN-T5 fine-tuning example](https://github.com/mcaimi/flan-t5-finetune-ita): train a model there, then serve it here.

## Features

- **KServe REST protocols** — supports both v1 (`/v1/models/{model_name}:predict`) and v2 (`/v2/models/{model_name}/infer`) request formats
- **Multi-task inference** — a single model can handle multiple task types via instruction prefixes:
  - `anonymize` — redact or anonymize sensitive text
  - `translate` — translate English to Italian
  - `summarize` — produce a summary of the input text
- **Batch processing** — multiple instances can be submitted in one request and are processed together
- **Hardware acceleration** — automatically selects the best available device (CUDA GPU, Apple MPS, or CPU) with appropriate precision
- **OpenShift AI integration** — includes a `ServingRuntime` manifest for deployment on OpenShift AI Model Serving
- **Container-ready** — ships with a `Containerfile` for building a deployable image

## Project Structure

```
.
├── model.py                    # Entry point — starts the KServe ModelServer
├── libs/
│   ├── transformer_class.py    # Seq2SeqModel: KServe Model implementation (load, preprocess, predict)
│   ├── tasks.py                # Task prefix mapping and text generation logic
│   └── utils.py                # Accelerator detection (CUDA / MPS / CPU)
├── manifests/
│   └── serving-runtime.yaml    # OpenShift AI ServingRuntime definition
├── tests/                      # Unit tests (pytest)
├── Containerfile               # Container image build definition
├── pyproject.toml              # Project metadata and dependencies (uv)
└── requirements.txt            # Pinned dependencies for container builds
```

### Key Components

| File | Role |
|------|------|
| `model.py` | Parses CLI arguments and starts the KServe `ModelServer` with a `Seq2SeqModel` instance |
| `libs/transformer_class.py` | Loads the Hugging Face model and tokenizer, validates incoming payloads, and runs inference |
| `libs/tasks.py` | Maps task names to FLAN-T5 instruction prefixes and performs batched tokenization and generation |
| `libs/utils.py` | Detects available hardware and selects the appropriate PyTorch device and dtype |
| `manifests/serving-runtime.yaml` | Registers this engine as a custom `ServingRuntime` in OpenShift AI |

## Configuration

The server is configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `/mnt/models` | Path or Hugging Face ID of the model checkpoint |
| `MAX_LENGTH` | `512` | Maximum token length for input and output |
| `NUM_BEAMS` | `4` | Number of beams for beam search during generation |
| `EARLY_STOPPING` | `true` | Stop beam search when all beams reach an end token |

## API

The server expects a JSON-encoded payload. The task name selects which instruction prefix is prepended to the source text before generation.

### v1 Request

```json
{
  "instances": [
    {
      "task": "anonymize",
      "source": "text string to be anonymized"
    }
  ]
}
```

### v2 Request

```json
{
  "inputs": [
    {
      "name": "anonymize",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["text string to be anonymized"]
    }
  ]
}
```

In v2 requests, the input `name` field carries the task name (e.g. `anonymize`, `translate`, `summarize`).

## Running Locally

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Start the server, pointing `MODEL_ID` at your fine-tuned checkpoint:

```bash
MODEL_ID="path/to/finetuned-checkpoint" uv run model.py --model_name flant5-finetuned
```

The server listens on port 8080 by default.

## Running Tests

```bash
uv run pytest
```

## Deploying on OpenShift AI

1. Build and push the container image:

   ```bash
   podman build -f Containerfile -t quay.io/<your-org>/kserve-seq2seq:latest .
   podman push quay.io/<your-org>/kserve-seq2seq:latest
   ```

2. Apply the serving runtime manifest (update the image reference first):

   ```bash
   oc apply -f manifests/serving-runtime.yaml
   ```

3. Create an `InferenceService` that references the `kserve-seq2seq` runtime and mounts your model at `/mnt/models`.

The manifest registers the runtime with OpenShift AI under the display name **KServe Custom Seq2Seq Engine**, supporting REST protocol versions v1 and v2 with NVIDIA GPU acceleration.
