#!/usr/bin/env python

import os

import torch

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "4"))
EARLY_STOPPING = os.environ.get("EARLY_STOPPING", "true").lower() == "true"

TASK_PREFIX_MAP: dict = {
    "anonymize": "anonymize",
    "translate": "translate English to Italian",
    "summarize": "summarize",
}


def batch_transform_text(task_text_pairs, model, tokenizer, max_length: int = MAX_LENGTH, truncation: bool = True, accelerator: str = "cpu"):
    if not task_text_pairs:
        return []

    input_texts = [f"{task}: {text}" for task, text in task_text_pairs]

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=truncation,
        padding=True,
    )
    inputs = {k: v.to(accelerator) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def transform_text(text, model, tokenizer, max_length: int = MAX_LENGTH, truncation: bool = True, accelerator: str = "cpu", task: str = "translate"):
    results = batch_transform_text(
        [(task, text)], model, tokenizer, max_length, truncation, accelerator
    )
    return results[0]


def anonymize_text(text, model, tokenizer, max_length: int = MAX_LENGTH, truncation: bool = True, accelerator: str = "cpu"):
    return transform_text(text, model, tokenizer, max_length, truncation, accelerator, task="anonymize")


def translate_text(text, model, tokenizer, max_length: int = MAX_LENGTH, truncation: bool = True, accelerator: str = "cpu"):
    return transform_text(text, model, tokenizer, max_length, truncation, accelerator, task="translate English to Italian")


def summarize_text(text, model, tokenizer, max_length: int = MAX_LENGTH, truncation: bool = True, accelerator: str = "cpu"):
    return transform_text(text, model, tokenizer, max_length, truncation, accelerator, task="summarize")
