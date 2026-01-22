#!/usr/bin/env python

try:
    import torch
except Exception as e:
    raise e


# transform text
def transform_text(text, model, tokenizer, max_length: int = 512, truncation: bool = True, accelerator: str = "cpu", task: str = "translate"):
    """
        call a specific model task
    """
    # Prepare input
    input_text = f"{task}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=truncation)
    
    # move to device
    inputs = {k: v.to(accelerator) for k, v in inputs.items()}
    model = model.to(accelerator)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    # Decode output
    transformed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transformed

# transform function
# ANONYMIZE
def anonymize_text(text, model, tokenizer, max_length: int = 512, truncation: bool = True, accelerator: str = "cpu"):
    """
        Anonymize PII in Italian text using the fine-tuned model
    """
    return transform_text(text, model, tokenizer, max_length, truncation, accelerator, task="anonymize")

# transform function
# TRANSLATE
def translate_text(text, model, tokenizer, max_length: int = 512, truncation: bool = True, accelerator: str = "cpu"):
    """
        Translate text using the fine-tuned model (to italian)
    """
    return transform_text(text, model, tokenizer, max_length, truncation, accelerator, task="translate English to Italian")

# transform function
# SUMMARIZE
def summarize_text(text, model, tokenizer, max_length: int = 512, truncation: bool = True, accelerator: str = "cpu"):
    """
        Summarize text using the fine-tuned model
    """
    return transform_text(text, model, tokenizer, max_length, truncation, accelerator, task="summarize")