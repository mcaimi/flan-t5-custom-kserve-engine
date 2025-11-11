#!/usr/bin/env python

try:
    import torch
except Exception as e:
    raise e

# transform function
# ANONYMIZE
def anonymize_text(text, model, tokenizer, max_length: int = 512, truncation: bool = True, accelerator: str = "cpu"):
    """
    Anonymize PII in Italian text using the fine-tuned model
    """
    # Prepare input
    input_text = f"anonymize: {text}"
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
    anonymized = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return anonymized