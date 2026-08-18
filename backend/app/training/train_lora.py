"""
QLoRA fine-tuning job. Designed to run as a scheduled/triggered CI/CD step
(see .github/workflows/train.yml) on a GPU runner — NOT on a laptop CPU.

Loads the base model in 4-bit, trains a small LoRA adapter on the dataset
produced by dataset_builder.py, and saves ONLY the adapter (a few MB)
rather than the full model. evaluate.py decides whether it's good enough
to promote.

Invoke from the repo root: PYTHONPATH=backend python -m app.training.train_lora
"""
import argparse
import pathlib

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from app.config import get_settings

settings = get_settings()


def format_example(example: dict) -> dict:
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": prompt}


def main(dataset_path: str, output_dir: str, epochs: int = 1):
    if not torch.cuda.is_available():
        raise SystemExit(
            "train_lora.py needs a CUDA GPU (bitsandbytes 4-bit training isn't "
            "supported on CPU). Run this on a GPU box, or a cloud GPU "
            "(Colab/RunPod/Lambda) instead of locally."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(format_example)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=512, padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)  # saves ONLY the LoRA adapter
    tokenizer.save_pretrained(output_dir)
    print(f"[train_lora] adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/training_sets/latest.jsonl")
    parser.add_argument("--output", default="models/adapters/candidate")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args.dataset, args.output, args.epochs)
