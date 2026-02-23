import argparse
import json
import random
from typing import Any, Dict, List

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(dtype_cfg: str, device: str) -> torch.dtype:
    if dtype_cfg == "auto":
        return torch.float32 if device == "mps" else torch.bfloat16

    lookup = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_cfg not in lookup:
        raise ValueError(f"Unsupported dtype '{dtype_cfg}'. Use one of: {', '.join(lookup)} or auto")
    return lookup[dtype_cfg]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_messages(prompt: str, target_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        },
    ]


def extract_target_text(ground_truth: str) -> str:
    gt = json.loads(ground_truth)
    lines = [line["text"] for line in gt.get("valid_line", []) if "text" in line]
    return " ".join(lines)


def main(config_path: str) -> None:
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    set_seed(seed)

    device = detect_device()
    dtype = resolve_dtype(config["model"].get("dtype", "auto"), device)

    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    dataset_split = config["dataset"].get("split", "train")
    prompt = config["dataset"]["text_prompt"]

    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    processor = LightOnOcrProcessor.from_pretrained(model_name)

    print(f"Loading dataset: {dataset_name} ({dataset_split})")
    train_dataset = load_dataset(dataset_name, split=dataset_split)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        text = extract_target_text(example["ground_truth"])
        messages = build_messages(prompt=prompt, target_text=text)

        chat_template = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=True,
        )

        inputs = processor(
            images=example["image"],
            text=chat_template,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].flatten().tolist()
        attention_mask = inputs["attention_mask"].flatten().tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }

    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)

    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        report_to=train_cfg.get("report_to", "none"),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    save_dir = f"{train_cfg['output_dir']}-lora"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved LoRA adapter and processor to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LightOnOCR with LoRA using YAML config")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
