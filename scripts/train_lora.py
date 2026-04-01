from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from hme_vlm.config import load_yaml_config
from hme_vlm.data import load_hf_hme_records, QwenVLTrainCollator
from hme_vlm.modeling import load_model_for_lora, load_processor


def resolve_train_eval_split(config: dict) -> str:
    eval_split = config.get("train_eval_split") or config.get("eval_split")
    if eval_split:
        return str(eval_split)

    eval_splits = config.get("eval_splits") or []
    if eval_splits:
        return str(eval_splits[0])

    raise ValueError("Config must define one of: train_eval_split, eval_split, or eval_splits.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def records_to_dataset(records):
    return Dataset.from_list(
        [
            {
                "sample_id": r.sample_id,
                "image": r.image,
                "latex": r.latex,
                "source": r.source,
            }
            for r in records
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    os.makedirs(config["output_dir"], exist_ok=True)
    set_seed(config["seed"])
    random.seed(config["seed"])

    train_records = load_hf_hme_records(
        dataset_id=config["train_dataset_id"],
        split=config["train_split"],
        max_samples=config.get("max_train_samples"),
        shuffle=config.get("shuffle_train", True),
        seed=config["seed"],
    )
    train_eval_split = resolve_train_eval_split(config)
    eval_records = load_hf_hme_records(
        dataset_id=config["eval_dataset_id"],
        split=train_eval_split,
        max_samples=config.get("max_eval_samples"),
        shuffle=False,
        seed=config["seed"],
    )

    train_dataset = records_to_dataset(train_records)
    eval_dataset = records_to_dataset(eval_records)

    processor = load_processor(
        model_id=config["model_id"],
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
    )
    model = load_model_for_lora(config)
    model.print_trainable_parameters()

    collator = QwenVLTrainCollator(
        processor=processor,
        system_prompt=config["system_prompt"],
        user_prompt=config["user_prompt"],
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        remove_unused_columns=False,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=config["save_total_limit"],
        bf16=(config["torch_dtype"] == "bfloat16"),
        fp16=(config["torch_dtype"] == "float16"),
        dataloader_num_workers=4,
        report_to=config.get("report_to", "none"),
        label_names=["labels"],
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()

    final_dir = Path(config["output_dir"]) / "checkpoint-final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print(f"Saved final checkpoint to: {final_dir}")


if __name__ == "__main__":
    main()
