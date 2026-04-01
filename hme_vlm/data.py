from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from PIL import Image
import torch


@dataclass
class HMERecord:
    sample_id: str
    image: Image.Image
    latex: str
    source: str


def load_hf_hme_records(dataset_id: str, split: str, max_samples: int | None, shuffle: bool, seed: int) -> list[HMERecord]:
    ds = load_dataset(dataset_id, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    records: list[HMERecord] = []
    for i, row in enumerate(ds):
        records.append(
            HMERecord(
                sample_id=f"{split}_{i}",
                image=row["image"].convert("RGB"),
                latex=row["label"],
                source=f"{dataset_id}:{split}",
            )
        )
    return records


def build_prompt_messages(image: Image.Image, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def build_train_messages(image: Image.Image, answer_text: str, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    messages = build_prompt_messages(image=image, system_prompt=system_prompt, user_prompt=user_prompt)
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
    )
    return messages


class QwenVLTrainCollator:
    def __init__(self, processor, system_prompt: str, user_prompt: str):
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        batch_messages = [
            build_train_messages(
                image=feature["image"].convert("RGB"),
                answer_text=feature["latex"],
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
            for feature in features
        ]
        prompt_messages = [
            build_prompt_messages(
                image=feature["image"].convert("RGB"),
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
            for feature in features
        ]

        full_texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in batch_messages
        ]
        prompt_texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in prompt_messages
        ]

        full_image_inputs, full_video_inputs = process_vision_info(batch_messages)
        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)

        batch = self.processor(
            text=full_texts,
            images=full_image_inputs,
            videos=full_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)

        prompt_batch = self.processor(
            text=prompt_texts,
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch.pop("token_type_ids", None)

        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        prompt_attention_mask = prompt_batch["attention_mask"]
        for row_index in range(labels.shape[0]):
            prompt_length = int(prompt_attention_mask[row_index].sum().item())
            labels[row_index, :prompt_length] = -100

        batch["labels"] = labels
        return batch


class QwenVLInferenceCollator:
    def __init__(self, processor, system_prompt: str, user_prompt: str):
        self.processor = processor
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        messages = [
            build_prompt_messages(
                image=feature["image"].convert("RGB"),
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
            for feature in features
        ]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        batch["raw_features"] = features
        return batch
