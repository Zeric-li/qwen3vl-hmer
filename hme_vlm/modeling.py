from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_processor(model_id: str, min_pixels: int, max_pixels: int):
    return AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def resolve_vl_model_class(model_id: str):
    model_id_lower = model_id.lower()
    if "qwen3-vl" in model_id_lower:
        return Qwen3VLForConditionalGeneration
    if "qwen2.5-vl" in model_id_lower:
        return Qwen2_5_VLForConditionalGeneration
    raise ValueError(f"Unsupported VLM family for model_id: {model_id}")


def load_model_for_lora(config: dict):
    model_class = resolve_vl_model_class(config["model_id"])
    model = model_class.from_pretrained(
        config["model_id"],
        torch_dtype=resolve_torch_dtype(config["torch_dtype"]),
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )

    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config.get("lora_bias", "none"),
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model


def load_model_for_inference(
    model_id_or_adapter_path: str,
    min_pixels: int,
    max_pixels: int,
    torch_dtype: str,
    attn_implementation: str = "sdpa",
):
    adapter_config_path = Path(model_id_or_adapter_path) / "adapter_config.json"
    is_adapter_path = adapter_config_path.exists()

    if is_adapter_path:
        peft_config = PeftConfig.from_pretrained(model_id_or_adapter_path)
        base_model_id = peft_config.base_model_name_or_path

        model_class = resolve_vl_model_class(base_model_id)
        base_model = model_class.from_pretrained(
            base_model_id,
            torch_dtype=resolve_torch_dtype(torch_dtype),
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        model = PeftModel.from_pretrained(base_model, model_id_or_adapter_path)
        processor = load_processor(base_model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        return model, processor

    model_class = resolve_vl_model_class(model_id_or_adapter_path)
    model = model_class.from_pretrained(
        model_id_or_adapter_path,
        torch_dtype=resolve_torch_dtype(torch_dtype),
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    processor = load_processor(model_id_or_adapter_path, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor
