from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from hme_vlm.config import load_yaml_config
from hme_vlm.data import build_prompt_messages, load_hf_hme_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="2019")
    parser.add_argument("--dataset-id", type=str, default="Neeze/CROHME-full")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def chunked(items: list[object], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def make_output_dir(checkpoint: str, split: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path / f"eval_{split}"

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", checkpoint).strip("-")
    return Path("outputs") / "evals" / safe_name / f"eval_{split}"


def load_inference_config(args: argparse.Namespace) -> tuple[dict[str, object], object, object]:
    from hme_vlm.modeling import load_model_for_inference

    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = {
            "torch_dtype": "bfloat16",
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "system_prompt": (
                "You are a handwritten mathematical expression transcription engine. "
                "Your entire reply must be exactly one raw LaTeX expression. "
                "Output only the expression content itself."
            ),
            "user_prompt": (
                "Transcribe the handwritten mathematical expression in the image into one raw LaTeX expression. "
                "Output exactly one line with only the expression."
            ),
            "max_new_tokens": 128,
        }

    model, processor = load_model_for_inference(
        model_id_or_adapter_path=args.checkpoint,
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        torch_dtype=config["torch_dtype"],
    )
    return config, model, processor


def main() -> None:
    args = parse_args()

    from qwen_vl_utils import process_vision_info

    config, model, processor = load_inference_config(args)
    records = load_hf_hme_records(
        dataset_id=args.dataset_id,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=False,
        seed=42,
    )

    output_dir = make_output_dir(args.checkpoint, args.split, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    max_new_tokens = int(config.get("max_new_tokens", 128))

    for batch_records in tqdm(list(chunked(records, args.batch_size)), desc=f"Inference {args.split}"):
        messages = [
            build_prompt_messages(
                image=record.image,
                system_prompt=config["system_prompt"],
                user_prompt=config["user_prompt"],
            )
            for record in batch_records
        ]
        texts = [
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_s = (time.perf_counter() - start) / max(len(batch_records), 1)

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for record, raw_output in zip(batch_records, decoded):
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "gold_latex_raw": record.latex,
                    "pred_text_raw": raw_output,
                    "latency_s": latency_s,
                }
            )

    if not rows:
        raise ValueError("No evaluation samples were loaded. Check --split, --dataset-id, or --max-samples.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "raw_predictions.csv", index=False)

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "num_samples": len(out_df),
        "avg_latency_s": float(out_df["latency_s"].mean()),
    }
    with open(output_dir / "inference_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved raw predictions to: {output_dir / 'raw_predictions.csv'}")


if __name__ == "__main__":
    main()
