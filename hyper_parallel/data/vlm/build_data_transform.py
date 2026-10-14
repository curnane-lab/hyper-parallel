# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build the VLM multimodal sample transform."""

import base64
import re
import io
from pathlib import Path
from typing import Any, Optional

import torch

from hyper_parallel.data.constants import IGNORE_INDEX

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")


class VLMChatTransform:
    """Encode one multimodal conversation into one padded model sample.

    ``messages`` follows the Qwen3-VL content-list format
    (``{"type": "image", "url": ...}`` / ``{"type": "text", "text": ...}``), and
    ``<image>`` string placeholders are also accepted. Every ``assistant`` turn
    is a training target (``train_all_turns``); user/system context is masked
    with -100.

    Args:
        processor: Qwen3-VL processor (tokenizer + image/video processors).
        max_seq_len: Target sequence length for truncation and padding.
    """

    def __init__(
            self,
            processor: Any,
            *,
            max_seq_len: int = 256,
            min_pixels: int = 65536,
            max_pixels: int = 16777216,
            drop_image_truncated_sample: bool = True,
            train_all_turns: bool = True,
    ) -> None:
        """Store the processor and target sequence length.

        Args:
            processor: Qwen3-VL processor (tokenizer + image/video processors).
            max_seq_len: Target sequence length for truncation and padding.
            min_pixels: Lower pixel budget forwarded to the image processor.
            max_pixels: Upper pixel budget forwarded to the image processor.
            drop_image_truncated_sample: When True, a sample whose image vision
                tokens exceed ``max_seq_len`` raises instead of silently training
                the text answer without the image. Dataset wrappers skip such
                records; disable to keep the legacy zero-out behavior.
            train_all_turns: When True (default), every ``assistant`` turn in a
                multi-turn conversation is a training target; only user/system
                context is masked. When False, only the final assistant turn is
                trained (legacy behavior).
        """
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.drop_image_truncated_sample = drop_image_truncated_sample
        self.train_all_turns = train_all_turns

    @staticmethod
    def _load_images(images: Any) -> list[Any]:
        """Load local paths, base64 data URIs, and embedded bytes into PIL images."""
        from PIL import Image

        loaded = []
        for image in images or []:
            if isinstance(image, str) and image.startswith("data:image/"):
                image = Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1])))
            elif isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, str):
                if Path(image).is_file():
                    image = Image.open(image)
                else:
                    continue
            if image.mode != "RGB":
                image = image.convert("RGB")
            loaded.append(image)
        return loaded

    def _normalize_messages(self, messages: Any, images: Any = None) -> Any:
        """Split ``<image>``/``<video>`` string placeholders into content-list parts."""
        if images is None:
            images = []
        image_queue = list(images)
        normalized = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                parts = []
                for token in re.split(r"(<image>|<video>)", content):
                    if token in ("<image>", "<video>"):
                        if not image_queue:
                            raise ValueError(
                                f"message content contains a {token} placeholder but no matching media is available"
                            )
                        media_type = "image" if token == "<image>" else "video"
                        media = image_queue.pop(0)
                        parts.append({"type": media_type, "url": media})
                    elif token:
                        parts.append({"type": "text", "text": token})
                content = parts
            normalized.append({"role": message.get("role"), "content": content})
        return normalized

    def _encode(
            self,
            messages: Any,
            *,
            add_generation_prompt: bool,
            image_kwargs: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Render and encode one conversation with the processor."""
        chat_template = getattr(self.processor, "chat_template", None)
        if not chat_template:
            tokenizer = getattr(self.processor, "tokenizer", None)
            chat_template = getattr(tokenizer, "chat_template", None)
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            chat_template=chat_template,
            **(image_kwargs or {}),
        )

    @staticmethod
    def _modal_runs(mm: torch.Tensor) -> list[tuple[int, int]]:
        """Return inclusive ``(start, end)`` positions of contiguous image-token runs."""
        is_modal = mm == 1
        runs: list[tuple[int, int]] = []
        start: Optional[int] = None
        for pos in range(int(mm.shape[0])):
            if bool(is_modal[pos]):
                if start is None:
                    start = pos
            elif start is not None:
                runs.append((start, pos - 1))
                start = None
        if start is not None:
            runs.append((start, int(mm.shape[0]) - 1))
        return runs

    def _drop_truncated_images(
            self, sample: dict[str, torch.Tensor], max_len: int
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Drop images whose vision-token run extends past ``max_len``.

        Returns the updated sample and the number of dropped images.
        """
        grid = sample["image_grid_thw"]
        if grid.numel() == 0:
            return sample, 0

        runs = self._modal_runs(sample["mm_token_type_ids"])
        keep = 0
        for run_idx, (_, end) in enumerate(runs):
            if run_idx >= int(grid.shape[0]) or end >= max_len:
                break
            keep += 1
        if keep == int(grid.shape[0]):
            return sample, 0

        dropped = int(grid.shape[0]) - keep
        mm = sample["mm_token_type_ids"].clone()
        input_ids = sample["input_ids"].clone()
        attention_mask = sample["attention_mask"].clone()
        labels = sample["labels"].clone()
        for start, end in runs[keep:]:
            mm[start:end + 1] = 0
            input_ids[start:end + 1] = 0
            attention_mask[start:end + 1] = 0
            labels[start:end + 1] = IGNORE_INDEX
        sample["mm_token_type_ids"] = mm
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["labels"] = labels

        pixel_rows = sum(int(g[0]) * int(g[1]) * int(g[2]) for g in grid[:keep])
        sample["pixel_values"] = sample["pixel_values"][:pixel_rows]
        sample["image_grid_thw"] = grid[:keep]
        return sample, dropped

    def _truncate_and_pad(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Truncate (dropping cut images) and pad sequence fields to ``max_seq_len``."""
        max_len = self.max_seq_len
        seq_len = int(sample["input_ids"].shape[0])
        if seq_len > max_len:
            sample, dropped = self._drop_truncated_images(sample, max_len)
            if dropped and self.drop_image_truncated_sample:
                raise ValueError(
                    f"{dropped} image(s) exceed max_seq_len={max_len} after encoding "
                    f"(sequence length {seq_len}); raise max_seq_len, lower max_pixels, "
                    f"or set drop_image_truncated_sample=False to keep the legacy behavior"
                )
            for field in _SEQ_FIELDS:
                sample[field] = sample[field][:max_len]

        for field in _SEQ_FIELDS:
            value = sample[field]
            if value.shape[0] < max_len:
                pad_value = IGNORE_INDEX if field == "labels" else 0
                sample[field] = torch.cat([
                    value,
                    torch.full((max_len - value.shape[0],), pad_value, dtype=value.dtype),
                ])
        return sample

    def __call__(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode one record into a padded model sample."""
        images = self._load_images(record.get("images"))
        messages = self._normalize_messages(record["messages"], images)
        for message in messages:
            if isinstance(message.get("content"), str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        image_kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        full = self._encode(messages, add_generation_prompt=False, image_kwargs=image_kwargs)

        input_ids = torch.tensor(full["input_ids"][0], dtype=torch.long)
        mm = full.get("mm_token_type_ids")
        if mm is None:
            image_token_id = int(getattr(self.processor, "image_token_id", -1))
            if image_token_id < 0:
                image_token_id = int(getattr(getattr(self.processor, "tokenizer", None), "image_token_id", -1))
            mm = (input_ids == image_token_id).long().unsqueeze(0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        if self.train_all_turns:
            train_turns = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
        else:
            train_turns = [len(messages) - 1]
        seq_len = int(input_ids.shape[0])
        for turn in train_turns:
            if turn == 0:
                start = 0
            else:
                prefix = self._encode(
                    messages[:turn], add_generation_prompt=True, image_kwargs=image_kwargs
                )
                start = len(prefix["input_ids"][0])
            if turn == len(messages) - 1:
                end = seq_len
            else:
                through_turn = self._encode(
                    messages[:turn + 1], add_generation_prompt=False, image_kwargs=image_kwargs
                )
                end = len(through_turn["input_ids"][0])
            start = min(start, seq_len)
            end = min(end, seq_len)
            if start < end:
                labels[start:end] = input_ids[start:end]

        sample = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(full["attention_mask"][0], dtype=torch.long),
            "mm_token_type_ids": torch.tensor(mm[0], dtype=torch.long),
            "labels": labels,
            "pixel_values": full["pixel_values"],
            "image_grid_thw": full["image_grid_thw"],
        }
        return self._truncate_and_pad(sample)


def build_vlm_data_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 256,
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
        drop_image_truncated_sample: bool = True,
        train_all_turns: bool = True,
        **transform_options: Any,
) -> VLMChatTransform:
    """Build the VLM sample transform.

    Args:
        processor: Qwen3-VL processor used to render and encode conversations.
        max_seq_len: Target sequence length for truncation and padding.
        min_pixels: Lower pixel budget forwarded to the image processor.
        max_pixels: Upper pixel budget forwarded to the image processor.
        drop_image_truncated_sample: Skip samples whose image no longer fits
            into ``max_seq_len`` instead of training text without the image.
        train_all_turns: Train every assistant turn of multi-turn
            conversations, not only the final one.
        **transform_options: Reserved model-specific transform options.

    Returns:
        The configured :class:`VLMChatTransform`.

    Raises:
        ValueError: If ``processor`` is not provided.
    """
    del transform_options
    if processor is None:
        raise ValueError("processor is required for the VLM data transform")
    return VLMChatTransform(
        processor,
        max_seq_len=max_seq_len,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        drop_image_truncated_sample=drop_image_truncated_sample,
        train_all_turns=train_all_turns,
    )


__all__ = ["VLMChatTransform", "build_vlm_data_transform"]
