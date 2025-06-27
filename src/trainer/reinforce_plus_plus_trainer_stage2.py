# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import re
import ast
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .reinforce_plus_plus_config import ReinforcePlusPlusConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor
from decord import VideoReader, cpu
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import copy
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ReinforcePlusPlusTrainerStage2(Trainer):
    def __init__(
        self,
        model1: Union[str, PreTrainedModel],
        model2: Union[str, PreTrainedModel],
        args: ReinforcePlusPlusConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        num_selected_frames: Optional[int] = 8,
        num_candidate_frames: Optional[int] = 128,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model2 if isinstance(model2, str) else model2.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ReinforcePlusPlusConfig(f"{model_name}-ReinforcePlusPlus")

        # Models
        # Trained model, i.e., model2
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model2, str):
            model_id = model2
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `ReinforcePlusPlusConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "LLaVA-Video" in model_id or "LLaVAVideo" in model_id:
                overwrite_config = {}
                tokenizer2, model, image_processor2, _ = load_pretrained_model(
                    model2, 
                    None, 
                    "llava_qwen", 
                    torch_dtype="bfloat16", 
                    overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
                self.tokenizer2 = tokenizer2
                self.conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
                self.processor2 = image_processor2
            elif "Qwen2-VL" in model_id or "qwen2_vl" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model2, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id or "qwen25vl" in model_id or "Qwen25VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model2, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model2, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model2, **model_init_kwargs)
        else:
            model_id = model2.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `ReinforcePlusPlusConfig`, but your model2 is already instantiated. "
                    "This argument can only be used when the `model2` argument is a string."
                )


        if peft_config is not None:
            print('Use peft.')
            model = get_peft_model(model, peft_config)


        if is_deepspeed_zero3_enabled():
            print("Deepspeed ZeRO3 is enabled. The eval model will be loaded.")
            # Load the model1
            if model1 is not None:
                self.model1_infer = Qwen2_5_VLForConditionalGeneration.from_pretrained(model1, **model_init_kwargs)
            else:
                self.model1_infer = None

        # processing class for model1
        if "Qwen2-VL" in model1 or "qwen2_vl" in model1 or "Qwen2.5-VL" in model1 or "qwen25vl" in model1 or "Qwen25VL" in model1 or "Aria" in model1:
            self.processing_class_1 = AutoProcessor.from_pretrained(model1)
            pad_token_id_1 = self.processing_class_1.tokenizer.pad_token_id
            self.processing_class_1.pad_token_id = pad_token_id_1
            self.processing_class_1.eos_token_id = self.processing_class_1.tokenizer.eos_token_id
        else:
            self.processing_class_1 = AutoTokenizer.from_pretrained(self.model1_infer.config._name_or_path, padding_side="left")
            pad_token_id_1 = self.processing_class_1.pad_token_id


        # Processing class for model2
        if processing_class is None:
            if "LLaVA-Video" in model_id or "LLaVAVideo" in model_id:
                processing_class = self.tokenizer2
            elif "Qwen2-VL" in model_id or "qwen2_vl" in model_id or "Qwen2.5-VL" in model_id or "qwen25vl" in model_id or "Qwen25VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    # processing_class.image_processor.max_pixels = max_pixels
                    # processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                # pad_token_id = processing_class.pad_token_id


        # Data collator
        def data_collator(features):  # No data collation is needed in REINFORCE++
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the REINFORCE++ paper
        self.num_generations = args.num_generations  # = G in the GRPO paper, for REINFORCE++ it's 1
        self.generation_config_1 = GenerationConfig(
            max_new_tokens=args.max_completion_length_stage1,
            use_cache=True,
            do_sample=True,
            temperature=1,
            pad_token_id=pad_token_id_1,
            eos_token_id=self.processing_class_1.tokenizer.eos_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in REINFORCE++, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.model1_infer is not None:
            if self.is_deepspeed_enabled:
                self.model1_infer = prepare_deepspeed(self.model1_infer, self.accelerator)
            else:
                self.model1_infer = self.accelerator.prepare_model(self.model1_infer, evaluation_mode=True)


        self.clip_eps = 0.2
        self.gamma = 1
        self.model1_res = 112
        self.model2_res = 896
        self.num_selected_frames = num_selected_frames
        self.num_candidate_frames = num_candidate_frames
        self.frame_step = self.num_candidate_frames // self.num_selected_frames
        self.model2 = model2


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In ReinforcePlusPlusTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos=None, video_grid_thw=None):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs


    @staticmethod
    def annotate_frame_with_pil(text, frame, position="bottom_right", font_size=40, color="red"):
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        
        width, height = frame.size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        margin = 0
        if position == "top_left":
            x, y = margin, margin
        elif position == "top_right":
            x, y = width - text_width - margin, margin
        elif position == "bottom_left":
            x, y = margin, height - text_height - margin
        elif position == "bottom_right":
            x, y = width - text_width - margin, height - text_height - margin
        elif position == "center":
            x, y = (width - text_width) // 2, (height - text_height) // 2
        else:
            raise ValueError("Invalid position argument")

        if position in ["bottom_left", "bottom_right"]:
            y -= text_height / 3

        draw.text((x, y), text, font=font, fill=color)
        return frame


    @staticmethod
    def _preprocess_image(image, image_resolution=224):
        r"""
        Pre-processes a single image.
        """
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        return image


    def _process_frame(self, frame):
        image = Image.fromarray(frame)
        resized_image = self._preprocess_image(image, image_resolution=self.model2_res)
        if resized_image.mode != "RGB":
            resized_image = resized_image.convert("RGB")
        return resized_image


    def get_video_message(self, video_root: str):
        frames = []
        vr = VideoReader(video_root, ctx=cpu(0))
        # fps = vr.get_avg_fps()
        # frame_rate = 2
        total_frames = len(vr)
        video_maxlen = self.num_candidate_frames
        # sample_frames = round(total_frames / fps) * frame_rate
        # sample_frames = min(total_frames, sample_frames, video_maxlen)
        sample_frames = min(total_frames, video_maxlen)
        sample_frames = int(sample_frames / 2) * 2
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        # Extract frames as numpy array
        video_frames = vr.get_batch(sample_indices).asnumpy()

        # Process frames in parallel
        with ThreadPoolExecutor() as executor:
            frames = list(executor.map(lambda frame: self._process_frame(frame), video_frames))
        return frames

    def save_frames(self, frames):
        path = "restore"
        os.makedirs(path, exist_ok=True)
        for i, frame in enumerate(frames):
            file_path = os.path.join(path, f'frame_{i}.png')
            frame.save(file_path)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The ReinforcePlusPlusTrainer does not support returning outputs")

        # inputs[0]: ['problem', 'solution', 'video', 'original_question', 'original_answer']

        ################################# stage 1 #################################
        videos = [x["video"] for x in inputs]
        with ThreadPoolExecutor(max_workers=4) as executor:
            video_raw = list(executor.map(self.get_video_message, videos))

        def preprocess_video(video_frames):
            resized_frames = [
                self._preprocess_image(frame, image_resolution=self.model1_res) for frame in video_frames
            ]
            return [
                self.annotate_frame_with_pil(str(idx), frame, font_size=20)
                for idx, frame in enumerate(resized_frames)
            ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            video_inputs = list(executor.map(preprocess_video, video_raw))

        prompts1 = [x["prompt1"] for x in inputs]
        prompts_text1 = [
            self.processing_class_1.apply_chat_template(conversation, add_generation_prompt=True)
            for conversation in prompts1
        ]

        prompt_inputs_1 = self.processing_class_1(
            text=prompts_text1,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs_1 = super()._prepare_inputs(prompt_inputs_1)
        del prompt_inputs_1["second_per_grid_ts"]

        prompt_ids_1 = prompt_inputs_1["input_ids"]
        prompt_length = prompt_ids_1.size(1)

        with torch.inference_mode():
            output_ids = self.model1_infer.generate(
                **prompt_inputs_1, generation_config=self.generation_config_1, use_model_defaults=False
            )
            completion_ids_1 = output_ids[:, prompt_length:]

        frame_idx_texts = self.processing_class_1.batch_decode(
            completion_ids_1, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(frame_idx_texts)


        ########################### frame selection ###########################
        def select_frames(text, video_frames):
            content_match = re.search(r"<index>(.*?)</index>", text, re.DOTALL)
            text = content_match.group(1).strip() if content_match else ''
            matches = re.findall(r'\d+\.?\d*', text)
            frame_index = (
                [float(x.strip()) for x in matches]
                if matches and len(matches) == self.num_selected_frames and all(0 <= float(x) < self.num_candidate_frames for x in matches)
                else [i*self.frame_step for i in range(self.num_selected_frames)]
            )
            frame_indexs = sorted(min(int(idx), self.num_candidate_frames-1) for idx in frame_index)
            return [video_frames[frame_idx] for frame_idx in frame_indexs]

        selected_frames = [
            select_frames(text, video_raw[i]) for i, text in enumerate(frame_idx_texts)
        ]
        # self.save_frames(selected_frames[0])


        ########################### stage2 ###########################
        if "LLaVA-Video" in self.model2 or "LLaVAVideo" in self.model2:
            ########################### processing ###########################
            device = self.accelerator.device
            prompts2 = [prompt for prompt in [x["prompt2"] for x in inputs] for _ in range(self.num_generations)] #[q1, q1, q2, q2, q3, q3, ...]
            prompts_text2 = [
                self.processing_class_1.apply_chat_template(conversation, add_generation_prompt=True)
                for conversation in prompts2
            ]
            prompts_text2 = [prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", DEFAULT_IMAGE_TOKEN) for prompt in prompts_text2]
            video = []
            for selected_frame in selected_frames:
                video.append(self.processor2.preprocess(selected_frame, return_tensors="pt")["pixel_values"].to(device).bfloat16())
            input_ids = []
            for prompt in prompts_text2:
                input_ids.append(tokenizer_image_token(prompt, self.processing_class, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device))
            attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
            prompt_ids_2 = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.processing_class.pad_token_id, padding_side='left')
            prompt_mask_2 = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side='left').to(device)
            prompt_length = prompt_ids_2.size(1)

            ########################### training ###########################
            solutions = []
            for example in inputs:
                solutions.extend([example['solution'].replace('<answer>', '').replace('</answer>', '')] * self.num_generations)
            solution_inputs = self.processing_class_1(text=solutions, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
            solution_inputs = super()._prepare_inputs(solution_inputs) # cpu --> gpu
            solution_input_ids = solution_inputs["input_ids"]
            solution_attention_masks = solution_inputs["attention_mask"]

            pad_input_ids = torch.full((solution_input_ids.size(0), 1), self.processing_class.pad_token_id, device=device)
            pad_attention_masks = torch.zeros_like(pad_input_ids)
            prompt_ids_2 = torch.cat([prompt_ids_2, solution_input_ids, pad_input_ids], dim=1)
            prompt_mask_2 = torch.cat([prompt_mask_2, solution_attention_masks, pad_attention_masks], dim=1)
            label_ids = prompt_ids_2.clone()
            label_ids[prompt_mask_2==0] = -100  # Mask the prompt part
            label_ids[:, :prompt_length] = -100  # Mask the prompt part

            output = model(
                input_ids=prompt_ids_2,
                attention_mask=prompt_mask_2,
                labels=label_ids,
                images=torch.stack(video, dim=0).to(device),
                modalities= ["video"]*len(prompt_ids_2)
            )
            llm_loss = output.loss
            output_tokens = torch.argmax(output.logits, dim=-1)[:, -3:] # the length of input will change due to the visual tokens in the LLaVA model
            answers = self.processing_class.batch_decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            answers = [answer.replace("\n", "") for answer in answers]
            print("solutions:", solutions)
            print("answers:", answers)

            return llm_loss
        else:
            ########################### processing ###########################
            prompts2 = [prompt for prompt in [x["prompt2"] for x in inputs] for _ in range(self.num_generations)]
            prompts_text2 = [
                self.processing_class.apply_chat_template(conversation, add_generation_prompt=True)
                for conversation in prompts2
            ]

            prompt_inputs_2 = self.processing_class(
                text=prompts_text2,
                videos=selected_frames,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs_2 = super()._prepare_inputs(prompt_inputs_2)
            if 'second_per_grid_ts' in prompt_inputs_2:
                del prompt_inputs_2["second_per_grid_ts"]

            prompt_ids_2, prompt_mask_2 = prompt_inputs_2["input_ids"], prompt_inputs_2["attention_mask"]
            prompt_length = prompt_ids_2.size(1)

            ########################### training ###########################
            solutions = []
            for example in inputs:
                solutions.extend([example['solution'].replace('<answer>', '').replace('</answer>', '')] * self.num_generations)
            solution_inputs = self.processing_class(text=solutions, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
            solution_inputs = super()._prepare_inputs(solution_inputs) # cpu --> gpu
            solution_input_ids = solution_inputs["input_ids"]
            solution_attention_masks = solution_inputs["attention_mask"]
            device = self.accelerator.device
            pad_input_ids = torch.full((solution_input_ids.size(0), 1), self.processing_class.pad_token_id, device=device)
            pad_attention_masks = torch.zeros_like(pad_input_ids)
            prompt_ids_2 = torch.cat([prompt_ids_2, solution_input_ids, pad_input_ids], dim=1)
            prompt_mask_2 = torch.cat([prompt_mask_2, solution_attention_masks, pad_attention_masks], dim=1)
            label_ids = prompt_ids_2.clone()
            label_ids[prompt_mask_2==0] = -100  # Mask the prompt part
            label_ids[:, :prompt_length] = -100  # Mask the prompt part

            pixel_values_videos_2 = prompt_inputs_2["pixel_values_videos"]
            video_grid_thw_2 = prompt_inputs_2["video_grid_thw"]
            output = model(
                input_ids=prompt_ids_2, 
                attention_mask=prompt_mask_2, 
                labels=label_ids, 
                pixel_values_videos=pixel_values_videos_2, 
                video_grid_thw=video_grid_thw_2
            )
            llm_loss = output.loss
            output_tokens = torch.argmax(output.logits, dim=-1)[:, prompt_length-1:-2]
            output_tokens[solution_attention_masks==0] = self.processing_class.pad_token_id
            answers = self.processing_class.batch_decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print("solutions:", solutions)
            print("answers:", answers)

            return llm_loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)


    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{xu2025viarl,
                title={{ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning}},
                author={Xu, Ziqiang and Dai, Qi and Xie, Tian and Yang, Yifan and Qiu, Kai and Chen, DongDong and Wu, Zuxuan and Luo, Chong},
                journal={arXiv preprint arXiv:2505.15447},
                year={2025}
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="ReinforcePlusPlusTrainer",
            trainer_citation=citation,
            paper_title="ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning",
            paper_id="2505.15447",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
