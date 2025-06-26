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


class ReinforcePlusPlusTrainerStage1(Trainer):
    def __init__(
        self,
        model1: Union[str, PreTrainedModel],
        model2: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
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
        model1_res: Optional[int] = 224,
        model2_res: Optional[int] = 896,
        num_selected_frames: Optional[int] = 8,
        num_candidate_frames: Optional[int] = 128,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model1 if isinstance(model1, str) else model1.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ReinforcePlusPlusConfig(f"{model_name}-ReinforcePlusPlus")

        # Models
        # Trained model, i.e., model1
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model1, str):
            model_id = model1
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
            if "Qwen2-VL" in model_id or "qwen2_vl" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model1, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id or "qwen25vl" in model_id or "Qwen25VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model1, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model1, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model1, **model_init_kwargs)
        else:
            model_id = model1.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `ReinforcePlusPlusConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Freeze visual model if needed
        if "Qwen2-VL" in model_id or "qwen2_vl" in model_id or "Qwen2.5-VL" in model_id or "qwen25vl" in model_id or "Qwen25VL" in model_id:
            print('Freeze visual model.')
            for p in model.visual.parameters():
                p.requires_grad = False

        if peft_config is not None:
            print('Use peft.')
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            print("Deepspeed ZeRO3 is enabled. The reference model will be loaded.")
            if "Qwen2-VL" in model_id or "qwen2_vl" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id or "qwen25vl" in model_id or "Qwen25VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Load the model2 (self.eval_model)
        if model2 is not None:
            if "LLaVA-Video" in model2 or "LLaVAVideo" in model2:
                overwrite_config = {}
                tokenizer2, self.eval_model, image_processor2, _ = load_pretrained_model(
                    model2, 
                    None, 
                    "llava_qwen", 
                    torch_dtype="bfloat16", 
                    overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
                self.tokenizer2 = tokenizer2
                self.conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
                self.processor2 = image_processor2
            else:
                self.eval_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model2, **model_init_kwargs)
        else:
            self.eval_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "qwen2_vl" in model_id or "Qwen2.5-VL" in model_id or "qwen25vl" in model_id or "Qwen25VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    # processing_class.image_processor.max_pixels = max_pixels
                    # processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in REINFORCE++
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the REINFORCE++ paper
        self.num_generations = args.num_generations  # = G in the GRPO paper, for REINFORCE++ it's 1
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.generation_config_2 = GenerationConfig(
            max_new_tokens=args.max_completion_length_2,
            use_cache=True,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in REINFORCE++, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

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

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if self.eval_model is not None:
            if self.is_deepspeed_enabled:
                self.eval_model = prepare_deepspeed(self.eval_model, self.accelerator)
            else:
                self.eval_model = self.accelerator.prepare_model(self.eval_model, evaluation_mode=True)


        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # Set the hyperparameters
        self.clip_eps = 0.2
        self.gamma = 1
        self.model1_res = model1_res
        self.model2_res = model2_res
        self.num_selected_frames = num_selected_frames
        self.num_candidate_frames = num_candidate_frames
        self.model2 = model2

        # Set the minimum number of tokens for the model to be trained.
        if num_selected_frames == 16:
            self.minimum_token_num = 110
        elif num_selected_frames == 8:
            self.minimum_token_num = 80
        else:
            NotImplementedError(f"num_selected_frames {num_selected_frames} not supported.")


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
        video_raw = []
        videos = [x["video"] for x in inputs]
        for video in videos:
            video_raw.append(self.get_video_message(video))

        video_inputs = []
        for video_frames in video_raw:
            with ThreadPoolExecutor() as executor:
                resized_frames = list(executor.map(lambda frame: self._preprocess_image(frame, image_resolution=self.model1_res), video_frames))
                resized_frames = list(executor.map(lambda args: self.annotate_frame_with_pil(str(args[0]), args[1], font_size=20), enumerate(resized_frames)))
                # self.save_frames(resized_frames)
            video_inputs.append(resized_frames)

        prompts1 = [x["prompt1"] for x in inputs]
        prompts2 = [x["prompt2"] for x in inputs]
        
        
        prompts_text1 = [self.processing_class.apply_chat_template(conversation, add_generation_prompt=True) for conversation in prompts1]

        prompt_inputs_1 = self.processing_class(
            text=prompts_text1,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs_1 = super()._prepare_inputs(prompt_inputs_1)
        del prompt_inputs_1["second_per_grid_ts"]

        prompt_ids_1, prompt_mask = prompt_inputs_1["input_ids"], prompt_inputs_1["attention_mask"]
        pixel_values_videos = prompt_inputs_1["pixel_values_videos"]
        video_grid_thw = prompt_inputs_1["video_grid_thw"]


        if self.max_prompt_length is not None:
            prompt_ids_1 = prompt_ids_1[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]


        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs_1, generation_config=self.generation_config, use_model_defaults=False)

            prompt_length = prompt_ids_1.size(1)
            prompt_ids_1 = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Decode the generated completions
        frame_idx_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        print(frame_idx_texts)

        ############################## per_token_logps ##############################
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values_videos = pixel_values_videos.repeat_interleave(self.num_generations, dim=0)
        video_grid_thw = video_grid_thw.repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values_videos, video_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model. For reinforce++, we use k1:
        per_token_kl = per_token_logps - ref_per_token_logps
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1


        ########################### frame selection ###########################
        selected_frames = []
        format_flags = []
        for i, text in enumerate(frame_idx_texts):
            content_match = re.search(r"<index>(.*?)</index>", text, re.DOTALL)
            if content_match:
                text = content_match.group(1).strip()
            else:
                text = ''
            matches = re.findall(r'\d+\.?\d*', text)
            if matches:
                extracted_list = [float(x.strip()) for x in matches]
                if len(extracted_list)==self.num_selected_frames and not None in extracted_list:
                    # is_sorted = all(extracted_list[i] <= extracted_list[i + 1] for i in range(len(extracted_list) - 1))
                    is_in_range = all(0 <= x < self.num_candidate_frames for x in extracted_list)
                    if is_in_range:
                       frame_index = extracted_list
                       format_flags.append("True")
                    else:
                        step = self.num_candidate_frames // self.num_selected_frames
                        frame_index = [i*step for i in range(self.num_selected_frames)] # default
                        format_flags.append("False")
                else:
                    step = self.num_candidate_frames // self.num_selected_frames
                    frame_index = [i*step for i in range(self.num_selected_frames)] # default
                    format_flags.append("False")
            else:
                step = self.num_candidate_frames // self.num_selected_frames
                frame_index = [i*step for i in range(self.num_selected_frames)] # default
                format_flags.append("False")
            
            frame_indexs = [min(int(idx), self.num_candidate_frames-1) for idx in (frame_index if isinstance(frame_index, list) else [frame_index])]
            frame_indexs.sort()
            video_idx = i // self.num_generations
            selected_frames.append([video_raw[video_idx][frame_idx] for frame_idx in frame_indexs])


        ########################### stage 2 ###########################
        if "LLaVA-Video" in self.model2 or "LLaVAVideo" in self.model2:
            prompts2 = [prompt for prompt in prompts2 for _ in range(self.num_generations)] #[q1, q1, q2, q2, q3, q3, ...]
            prompts_text2 = [
                self.processing_class.apply_chat_template(conversation, add_generation_prompt=True)
                for conversation in prompts2
            ]
            prompts_text2 = [prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", DEFAULT_IMAGE_TOKEN) for prompt in prompts_text2]
            video = []
            for selected_frame in selected_frames:
                video.append(self.processor2.preprocess(selected_frame, return_tensors="pt")["pixel_values"].to(device).bfloat16())
            input_ids = []
            for prompt in prompts_text2:
                input_ids.append(tokenizer_image_token(prompt, self.tokenizer2, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device))
            attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer2.pad_token_id, padding_side='left')
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side='left').to(device)
            
            ############### eval_model ###############
            with torch.inference_mode():
                with unwrap_model_for_generation(self.eval_model, self.accelerator) as unwrapped_eval_model:
                    output_ids = unwrapped_eval_model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        images=torch.stack(video, dim=0).to(device),
                        modalities= ["video"]*len(input_ids),
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=128,
                    )
                    answers = self.tokenizer2.batch_decode(output_ids, skip_special_tokens=True)
            print(answers)
        else:
            # Generate the final answer
            prompts2 = [prompt for prompt in prompts2 for _ in range(self.num_generations)] #[q1, q1, q2, q2, q3, q3, ...]
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

            ############### eval_model ###############
            with torch.inference_mode():
                output_ids = self.eval_model.generate(**prompt_inputs_2, generation_config=self.generation_config_2, use_model_defaults=False)
                completion_ids_2 = output_ids[:, prompt_length:]

            answers = self.processing_class.batch_decode(
                completion_ids_2, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print(answers)


        ########################### reward, advantage, loss ###########################
        # completions = [[{"role": "assistant", "content": f"<index>{index}</index><answer>{answer}</answer>"}] for (index, answer) in zip(frame_idx_texts, answers)]
        completions = [[{"role": "assistant", "content": f"{index}<answer>{answer}</answer>"}] for (index, answer) in zip(frame_idx_texts, answers)]
        prompts = [prompt for prompt in prompts1 for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]

                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompts", "prompt1", "prompt2", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        # three columns in rewards_per_func --> "accuracy", "index", "format"
        # if any condition of index is violated, set the accuracy to 0
        rewards_per_func[:, 0] = rewards_per_func[:, 0] * rewards_per_func[:, 1]
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        
        # length reward
        length_list = completion_mask.sum(1)
        assert len(length_list) == len(rewards)
        for idx in range(len(length_list)):
            if self.minimum_token_num <= length_list[idx] <= 512:
                rewards[idx] += 0.2


        rewards = rewards.reshape(-1, self.num_generations)
        # rewards = rewards - rewards.mean(-1, keepdim=True)
        rewards = rewards.reshape(-1)


        reward_tensor = torch.zeros_like(completion_ids, dtype=torch.float32)
        for i, idx in enumerate(eos_idx):
            idx = min(idx, reward_tensor.size(1) - 1)
            reward_tensor[i, idx] = rewards[i].to(reward_tensor.dtype)

        reward_tensor = reward_tensor - self.beta * per_token_kl

        with torch.no_grad():
            response_length = reward_tensor.size(1)
            advantages = torch.zeros_like(reward_tensor)
            cumulative_return = torch.zeros(reward_tensor.size(0), device=device)

            # Calculate returns by accumulating discounted rewards
            for t in reversed(range(response_length)):
                cumulative_return = reward_tensor[:, t] + self.gamma * cumulative_return
                advantages[:, t] = cumulative_return
                # cumulative_return = cumulative_return * completion_mask[:, t]  # Reset cumulative return if the token is masked
            advantages = masked_whiten(advantages, completion_mask)
            advantages = advantages * completion_mask


        # x - x.detach() allows for preserving gradients from x
        ratio = (per_token_logps - per_token_logps.detach()).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        # loss = (loss * completion_mask * same_answer_mask_tensor).sum() / completion_mask.sum()
        loss = (loss * completion_mask).sum() / completion_mask.sum()

        
        ########################### Log the metrics ###########################
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards_per_func).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        
        if self.accelerator.is_main_process:
            question = "\n".join(textwrap.wrap(inputs[0]['problem'], width=50))
            answer = "\n".join(textwrap.wrap(inputs[0]['solution'], width=50))
            think = "\n".join(textwrap.wrap(frame_idx_texts[0], width=50))
            flag = format_flags[0]
            pred = "\n".join(textwrap.wrap(answers[0], width=50))
            formatted_text = f"Question: {question}\nAnswer: {answer}\nThink: {think}\nFormat: {flag}\nPred: {pred}"
            wandb.log({f"Q-A-T-F-P": wandb.Html(f"<pre>{formatted_text}</pre>")}, step=self.state.global_step)

            # Horizontally concatenate frames in a single row
            vis_images = selected_frames[0]
            widths, heights = zip(*(frame.size for frame in vis_images))
            total_width = sum(widths)
            max_height = max(heights)

            # Create a new blank image with the appropriate size
            concatenated_image = Image.new("RGB", (total_width, max_height))

            # Paste each frame into the new image
            x_offset = 0
            for frame in vis_images:
                concatenated_image.paste(frame, (x_offset, 0))
                x_offset += frame.size[0]

            # Convert concatenated image to wandb.Image and log it
            # self.think_metrics["selected_images"].append(wandb.Image(concatenated_image))
            wandb.log({f"Selected Frames": wandb.Image(concatenated_image)}, step=self.state.global_step)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

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
