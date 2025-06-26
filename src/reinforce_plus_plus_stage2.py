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
import re
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict

# from math_verify import parse, verify
from trainer.reinforce_plus_plus_trainer_stage2 import ReinforcePlusPlusTrainerStage2
from trainer.reinforce_plus_plus_config import ReinforcePlusPlusConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config


# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger, Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
# --------------------------------------------------------------------------------------------


# ---------------------------------------------- Script Arguments ----------------------------------------------
# For 8 frames
num_selected_frames = 8
num_candidate_frames = 128

# For 16 frames, uncomment the following lines:
# num_selected_frames = 16
# num_candidate_frames = 256

@dataclass
class ReinforcePlusPlusScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        max_pixels (int, optional): Maximum number of pixels for the image. Defaults to 12845056.
        min_pixels (int, optional): Minimum number of pixels for the image. Defaults to 3136.
        jsonl_path (str, optional): Path to a JSONL file containing the dataset.
        max_completion_length_stage1 (int, optional): Maximum length of the generated completion. Defaults to 256.
        model_name_or_path_2 (str): The second MLLM model name or path.
    """
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )
    max_completion_length_stage1: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    model_name_or_path_2: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "The second MLLM."},
    )
# --------------------------------------------------------------------------------------------



# ---------------------------------------------- System Prompt ----------------------------------------------
# The system prompt is used to instruct the model on how to select relevant video frames based on the question.
# It provides guidelines on how to think about the keywords in the question, how to describe the visual appearance of the frames, and how to format the output.
# The prompt also specifies the number of frames to select and the range of frame indices.
SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for selecting the relevant video frames according to a question.\n"
    f"User will provide you a video with {num_candidate_frames} frames and a short question. \n"
    f"The red numbers in the bottom right corner of each frame represent the frame indice. The frame index is an integer in the range of 0 to {num_candidate_frames-1}. \n"
    f"Your task is to output {num_selected_frames} indices of the frames that can help you answer the question better.\n"
    "Here's how you can accomplish the task:\n"
    "------------------\n"
    "##1. Think about the keywords from the question: \n"
    "- Check if the physical entities are mentioned.\n"
    "- Check if the occurrence time is mentioned.\n"
    "- Check if the place or location is mentioned.\n"
    "- Check if the action is mentioned.\n\n"
    "##2. Provide the appearance reference based on the keywords and video: \n"
    f"- Describe the visual appearance of the {num_selected_frames} frames that are most relevant to the keywords.\n\n"
    "##3. Provide the target list: \n"
    f"- A list of {num_selected_frames} frame indices, that the corresponding frames are most helpful to answer the question.\n\n"
    "Your output should follow this format strictly:\n"
    "<think> thinking about keywords and appearance here </think><index> target list here </index>\n\n"
    "Specific requirements are as follows: \n"
    "**Ensure that anyone can uniquely identify these target frames in the video through the references.**\n"
    "**Ensure that the references are complete and independent.**\n"
    "**Don't output the words '<think> thinking about keywords and appearance here </think>' directly.**\n"
    f"**Ensure that the list consists of {num_selected_frames} values.**\n\n"
)


QUESTION_TEMPLATE_1 = """The question is "{Question}"
"""
QUESTION_TEMPLATE_2 = "{Question}\n Answer with the option's letter from the given choices directly."
QUESTION_TEMPLATE_3 = "{Question}\n Please provide your text answer."


# ---------------------------------------------- Main ----------------------------------------------
def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)

    print("base_dataset: ", len(base_dataset))
    
    return DatasetDict({
        "train": base_dataset
    })


def make_conversation_video(example): 
    problem_type = example['problem_type']
    if problem_type == "open_ended":
        Q1 = example["problem"].replace("?", "")
        Q2 = example["problem"]
    elif problem_type == "multi_choice":
        Q1 = example["problem"].split('\n')[0].replace("?", "")
        Q2 = example["problem"]

    return {
        "prompt1": [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": QUESTION_TEMPLATE_1.format(Question=Q1)},
                ],
            },
        ],
        "prompt2": [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {
                        "type": "text", 
                        "text": QUESTION_TEMPLATE_2.format(Question=Q2) if problem_type == "multi_choice" else QUESTION_TEMPLATE_3.format(Question=Q2)
                    },
                ],
            },
        ],
    }


def main(script_args, training_args, model_args):
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    if "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )


    # Initialize the GRPO trainer
    trainer = ReinforcePlusPlusTrainerStage2(
        model1=model_args.model_name_or_path,
        model2=model_args.model_name_or_path_2,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        num_selected_frames=num_selected_frames,
        num_candidate_frames=num_candidate_frames,
    )

    # Train and push the model to the Hub
    trainer.train()


    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ReinforcePlusPlusScriptArguments, ReinforcePlusPlusConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    training_args.max_completion_length_stage1 = script_args.max_completion_length_stage1
    model_args.model_name_or_path_2 = script_args.model_name_or_path_2
    
    os.makedirs(os.path.dirname(os.getenv("LOG_PATH")), exist_ok=True)
    main(script_args, training_args, model_args)