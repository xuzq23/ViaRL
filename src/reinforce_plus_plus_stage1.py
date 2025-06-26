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
from datetime import datetime
from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict

# from math_verify import parse, verify
from trainer.reinforce_plus_plus_trainer_stage1 import ReinforcePlusPlusTrainerStage1
from trainer.reinforce_plus_plus_config import ReinforcePlusPlusConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from concurrent.futures import ThreadPoolExecutor


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
        reward_funcs (list[str]): List of reward functions. Possible values: 'accuracy', 'index', 'format'.
        max_pixels (int, optional): Maximum number of pixels for the image. Defaults to 12845056.
        min_pixels (int, optional): Minimum number of pixels for the image. Defaults to 3136.
        jsonl_path (str, optional): Path to a JSONL file containing the dataset.
        max_completion_length_stage1 (int, optional): Maximum length of the generated completion. Defaults to 256.
        model_name_or_path_2 (str): The second MLLM model name or path.
        model1_res (int, optional): Frame resolution for model1. Defaults to 112.
        model2_res (int, optional): Frame resolution for model2. Defaults to 896.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "index", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
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
    max_completion_length_2: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    model_name_or_path_2: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "The second MLLM."},
    )
    model1_res: Optional[int] = field(
        default=112,
        metadata={"help": "frame resolution for model1."},
    )
    model2_res: Optional[int] = field(
        default=896,
        metadata={"help": "frame resolution for model2."},
    )
# --------------------------------------------------------------------------------------------


# ---------------------------------------------- Rewards ----------------------------------------------
# The rewards consist of three functions:
# 1. `accuracy_reward`: Computes the accuracy of the generated answer against the ground truth answer.
# 2. `format_reward`: Checks if the generated completion follows a specific format.
# 3. `index_reward`: Checks if the indices in the generated indexes are valid and unique.
# The Response Length Reward is in the `ReinforcePlusPlusTrainerStage1` class, which is not shown here.

def evaluate_open_vqa(question, answer, pred):
    """
    Evaluates question and answer pairs using GPT-4
    Returns a score for correctness.
    """
    pass


def evaluate_mc_vqa(answer, pred):
    score = 2.0 if pred.replace('.', '').strip() == answer.replace('.', '').strip() else 0.0
    return score

def accuracy_reward(completions, solution, **kwargs):
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def compute_reward(args):
        problem_type, problem, content, sol = args
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if problem_type == "multi_choice":
                return evaluate_mc_vqa(gt_ans, output_ans)
            elif problem_type == "open_ended":
                return evaluate_open_vqa(problem, gt_ans, output_ans)
            else:
                return 0.0
        except Exception as e:
            print(f"Error in reward_fn for problem_type '{problem_type}': {e}")
            return 0.0

    problem_types = kwargs['problem_type']
    problems = kwargs['problem']
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Prepare arguments for parallel processing
    args_list = zip(problem_types, problems, contents, solution)

    # Use ThreadPoolExecutor for parallel computation
    with ThreadPoolExecutor() as executor:
        rewards = list(executor.map(compute_reward, args_list))

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path, "a", encoding="utf-8") as f:
            for problem, content, sol, reward in zip(problems, contents, solution, rewards):
                f.write(f"------------- {current_time} -------------\n")
                f.write(f"Problem: {problem}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
                f.write(f"Accuracy reward: {reward}\n")

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<index>.*?</index><answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def index_reward(completions, max_frame_idx=num_candidate_frames, **kwargs):
    """Reward function that checks if the index has a specific format."""

    completion_contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content in completion_contents:
        content_match = re.search(r"<index>(.*?)</index>", content, re.DOTALL)
        if content_match:
            text = content_match.group(1).strip()
        else:
            text = ''
        matches = re.findall(r'\d+\.?\d*', text)
        if matches:
            extracted_list = [float(x.strip()) for x in matches]
            if len(extracted_list)==num_selected_frames and not None in extracted_list:
                are_all_values_unique = len(extracted_list) == len(set(extracted_list))
                # is_sorted = all(extracted_list[i] <= extracted_list[i + 1] for i in range(len(extracted_list) - 1))
                is_in_range = all(0 <= x < max_frame_idx for x in extracted_list)
                if are_all_values_unique and is_in_range:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "index": index_reward,
}
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
        Q1 = example["problem"]
        Q2 = example["problem"]
    elif problem_type == "multi_choice":
        Q1 = example["problem"].split('\n')[0]
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
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
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
    trainer = ReinforcePlusPlusTrainerStage1(
        model1=model_args.model_name_or_path,
        model2=model_args.model_name_or_path_2,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        model1_res=script_args.model1_res,
        model2_res=script_args.model2_res,
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
    
    training_args.max_completion_length_2 = script_args.max_completion_length_2
    model_args.model_name_or_path_2 = script_args.model_name_or_path_2

    os.makedirs(os.path.dirname(os.getenv("LOG_PATH")), exist_ok=True)
    main(script_args, training_args, model_args)