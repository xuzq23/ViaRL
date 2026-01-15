import os
import os.path as osp
import json
import re
import yaml
import datetime
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

from viarl.dataset_utils import get_dataset, get_eval_methods


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def trimm_results(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]



num_selected_frames = 8  # Define the variable for the number of frames
num_candidate_frames = 128


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


QUESTION_TEMPLATE_RL = """The question is "{Question}"
"""

class InferClient:
    def __init__(self, model_path1, model_path2, exp_configs, device) -> None:
        self.device = device
        self.do_sample = exp_configs.get('do_sample', False)
        self.method = exp_configs['method']
        self.type = exp_configs['type']
        self.load_model(model_path1, model_path2, device)

    def load_model(self, model_path1, model_path2, device):
        self.model1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path1, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device).eval()
        self.processor1 = AutoProcessor.from_pretrained(model_path1)
        print("load model1 from ", model_path1)

        self.model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path2, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).to(device).eval()
        self.processor2 = AutoProcessor.from_pretrained(model_path2)
        print("load model2 from ", model_path2)

    @staticmethod
    def _preprocess_image(image, image_resolution: int):
        r"""
        Pre-processes a single image.
        image_resolution: The longest side of the image after resizing.
        """
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @staticmethod
    def annotate_frame_with_pil(text, frame, position="bottom_right", font_size=40, color="red"):
        draw = ImageDraw.Draw(frame)
        base_dir = Path(__file__).resolve().parent.parent.parent
        font = ImageFont.truetype(str(base_dir / "DejaVuSans-Bold.ttf"), font_size)

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

    def infer(self, message, question_id, output_dir):
        raw_video = message['video']

        # Process frames in parallel
        with ThreadPoolExecutor() as executor:
            resized_frames = list(executor.map(lambda frame: self._preprocess_image(frame, 112), raw_video))
            video_step1 = list(executor.map(lambda args: self.annotate_frame_with_pil(str(args[0]), args[1], font_size=20), enumerate(resized_frames)))


        QA = message['question']
        Q = QA.split('\nOptions')[0]
        
        conversation_1 = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text", 
                        "text": QUESTION_TEMPLATE_RL.format(Question=Q)
                    },
                ],
            }
        ]
        # Preprocess the inputs
        text_prompt = self.processor1.apply_chat_template(conversation_1, add_generation_prompt=True)
        inputs = self.processor1(text=[text_prompt], videos=[video_step1], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)


        output_ids = self.model1.generate(**inputs, use_cache=True, do_sample=True, temperature=1, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor1.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_text[0]
        print(output_text)

        content_match = re.search(r"<index>(.*?)</index>", output_text, re.DOTALL)
        if content_match:
            text = content_match.group(1).strip()
        else:
            text = ''
        matches = re.findall(r'\d+\.?\d*', text)
        relevant_list = [int(float(match)) for match in matches]
        are_all_values_unique = len(relevant_list) == len(set(relevant_list))
        if len(relevant_list)==num_selected_frames and are_all_values_unique:
            relevant_list.sort()
        else:
            relevant_list = np.linspace(0, num_candidate_frames-1, num_selected_frames).astype(np.int32).tolist()

        relevant_list = [min(idx, num_candidate_frames-1) for idx in relevant_list]

        selected_frames = []
        for i in range(len(relevant_list)):
            selected_frames.append(raw_video[relevant_list[i]])


        conversation_2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text", 
                        "text": QA
                    }
                ],
            }
        ]
        text_prompt = self.processor2.apply_chat_template(conversation_2, add_generation_prompt=True)
        inputs = self.processor2(text=[text_prompt], videos=[selected_frames], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        output_ids = self.model2.generate(**inputs, use_cache=True, do_sample=False, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor2.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_text[0]
        return output_text, relevant_list



def parse_arguments():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ViaRL Evaluation")
    parser.add_argument('--model_path1', 
                        type=str, 
                        default='Qwen/Qwen2.5-VL-3B-Instruct', 
                        help="Path to the experimental config")
    parser.add_argument('--model_path2', 
                        type=str, 
                        default='Qwen/Qwen2.5-VL-7B-Instruct', 
                        help="Path to the experimental config")
    parser.add_argument('--config_path', 
                        type=str, 
                        default='configs/videomme.yaml', 
                        help="Path to the experimental config")
    parser.add_argument('--video_frame_extraction_fps', 
                        type=int, 
                        default=25, 
                        help="Video frame extraction FPS")
    parser.add_argument('--n_gpus', 
                        type=int, 
                        default=8, 
                        help="Number of GPUs to use")
    parser.add_argument('--timeout', 
                        type=int, 
                        default=30, 
                        help="Timeout for waiting each GPU finish inference (in minutes).")
    args = parser.parse_args()
    return args


def setup(rank, world_size, timeout):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=timeout))
    torch.cuda.set_device(f'cuda:{rank}')


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    device = f'cuda:{rank}'
    setup(rank, world_size, args.timeout)

    exp_configs = load_yaml(args.config_path)

    dataset = get_dataset(
        method=exp_configs['method'],
        dataset_name=exp_configs['dataset_name'],
        anno_file=exp_configs['anno_file'], 
        processor_kwargs=dict(
            video_fps=exp_configs['sample_fps'],
            video_maxlen=exp_configs['max_num_frames'],
            image_resolution=exp_configs['longsize_resolution'],
            video_frame_extraction_fps=args.video_frame_extraction_fps
        )
    )


    # Split dataset into shards
    indices = [i for i in range(len(dataset)) if i % world_size == rank]
    shard_dataset = Subset(dataset, indices)
    dataloader = DataLoader(shard_dataset, batch_size=None, num_workers=exp_configs['dataloader_num_workers'])

    # model
    client = InferClient(args.model_path1, args.model_path2, exp_configs, device)

    # Inference
    anno_id2result = {}
    anno_id2meta = {}
    os.makedirs(exp_configs['output_dir'], exist_ok=True)
    log_path = os.path.join(exp_configs['output_dir'], exp_configs['log_file'])


    for sample in tqdm(dataloader, desc=f'rank {rank}'):
        try:
            assert sample is not None
            idx, message, meta = sample
            question_id = meta['question_id']
            pred_answer, relevant_list = client.infer(message, question_id, exp_configs['output_dir'])

            with open(log_path, "a") as f:
                f.write(f"Content: {pred_answer}\n")
                f.write(f"Solution: {meta}\n")
                f.write(f"Relevant: {relevant_list}\n") 
                f.write("\n")
            pred_answer = trimm_results(pred_answer)
            anno_id2result[idx] = pred_answer
            anno_id2meta[idx] = meta
            print("pred: ", pred_answer, " answer: ", meta['answer'])
        except Exception as e:
            print(f"Error in rank {rank} for sample: {e}")


    # Gather results from all processes
    all_anno_id2result = [None] * world_size
    all_anno_id2meta = [None] * world_size
    dist.barrier()
    dist.all_gather_object(all_anno_id2result, anno_id2result)
    dist.all_gather_object(all_anno_id2meta, anno_id2meta)


    if rank == 0:
        # Merge results
        merged_anno_id2result = {k: v for d in all_anno_id2result for k, v in d.items()}
        merged_anno_id2meta = {k: v for d in all_anno_id2meta for k, v in d.items()}

        # Evaluate
        eval_func = get_eval_methods(exp_configs['dataset_name'])
        eval_result_df, infer_result_df = eval_func(merged_anno_id2result, merged_anno_id2meta)

        # Dump inference & evaluation results
        answer_file = os.path.join(exp_configs['output_dir'], "anno_id2result.json")
        infer_res_file = os.path.join(exp_configs['output_dir'], "infer_results.csv")
        eval_res_file = os.path.join(exp_configs['output_dir'], "eval_results.csv")

        with open(answer_file, 'w') as F:
            json.dump(merged_anno_id2result, F)
        infer_result_df.to_csv(infer_res_file, index=False)
        eval_result_df.to_csv(eval_res_file, index=True)

    cleanup()


if __name__ == "__main__":
    args = parse_arguments()
    world_size = args.n_gpus
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
