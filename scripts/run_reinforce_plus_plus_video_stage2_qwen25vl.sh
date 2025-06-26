cd src
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/reinforce++/log-cycle1-stage2-M1-Qwen25VL3B-M2-Qwen25VL7B.txt"
# export CUDA_LAUNCH_BLOCKING=1


model1="your-m1-path/cycle1-stage1-M1-Qwen25VL3B-M2-Qwen25VL7B"
model2="your-m2-path/Qwen2.5-VL-7B-Instruct"
output_dir="your-save-path/cycle1-stage2-M1-Qwen25VL3B-M2-Qwen25VL7B"
data_path="your-data-path/2_3_m_youtube_mc_v0_1_qa_random_8k.jsonl"           # 8 and 16 frames


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    reinforce_plus_plus_stage2.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model1} \
    --model_name_or_path_2 ${model2} \
    --dataset_name xxx \
    --jsonl_path ${data_path} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length_stage1 512 \
    --max_completion_length 256 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.02 \
    --weight_decay 0.01 \
    --beta 1e-3 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name cycle1-stage2-M1-Qwen25VL3B-M2-Qwen25VL7B \
    --save_steps 200 \
    --max_grad_norm 20 \
    --save_only_model true \
    --num_generations 1   # M1 is freezed, so we only need to generate 1 output for M2