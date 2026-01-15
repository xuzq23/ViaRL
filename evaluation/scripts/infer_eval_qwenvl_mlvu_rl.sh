export PYTHONPATH="./:$PYTHONPATH"


python viarl/infer_eval_batch.py \
    --model_path1 your-m1-path \
    --model_path2 your-m2-path \
    --config_path configs/qwen25vl_mlvu_rl.yaml \
    --video_frame_extraction_fps 25 \
    --n_gpus 4
