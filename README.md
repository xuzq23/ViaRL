# ViaRL
[ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning](https://arxiv.org/abs/2505.15447)


## Setup
```
conda create -n viarl python=3.10 -y
conda activate viarl
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1
pip install transformers==4.51.3
```

## Dataset
Please download LLaVA-Video-178k firstly. Next, download our jsonl files.

## Training
Note: Please set model1/model2/output_dir/data_path in the bash file. 
Training sequence: cycle1-stage1 -> cycle1-stage2 -> cycle2-stage1 -> cycle2-stage2.
Update model1 or model2 after the corresponding training.

cycle1-stage1/cycle2-stage1
```
bash scripts/run_reinforce_plus_plus_video_stage1_qwen25vl.sh
```

cycle1-stage2/cycle2-stage2
```
bash scripts/run_reinforce_plus_plus_video_stage2_qwen25vl.sh
```


## Evaluation
https://huggingface.co/ViaRL
