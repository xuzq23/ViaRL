# ViaRL
[ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning](https://arxiv.org/abs/2505.15447)


## Setup
```bash
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
```bash
bash scripts/run_reinforce_plus_plus_video_stage1_qwen25vl.sh
```

cycle1-stage2/cycle2-stage2
```bash
bash scripts/run_reinforce_plus_plus_video_stage2_qwen25vl.sh
```


## Evaluation
The relevant code can be found in the evaluation directory.

```bash
cd evaluation
```

- Step1: Download the models from https://huggingface.co/ViaRL

- Step2: Prepare the datasets following the docs.
  - Prepare [VideoMME](evaluation/docs/prepare_videomme.md)
  - Prepare [MLVU](evaluation/docs/prepare_mlvu.md)
  - Prepare [LVBench](evaluation/docs/prepare_lvbench.md)

- Step3: Run script
Note: Please set model_path1/model_path2/n_gpus in the bash file. 

  ```bash
  bash scripts/infer_eval_qwenvl_videomme_rl.sh
  bash scripts/infer_eval_qwenvl_mlvu_rl.sh
  bash scripts/infer_eval_qwenvl_lvbench_rl.sh
  ```


## Citation
If you find this work useful, please cite
```
@misc{xu2025viarladaptivetemporalgrounding,
      title={ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning}, 
      author={Ziqiang Xu and Qi Dai and Tian Xie and Yifan Yang and Kai Qiu and DongDong Chen and Zuxuan Wu and Chong Luo},
      year={2025},
      eprint={2505.15447},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.15447}, 
}
```