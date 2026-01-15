## Prepare LVBench Dataset


### Step 1: download LVBench data from [huggingface](https://huggingface.co/datasets/THUDM/LVBench/tree/main)
```bash
git clone https://huggingface.co/datasets/THUDM/LVBench # Contain annotations only
git clone https://huggingface.co/datasets/AIWinter/LVBench # Contain videos only
```
Move all_files in `AIWinter/LVBench` into `THUDM/LVBench`.

Denote the root directory of download LVBench dataset as `lvbench_root`, it should has the following structure:
```
${lvbench_root}/
├── docs/
├── video_info.meta.jsonl
├── all_videos_split.zip.001
├── all_videos_split.zip.002
├── ...
└── all_videos_split.zip.014
```


### Step 2: Unzip everything
```bash
cd ${lvbench_root}
cat all_videos_split.zip.* > all_videos.zip
unzip all_videos.zip
```


### Step 3: Build LVBench dataset
```bash
cd evaluation
python scripts/utils/build_lvbench_dataset.py --hf_root ${lvbench_root}
```

The file structure is as follows:
```
evaluation/
├── dataset/
    ├── lvbench/
        ├── lvbench.json
├── ...
```