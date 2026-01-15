## Prepare MLVU Dataset


### Step 1: download MLVU dataset from [huggingface](https://huggingface.co/datasets/MLVU/MVLU)
```bash
git clone https://huggingface.co/datasets/MLVU/MVLU
```

Denote the root directory of download MLVU dataset as `mlvu_root`, it should has the following structure:
```
${mlvu_root}/
├── MLVU/
    ├── json
        ...
    ├── video
        ...
├── figs/
```


### Step 2: Build MLVU dataset
```bash
cd evaluation
python scripts/utils/build_mlvu_dataset.py --hf_root ${mlvu_root}
```

The file structure is as follows:
```
evaluation/
├── dataset/
    ├── mlvu/
        ├── mlvu.json
├── ...
```