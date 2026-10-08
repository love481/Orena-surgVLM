# Long Surgical Video Understanding with VideoChat-Flash

Fine-tuning [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash) for question answering and reasoning over long surgical videos, using data from the [ORENA-FOCUS](https://github.com/IMSY-DKFZ/orena-focus) benchmark.

## Overview

The pipeline has three stages:

1. **Data preparation**: download surgical videos, overlay timestamps, and convert annotations into the VideoChat-Flash training format.
2. **Training**: stage-3 video SFT starting from the `UMT-Qwen2_5_1M_7B` checkpoint.
3. **Inference and evaluation**: run the fine-tuned model on the benchmark and score its answers.


## Requirements

- Linux with NVIDIA GPU(s) (multi-GPU recommended for 7B training, 4 Grace Hopper GH200 used)
- Python environments for both repositories (see their READMEs)
- Local clones of:
  - [IMSY-DKFZ/orena-focus](https://github.com/IMSY-DKFZ/orena-focus)
  - [OpenGVLab/VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash)

## 1. Data Preparation

Work inside the `orena-focus` repository. See the [official repo](https://github.com/IMSY-DKFZ/orena-focus) for full details.

```bash
cd orena-focus

# Download surgical videos and overlay timestamps
python download.py

# Create the offline JSON files for VLM training (e.g. Qwen-style format)
python create_data_train.py

# Convert the data into the format required by VideoChat-Flash training
python make_data_chatflash.py
```

| Script | Purpose |
|---|---|
| `download.py` | Downloads videos and burns timestamps into the frames |
| `create_data_train.py` | Builds generic offline JSON annotations for VLM training |
| `make_data_chatflash.py` | Converts annotations to the VideoChat-Flash format |

## 2. VideoChat-Flash Training

Work inside the `VideoChat-Flash/llava-train_videochat` directory. See the [official training guide](https://github.com/OpenGVLab/VideoChat-Flash/blob/main/llava-train_videochat/README.md) for details.

### Steps

1. **Download the stage-3 checkpoint**: `UMT-Qwen2_5_1M_7B`.
2. **Prepare the dataset YAML**: edit `data/stage3_surgvqa.yaml` so it points to the JSON produced by `make_data_chatflash.py` and to your video directory.
3. **Launch training**:

```bash
bash scripts/train/stage3-video_sft/stage3_torch.sh
```

> Make sure the checkpoint path and dataset YAML referenced in `stage3_torch.sh` match the files from steps 1 and 2.

## 3. Inference and Evaluation
Download the trained checkpoints from the following [huggingface link](https://huggingface.co/gc-anurag/SurgVideoChat/tree/main) if you want just inference.

Run from inside `orena-focus`:

```bash
# Generate predictions with the fine-tuned VideoChat-Flash model
python examples/inference_chatflash.py

# Score the predictions
python examples/evaluation.py
```

## Repository Layout (relevant parts)

```
orena-focus/
├── download.py
├── create_data_train.py
├── make_data_chatflash.py
├── ....
└── examples/
    ├── inference_chatflash.py
    └── evaluation.py

VideoChat-Flash/llava-train_videochat/
├── data/stage3_surgvqa.yaml
└── scripts/train/stage3-video_sft/stage3_torch.sh
```

## Acknowledgements
- [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash) by OpenGVLab
