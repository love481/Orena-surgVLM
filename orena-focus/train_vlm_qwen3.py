from focus import Track
from focus import FocusConfig, set_config

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    AutoConfig,
    )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import json
from focus.data.video_dataset import FocusVideoDataset, VideoSample
import wandb
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List, Any, Dict
from transformers.video_utils import VideoMetadata
import decord, re
import numpy as np
from qwen_vl_utils.vision_process import (
    smart_resize,
    SPATIAL_MERGE_SIZE,
    VIDEO_MIN_TOKEN_NUM,
    VIDEO_MAX_TOKEN_NUM,
    FRAME_FACTOR,
    MODEL_SEQ_LEN,
)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import logging

logger = logging.getLogger(__name__)
from torchvision.transforms import v2

TRANSFORM = v2.CenterCrop((360, 420))
VIDEO_MAX_TOKEN_NUM = 90
VIDEO_MIN_TOKEN_NUM = 45
MODEL_SEQ_LEN = 40000


def resize_custom_video(video_tensor, image_patch_size=14, ele=None):
    """
    Applies the same resize / pixel-budget logic fetch_video() uses internally,
    to a video tensor you've already extracted/sampled yourself.

    video_tensor: (T, C, H, W) tensor, raw/native resolution
    ele: optional dict, same keys fetch_video() would accept
         (min_pixels, max_pixels, total_pixels, resized_height, resized_width)
    """
    ele = ele or {}
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    #video_tensor = TRANSFORM(video_tensor) ## cropping the black area
    nframes, _, height, width = video_tensor.shape

    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get(
        "total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9
    )
    max_pixels = max(
        min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,

        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    video_tensor = transforms.functional.resize(
        video_tensor,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    return video_tensor


os.environ["WANDB_PROJECT"] = "vqa_check"
# wandb.init(project="vqa_check")
##COnfig for the vqa datasets
CONFIG = {
    "root_dir": "/iopsstor/scratch/cscs/lpanta32/focus",
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dataset_name": "heico",
    "track": Track.SEGMENT,
    "video_stride": 25,  # keep every 25th frame (1 fps at 25 fps source)
    "video_resolution": (640, 360),
    "use_overlay": False,  # use timestamp-overlaid videos
    "num_eval": None,  # set to None to evaluate all test samples
    "output_dir": None,  # set to a path to persist results CSV files
}

set_config(FocusConfig(root_dir=CONFIG["root_dir"]))


def build_metadata_from_offset(num_frames, native_fps, new_fps, abs_start_frame):
    """
    num_frames      : number of frames in your resampled (e.g. 1fps) video/frame-list
    native_fps      : fps of the ORIGINAL video (e.g. 25)
    new_fps         : fps of your resampled/saved video (e.g. 1)
    abs_start_frame : frame index in the ORIGINAL video where this clip starts (e.g. 1000)
    """
    # start_seconds = abs_start_frame / native_fps
    #
    # # frames_indices must be integers that, divided by `fps` below, reproduce real_time(i)
    # frames_indices = [round(start_seconds * new_fps) + i for i in range(num_frames)]
    frames_indices = [round(abs_start_frame + i * new_fps) for i in range(num_frames)]
    return VideoMetadata(
        fps=new_fps,
        frames_indices=frames_indices,
        total_num_frames=num_frames,
        duration=num_frames / new_fps,
    )


## test setup
# meta = build_metadata_from_offset(num_frames=10, native_fps=25, new_fps=1, abs_start_frame=1000)
# real_times = [idx / meta.fps for idx in meta.frames_indices]
# print(real_times)
# # abs_start_frame=1000 at 25fps -> starts at 40.0s
# # -> [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0]


# Open and load the JSON file
with open(
    os.path.join(CONFIG["root_dir"], "heico_train.json"), "r", encoding="utf-8"
) as file:
    dataset_heico = json.load(file)
with open(
    os.path.join(CONFIG["root_dir"], "lapchole_train.json"), "r", encoding="utf-8"
) as file:
    dataset_lapchole = json.load(file)

Additional_time_information = " Make sure that the time indices token are provided in seconds but question are asked in hh:mm:ss format."

dataset_json = []
for i, each_sample in enumerate(dataset_heico.values()):
    each_sample[1]["content"][0]["resized_height"] = CONFIG["video_resolution"][1]
    each_sample[1]["content"][0]["resized_width"] = CONFIG["video_resolution"][0]
    each_sample[0]["content"] = each_sample[0]["content"] + Additional_time_information
    dataset_json.append(each_sample)

for i, each_sample in enumerate(dataset_lapchole.values()):
    each_sample[1]["content"][0]["resized_height"] = CONFIG["video_resolution"][1]
    each_sample[1]["content"][0]["resized_width"] = CONFIG["video_resolution"][0]
    each_sample[0]["content"] = each_sample[0]["content"] + Additional_time_information
    dataset_json.append(each_sample)


MODEL_ID = "/iopsstor/scratch/cscs/lpanta32/checkpoints/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_ID)
config.text_config.rope_theta = 500000
config.vision_config.temporal_patch_size = 20

config.vision_config.patch_size = 16
config.vision_config.spatial_merge_size = 2



ignore_index = -100
def format_time(sec):
    s = int(round(sec))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def collate_fn(all_msgs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    texts = [
        processor.apply_chat_template(
            msgs[:3], tokenize=False, add_generation_prompt=False
        )
        for msgs in all_msgs
    ]
    ele = {}
    ele["resized_height"] = CONFIG["video_resolution"][1]
    ele["resized_width"] = CONFIG["video_resolution"][0]
    video_inputs = []
    video_metadatas = []
    for msgs in all_msgs:
        for i in range(len(msgs[1]["content"])):
            if "video" in msgs[1]["content"][i]:
                vr = decord.VideoReader(str(msgs[1]["content"][i]["video"]))
                s_r = 1
                target_fps = vr.get_avg_fps() * s_r  # double the native fps
                duration = len(vr) / vr.get_avg_fps()
                num_frames = int(duration * target_fps)
                frame_indices = [
                    int(i * len(vr) / num_frames) for i in range(num_frames)
                ]
                frame_indices = [min(idx, len(vr) - 1) for idx in frame_indices]
                frames = vr.get_batch(frame_indices).asnumpy()
                video_tensor = (
                    torch.tensor(frames).permute(0, 3, 1, 2).float()
                )  # (T, C, H, W)

                
                patch_size = (
                    processor.image_processor.patch_size
                )  # confirm this matches your model, e.g. 16
                video_tensor = resize_custom_video(
                    video_tensor, image_patch_size=patch_size, ele=ele
                )
                # print("video_size",video_tensor.shape)
                metadata = build_metadata_from_offset(
                    num_frames=num_frames,
                    native_fps=25,
                    new_fps=msgs[1]["content"][i]["fps"] * s_r,
                    abs_start_frame=msgs[3]["start_time"],
                )
                video_metadatas.append(metadata)
                video_inputs.append(video_tensor)
    if not video_inputs:
        video_inputs = None
        video_metadatas = None
    else:
        assert all(
            m["fps"] is not None for m in video_metadatas
        ), "some video_metadata has fps=None — timestamps will silently default to fps=24"
    # print(np.array(video_inputs).shape)
    batch = processor(
        text=texts,
        videos=video_inputs,
        video_metadata=video_metadatas,
        return_tensors="pt",
        padding=True,
        do_resize=False,
        do_sample_frames=False,
        temporal_patch_size=20,
        merge_size=2,
        patch_size=16,

    )
    # print(list(batch.keys()))
    # print("input_ids ", batch["input_ids"].shape)  # (1, seq_len)
    # print("video grid thw ", batch["video_grid_thw"])  # (num_groups, 3) -> [t=1, h, w] per group
    # print("pixel_values_videos ",batch["pixel_values_videos"].shape)
    # print("attention_mask ",batch["attention_mask"].shape)
    # skip_ids = {
    #     processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
    #     # processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
    # }

    # ids_no_pad = [i for i in batch["input_ids"][0].tolist() if i not in skip_ids]
    # decoded = processor.tokenizer.decode(ids_no_pad)
    # print(decoded)


    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_index

    # mask video pad tokens out of the loss
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    labels[labels == video_token_id] = ignore_index

    batch["labels"] = labels
    return batch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-8B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "qwen2.5-vl": {
        "vision_encoder": [
            "visual.patch_embed",
            "visual.rotary_pos_emb",
            "visual.blocks",
        ],
        "vision_projector": ["visual.merger"],
        "llm": ["model"],
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "vision_encoder": [
            "model.visual",
        ],
        "vision_projector": ["model.visual.merger","model.visual.deepstack_merger_list"],
        "llm": ["model.language_model"],
    },
}

## Our model parameters to tune
model_args = ModelArguments(
    model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
    tune_mm_llm=True,
    tune_mm_mlp=True,
    tune_mm_vision=True,
)
use_lora = True
use_vision_lora = False

# freeze certain params
# # Define the 4-bit quantization configuration
vision_encoder_keys = MODULE_KEYWORDS[model_args.model_name_or_path]["vision_encoder"]
vision_projector_keys = MODULE_KEYWORDS[model_args.model_name_or_path][
    "vision_projector"
]
llm_keys = MODULE_KEYWORDS[model_args.model_name_or_path]["llm"]


skip_quantize = []
if not (model_args.tune_mm_vision and use_vision_lora) and model_args.tune_mm_vision:
    skip_quantize.extend(vision_encoder_keys)
if not use_lora:
    skip_quantize.extend(llm_keys)

if model_args.tune_mm_mlp:
    skip_quantize.extend(vision_projector_keys)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    # quantization_config=quantization_config,
    device_map="auto",
    config=config,
    attn_implementation="eager",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
)

print(model)
if not model_args.tune_mm_vision:
    print(f"Vision encoder is freezed... including:")
    for module in vision_encoder_keys:
        eval(f"model.{module}").requires_grad_(False)
else:
    print(f"Vision encoder is unfreezed... including:")
    for module in vision_encoder_keys:
        eval(f"model.{module}").requires_grad_(True)



if not model_args.tune_mm_mlp:
    print(f"Vision projector is freezed... including:")
    for module in vision_projector_keys:
        eval(f"model.{module}").requires_grad_(False)
else:
    print(f"Vision projector is unfreezed... including:")
    for module in vision_projector_keys:
        eval(f"model.{module}").requires_grad_(True)

if not model_args.tune_mm_llm:
    print(f"llm is freezed... including:")
    for module in llm_keys:
        eval(f"model.{module}").requires_grad_(False)
else:
    print(f"llm is unfreezed... including:")
    for module in llm_keys:
        eval(f"model.{module}").requires_grad_(True)

# other components preparation (e.g., image_newline, vision_resampler)
# we will just freeze these
if "others" in MODULE_KEYWORDS[model_args.model_name_or_path]:
    print(f"Other multimodal component is freezed... including:")
    for other_key in MODULE_KEYWORDS[model_args.model_name_or_path]["others"]:
        eval(f"model.{other_key}").requires_grad_(False)

# lora preparation

named_modules = {n: m for n, m in model.named_modules()}
lora_modules = []
full_modules = []


## fining all lienar leayer to finetune the qwen model
def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if "lm_head" in name:  # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


if model_args.tune_mm_vision and use_vision_lora:
    print("LoRA for vision encoder enabled...")
    lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
elif model_args.tune_mm_vision:
    print("Vision encoder will be fully trained...")
    full_modules.extend(vision_encoder_keys)

if model_args.tune_mm_llm:
    if use_lora:
        print("LoRA for LLM enabled...")
        lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
    else:
        print("LLM will be fully trained...")
        full_modules.extend(llm_keys)

if model_args.tune_mm_mlp:
    print("Vision projector will be fully trained...")
    full_modules.extend(vision_projector_keys)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
##Lora config and others
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=lora_modules,
    modules_to_save=full_modules,
    # target_modules=[
    #     "q_proj", "k_proj", "v_proj", "o_proj",
    #     "gate_proj", "up_proj", "down_proj"
    # ],
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_params:,} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}"
    )

print_trainable_parameters(model)

## Training pipeline
BATCH_SIZE = 4
GRAD_ACCUM = 16
EPOCS = 1
num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
num_training_steps = (len(dataset_json) // (BATCH_SIZE * GRAD_ACCUM * num_gpus)) * EPOCS
# num_training_steps = (
#     len(dataset_json) // (BATCH_SIZE * GRAD_ACCUM)
# ) * EPOCS  # samples / (batch * grad_accum) * epochs
#
args = TrainingArguments(
    output_dir="/iopsstor/scratch/cscs/lpanta32/checkpoints/qwen3vl-8b-lora_tune",
    num_train_epochs=EPOCS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=int(
        0.03 * num_training_steps
    ),  # warmup_ratio is deprecated in transformers>=5
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_json,
    data_collator=collate_fn,
    processing_class=processor,
)

trainer.train()
trainer.save_model("/iopsstor/scratch/cscs/lpanta32/checkpoints/qwen3vl-8b-lora_tune/final")