from transformers import AutoModel, AutoTokenizer
import torch

# model setting
# model_path = '/iopsstor/scratch/cscs/lpanta32/checkpoints/VideoChat-Flash-Qwen2_5-7B-1M_res224'
model_path = '/iopsstor/scratch/cscs/lpanta32/checkpoints/stage3-video_sft/stage3-surgvqa_overlay_refined'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path,  attn_implementation="sdpa", trust_remote_code=True, torch_dtype=torch.float16, device_map = "auto").eval()
# image_processor = model.get_vision_tower().image_processor

mm_llm_compress = False # use the global compress or not
if mm_llm_compress:
    model.config.mm_llm_compress = True
    model.config.llm_compress_type = "uniform0_attention"
    model.config.llm_compress_layer_list = [4, 18]
    model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
else:
    model.config.mm_llm_compress = False

# evaluation setting
max_num_frames = 100
generation_config = dict(
    do_sample=False,
    temperature=0.2,
    max_new_tokens=100,
    top_p=0.1,
    num_beams=1
)

def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to hh:mm:ss format."""
    seconds = int(round(seconds))
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

convert_time = 230
convert_time = seconds_to_hhmmss(convert_time)

time_info1 = (
                f" This video clip is extracted from a longer procedure video and starts at "
                f"{convert_time} (hh:mm:ss) within that full video. Any timestamp mentioned in "
                "the question is given relative to the full video, not this clip. To find the "
                "relevant moment in this clip, subtract this start time from the question's "
                "timestamp. When your answer includes a timestamp, add this start time back so "
                "it is expressed relative to the full video, not this clip. "
)

video_path = "/iopsstor/scratch/cscs/lpanta32/focus/lapchole_clips_test/tmpmhmwygn1.mp4"

# single-turn conversation
question1 = "What is this video about?"
output1, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question1, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

print(output1)

# multi-turn conversation
question2 = f"What instruments is visible and what is happening at 00:04:00 . {time_info1}"
output2, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question2, chat_history=chat_history, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

print(output2)
