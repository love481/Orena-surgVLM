import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "/iopsstor/scratch/cscs/lpanta32/checkpoints/uAI-NEXUS-MedVLM-1.0a-7B-RL"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

video_path = "/iopsstor/scratch/cscs/lpanta32/focus/lapchole_clips_test/tmpmhmwygn1.mp4"

# Set desired sampling strategy:
# Option A: Set target FPS (e.g., 1 frame per second, 2 fps, 0.5 fps)
TARGET_FPS = 0.5

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": video_path,
            "fps": TARGET_FPS,          # <--- Controls frame rate sampling dynamically
            # "nframes": 32,             # <--- Alternatively: Force N uniformly spaced frames regardless of duration
            # "min_pixels": 256 * 28 * 28, # <--- Optional resolution controls
            # "max_pixels": 1280 * 28 * 28,
        },
        {"type": "text", "text": "can you describe the event after 00:34:00 until that white object is removed?"},
    ],
}]

# Prepare inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Generate response
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)