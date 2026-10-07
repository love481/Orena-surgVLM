from llava.model.multimodal_encoder.builder import build_vision_tower
from types import SimpleNamespace
import torch
model_args = SimpleNamespace(**{
    "mm_vision_tower": "vjepa2-large",
    "mm_vision_tower_pretrained": True,
    "mm_vision_select_layer": -2,
    "mm_local_num_frames": 4,
})
device = "cuda:0" if torch.cuda.is_available() else "cpu"
input_image = torch.randn(1, 4, 3, 224, 224).to(dtype=torch.bfloat16, device=device)  # B T C H W
vision_tower = getattr(model_args, "mm_vision_tower", getattr(model_args, "vision_tower", None))

vision_tower = build_vision_tower(model_args,pt_type='origin').to(dtype=torch.bfloat16, device=device) 
print(vision_tower)
output = vision_tower(input_image)
print(output.shape)


