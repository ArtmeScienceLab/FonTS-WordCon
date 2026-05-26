import os
import sys
from safetensors import safe_open
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) 

import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image
from src.flux.generate_ours import generate, seed_everything
from safetensors.torch import load_file
from src.flux.lora_utils_ours import load_lora

local_path_dev = 'pretrianed/FLUX.1-dev'
pipe = FluxPipeline.from_pretrained(
    local_path_dev, torch_dtype=torch.bfloat16
)

lora_path = 'wordcon_weights/pytorch_lora_weights.safetensors'

pipe = load_lora(
    pipe, 
    lora_path, 
    use_adapter_method=False 
)

GPU_id = 0
pipe = pipe.to(f"cuda:{GPU_id}")

save_path = 'save_your_results' 
if not os.path.exists(save_path):
    os.makedirs(save_path)


pairs = {
    0: {
        'prompt': "A large road sign at a countryside intersection surrounded by golden wheat fields, presenting black Text: \"What a trifle scares you!\", make 'scares' underline"
    },
    1: {
        'prompt': "This basketball features the bold slogan 'Shoot Your Shot'printed across its surface, make 'Shoot' bold."
    }
}

for pair_id, pair in pairs.items():        
    prompt = pair["prompt"]

    seeds = [7,8,38,41,45,48,50,53,57] 
    for seed in seeds:
        seed_everything(seed)
        result_img = generate(
            pipe,
            prompt=prompt,
            num_inference_steps=50,
            height=512,
            width=512,
        ).images[0]

        save_name = os.path.join(save_path, f'img_pair={pair_id}_{seed}.png')
        result_img.save(save_name)
        print(f"The results are saved at: {save_name}")
