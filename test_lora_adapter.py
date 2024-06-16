import os
import argparse
import torch
from PIL import Image
import numpy as np

from t2i_adapters import patch_pipe, Adapter, sketch_extracter

from t2i_adapters.api import get_cond_canny, get_cond_depth
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

DEFAULT_NEGATIVE_PROMPT = "ugly, bad anatomy, bad proportions, bad quality, blurry, cropped, \
deformed, error, worst quality, low quality, jpeg, jpeg artifacts, pixel, pixelated, grainy"

if __name__ == "__main__":
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help='path to checkpoint of stable diffusion model',
    )
    parser.add_argument(
        '--LoRA_ckp',
        type=str,
        default='./contents/pytorch_lora_weights.safetensors',
        help='path to checkpoint of LoRA, .safetensor are supported',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default='1',
        help='number of generated samples',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of denoising steps',
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )
    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="the photo of living prompt",
        help='prompt',
    )
    parser.add_argument(
        '--cond_img_path',
        type=str,
        default="./example/img1.png",
        help='condition image path',
    )
    parser.add_argument(
        '--cond_name',
        type=str,
        default="canny",
        help='Condition name',
    )
    parser.add_argument(
        '--resize_short_edge',
        type=int,
        default=512,
        help='Resize Short Edge',
    )
    global_opt = parser.parse_args()

    global_opt.max_resolution = 512 * 512
    global_opt.sampler = 'ddim'
    global_opt.cond_weight = 1.0
    global_opt.C = 4
    global_opt.f = 8
    global_opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    # 0. Define model

    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    patch_pipe(pipe)

    pipe.load_lora_weights(global_opt.LoRA_ckp)
    pipe.fuse_lora(lora_scale = 0.9)

    for ext_type, prompt in [(global_opt.cond_name, global_opt.prompt)]:
        adapter = Adapter.from_pretrained(ext_type).to(device)

        # 2. Prepare Condition via adapter.
        cond_img_src = Image.open(global_opt.cond_img_path)

        if ext_type == "canny":
            cond_img, cond_img_tensor = get_cond_canny(global_opt, global_opt.cond_img_path)

        if ext_type == "depth":
            cond_img, cond_img_tensor = get_cond_depth(global_opt, global_opt.cond_img_path)

        with torch.no_grad():
            adapter_features = adapter(cond_img_tensor)

        pipe.unet.set_adapter_features(adapter_features)

        pipe.safety_checker = None
        neg_prompt = global_opt.neg_prompt
        torch.manual_seed(1)

        imgs = pipe(
            [global_opt.prompt] * global_opt.num_samples,
            negative_prompt=[global_opt.neg_prompt] * global_opt.num_samples,
            num_inference_steps=global_opt.steps,
            guidance_scale=global_opt.guidance_scale,
            height=cond_img_tensor.shape[2],
            width=cond_img_tensor.shape[3],
        ).images

        img_name, _ = os.path.splitext(os.path.basename(global_opt.cond_img_path))

        #save result
        for i, img in enumerate(imgs):
            img.save(f"examples/result_{img_name}_{i}_{ext_type}.jpg")

        cv2.imwrite(f"examples/{ext_type}_{img_name}.png", cond_img)
