import argparse
import os
from pathlib import Path
import torch
from diffusers import CogVideoXDPMScheduler
from pipeline.pipeline_cogvideo_mobius import CogVideoXMobiusPipeline
from diffusers.utils import export_to_video

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    shift_skip: int = 6,
    frame_invariance_decoding: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    pipe = CogVideoXMobiusPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    # pipe.to("cuda")

    pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.

    video_generate = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        shift_skip=shift_skip,
        frame_invariance_decoding=frame_invariance_decoding,
    ).frames[0]

    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    parent_dir = Path(output_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    export_to_video(video_generate, output_path, fps=8)
    output_gif_path = output_path[:-4] + ".gif"
    export_to_video(video_generate, output_gif_path, fps=8)

def load_prompts(prompts_path):
    prompts = []
    with open(prompts_path, 'r', encoding='utf-8') as file:
        prompt = []
        for line in file:
            line = line.strip()
            if line != "...":
                prompt.append(str(line))
            else:
                prompts.append(prompt)
                prompt = []
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a looping video from a text prompt using CogVideoX")
    parser.add_argument("--prompts_path", type=str, required=True, help="The path to the description of the video to generate")
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--shift_skip", type=int, default=6, help="Set the skip step of latent shift")
    parser.add_argument("--frame_invariance_decoding", action="store_true", default=False, help="Enable or disable frame invariance decoding")

    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    prompts = load_prompts(args.prompts_path)

    for i, prompt in enumerate(prompts):
        print(prompt)
        output_path = os.path.join(args.output_path, f"case_{i + 1}.mp4")

        generate_video(
            prompt=prompt,
            model_path=args.model_path,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            output_path=output_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos_per_prompt,
            dtype=dtype,
            seed=args.seed,
            shift_skip=args.shift_skip,
            frame_invariance_decoding=args.frame_invariance_decoding,
        )
        print(f"case_{i + 1} have been generated.")
