import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling, Looping_video_via_latent_shift
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
import imageio

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='/path/to/your_models_path/VideoCrafter2/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default='configs/inference_t2v_512_v2.0.yaml', help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default="prompts/samples_vc2.txt", help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default="results_vc2/samples", help="results saving path")
    parser.add_argument("--savefps", type=str, default=8, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=16, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    ## for looping video
    parser.add_argument("--shift_skip", type=int, default=9, help="Set the skip step of latent shift")
    return parser

def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    
    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    prompts, prompt = [], []
    for pt in prompt_list:
        if pt == "...":
            prompts.append(prompt)
            prompt = []
        else:
            prompt.append(pt)

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompts)
    for idx in range(0, n_rounds):
        batch_size = 1
        filenames = []
        filenames.append(f"case_{idx + 1}")
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

        prompt = prompts[idx]
        if isinstance(prompt, str):
            prompt = [prompt]
        print([prompt])
        #prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompt[0])

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps}
        else:
            raise NotImplementedError

        ## inference
        # video_frames = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
        #                                         args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
        ddim_sampler = DDIMSampler(model)
        ddim_sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

        # Looping Video generation
        video_frames = Looping_video_via_latent_shift(
            args, model, cond, noise_shape, ddim_sampler, prompt, args.unconditional_guidance_scale, args.shift_skip,
        ).cpu()

        ## b,samples,c,t,h,w
        save_videos(video_frames, args.savedir, filenames, fps=args.savefps)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)