"""
THis is the main file for the gradio web demo. It uses the CogVideoX-2B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

This demo only supports the text-to-video generation model.
If you wish to use the image-to-video or video-to-video generation models,
please use the gradio_composite_demo to implement the full GUI functionality.

Usage:
    OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

import os
import threading
import time

import gradio as gr
import torch
from diffusers import CogVideoXDPMScheduler
from pipeline.pipeline_cogvideo_mobius import CogVideoXMobiusPipeline
from diffusers.utils import export_to_video
from datetime import datetime, timedelta
import os
from openai import OpenAI
from moviepy import *

MODEL_PATH = "THUDM/CogVideoX-5b"

pipe = CogVideoXMobiusPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

pipe.enable_sequential_cpu_offload()

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

os.makedirs("./results_gradio", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

def infer(prompt: str, shift_skip: int,
          frame_invariance_decoding: bool, progress=gr.Progress(track_tqdm=True)):

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=6.0,
        generator=torch.Generator().manual_seed(42),
        shift_skip=shift_skip,
        frame_invariance_decoding=frame_invariance_decoding,
    ).frames[0]

    return video


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./results_gradio/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    gif_path = video_path.replace(".mp4", ".gif")
    export_to_video(tensor, gif_path, fps=8)
    return video_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./results_gradio", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               Mobius Gradio Simple Space🤗
            """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            with gr.Column():

                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Set the skip step of latent shift.<br>"
                    "Frame-Invariance Decoding is used to alleviate first frame artifacts.<br>"
                )
                with gr.Row():
                    shift_skip = gr.Number(label="🎢Shift Skip", value=6)
                    frame_invariance_decoding = gr.Checkbox(label="✨Frame-Invariance Decoding", value=True)

                generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="Mobius Generate Video", width=720, height=480, autoplay=True, loop=True)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)

    gr.Markdown("""
    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            🎥 Video Gallery
        </div>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A sleepy koala, nestled comfortably on a tree branch, lazily munches on eucalyptus leaves, its fluffy grey fur blending with the textured bark of the tree. The leaves sway slightly in the breeze as the koala picks them one by one, its black nose twitching with each bite.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/8f5da99a-ee55-4b01-895c-c45ecb56795b" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A woman stands behind a kitchen counter, expertly blending a mix of fresh fruits—bananas, strawberries, and spinach—into a vibrant smoothie. The blender whirs, and the colorful ingredients combine, creating a swirling blend of red and green hues. As she pours the finished smoothie into a glass, the scene conveys a sense of health and vitality, with the kitchen’s clean, modern decor highlighting the simple, wholesome process.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/f8034bcd-06ac-4851-9cfc-bbb2eece4b35" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A young female activist stands tall, holding a flag high above her head with determination in her eyes. The flag flutters in the breeze, its bold colors contrasting with the backdrop of a city street or public space. Her posture is confident, embodying strength and resolve as she becomes a symbol of the cause she represents. The surrounding environment captures the energy and passion of her movement.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/48972e51-e844-44ad-8923-6f5823fdbcd3" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A woman in a flowing, white sundress and sunglasses, her hair tousled by the sea breeze, runs along a golden sandy beach as the late afternoon sun casts long shadows. The ocean's waves crash rhythmically in the background, and seagulls cry overhead. Her determined stride contrasts with the serene sunset hues painting the sky, capturing a moment of freedom and escape as the day transitions to evening. The scene embodies the perfect blend of tranquility and vitality, with the woman's silhouette framed against the fading light.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/c80e7037-373f-4005-9e01-1e2d0223d1d3" width="100%" controls autoplay loop></video>
            </td>
        </tr>
        <tr>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A rugged man in a sleek, insulated black ski jacket and matching ski pants glides effortlessly down a pristine, powdery white slope, his movements fluid and graceful. His goggles reflect the brilliant sunlight, and his breath forms visible puffs in the crisp mountain air. The backdrop is a breathtaking panorama of snow-capped peaks and towering pine trees, their branches lightly dusted with snow. As he navigates the slopes, his ski poles rhythmically punctuate the snow, leaving a trail of precise, parallel lines. The scene transitions to a close-up of his focused expression, the wind tousling his hair, capturing the exhilaration and freedom of the sport.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/a5cc0ade-7037-4812-a570-85f624f4ac22" width="100%" controls autoplay loop></video>
            </td>
            <td style="width: 25%; vertical-align: top; font-size: 0.9em;">
                <p>A hacker, surrounded by holographic screens, works feverishly in a dimly lit, tech-heavy room. The digital interference around them creates an intense atmosphere of high-tech rebellion in the cyberpunk world.</p>
            </td>
            <td style="width: 25%; vertical-align: top;">
                <video src="https://github.com/user-attachments/assets/ef922145-79b5-4299-a5e7-d3f03eb81274" width="100%" controls autoplay loop></video>
            </td>
        </tr>
    </table>
        """)

    def generate(prompt, shift_skip, frame_invariance_decoding, model_choice, progress=gr.Progress(track_tqdm=True)):
        if not prompt.strip():
            return gr.Error("Prompt cannot be empty. Please enter a valid prompt.")

        tensor = infer(prompt, shift_skip, frame_invariance_decoding, progress=progress)
        video_path = save_video(tensor)
        video_update = gr.update(visible=True, value=video_path)
        gif_path = video_path.replace(".mp4", ".gif")
        gif_update = gr.update(visible=True, value=gif_path)

        return video_path, video_update, gif_update

    generate_button.click(
        generate,
        inputs=[prompt, shift_skip, frame_invariance_decoding],
        outputs=[video_output, download_video_button, download_gif_button],
    )

if __name__ == "__main__":
    demo.launch()
