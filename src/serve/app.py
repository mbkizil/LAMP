# ---------------------------------------------------------------------------------
# This script is adapted from the Qwen-VL-Series-Finetune repository.
# Original Source: https://github.com/2U1/Qwen-VL-Series-Finetune
# ---------------------------------------------------------------------------------


import argparse
import warnings
import os
import subprocess
import shutil
from threading import Thread
from transformers import TextIteratorStreamer
import gradio as gr

from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

TEMP_DIR = "temp"

def process_camera_text(cam_text: str):
    """Processes Qwen output (cam.txt) to create the cam_json/ directory."""
    cam_filepath = os.path.join(TEMP_DIR, "cam.txt")
    cam_tag_filepath = os.path.join(TEMP_DIR, "cam_tag.txt") 
    cam_traj_filepath = os.path.join(TEMP_DIR, "cam_traj.txt")
    cam_traj_matrix_filepath = os.path.join(TEMP_DIR, "cam_traj_matrix.txt")
    cam_json_dir = os.path.join(TEMP_DIR, "cam_json")
    
    # Required for fallback script: path to bbox.txt
    bbox_filepath = os.path.join(TEMP_DIR, "bbox.txt") 
    bbox_traj_path = os.path.join(TEMP_DIR, "bbox_traj.txt")
    
    os.makedirs(cam_json_dir, exist_ok=True)
    
    with open(cam_filepath, 'w', encoding='utf-8') as f:
        f.write(cam_text)
    
    try:
        print("Running qwen_to_tag...")
        subprocess.run(["python", "src/scripts/qwen_to_tag.py", cam_filepath, cam_tag_filepath], check=True, text=True, capture_output=True)
        
        # === FALLBACK MECHANISM ===
        print("Running tag_to_traj (Attempt 1)...")
        try:
            # 1. Try the main script
            subprocess.run(["python", "src/scripts/generate_camera_trajectory.py", cam_tag_filepath, cam_traj_filepath], check=True, text=True, capture_output=True)
            print("Attempt 1 (generate_camera_trajectory.py) successful.")
        except subprocess.CalledProcessError as e1:
            # 2. If main script fails: Try fallback script
            print(f"Attempt 1 failed: {e1.stderr}. Trying fallback...")
            
            # Fallback requires 'bbox.txt', check existence
            if not os.path.exists(bbox_filepath):
                print(f"ERROR: {bbox_filepath} not found for fallback.")
                raise FileNotFoundError(f"Fallback dependency missing: {bbox_filepath}")

            fallback_command = [
                "python", 
                "src/scripts/generate_relative_trajectory.py",
                "-obj", bbox_traj_path,
                "-tag", cam_tag_filepath,
                "-o", cam_traj_filepath
            ]
            
            print(f"Running fallback: {' '.join(fallback_command)}")
            subprocess.run(fallback_command, check=True, text=True, capture_output=True)
            print("Attempt 2 (generate_relative_trajectory.py) successful.")
        # === END FALLBACK MECHANISM ===

        print("Running traj_to_et...")
        subprocess.run(["python", "src/scripts/traj_to_et.py", cam_traj_filepath, cam_traj_matrix_filepath], check=True, text=True, capture_output=True)
        
        print("Running et_to_json...")
        subprocess.run(["python", "src/scripts/et_to_json.py", cam_traj_matrix_filepath, cam_json_dir], check=True, text=True, capture_output=True)
        
        print("Camera processing complete.")
        return True, None
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR (Camera Chain):\nCommand: {' '.join(e.cmd)}\nstdout: {e.stdout}\nstderr: {e.stderr}"
        print(error_msg)
        return False, error_msg
    except FileNotFoundError as e:
        error_msg = f"ERROR: Script or file not found: {e}"
        print(error_msg)
        return False, error_msg

def process_bbox_text(bbox_text: str):
    """Processes Qwen output (bbox.txt) to create bbox.json."""
    bbox_filepath = os.path.join(TEMP_DIR, "bbox.txt")
    bbox_json_filepath = os.path.join(TEMP_DIR, "bbox.json")
    bbox_traj_path = os.path.join(TEMP_DIR, "bbox_traj.txt")
    
    with open(bbox_filepath, 'w', encoding='utf-8') as f:
        f.write(bbox_text)
    
    try:
        print("Bbox tags to traj conversion complete.")
        subprocess.run(["python", "src/scripts/qwen_to_tag.py", bbox_filepath, bbox_filepath], check=True, text=True, capture_output=True)
        subprocess.run(["python", "src/scripts/bbox_to_traj.py", bbox_filepath, bbox_traj_path], check=True, text=True, capture_output=True)
        print("Bbox traj generation complete.")
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR (Bbox to Traj):\nstdout: {e.stdout}\nstderr: {e.stderr}"
        print(error_msg)
        return False, error_msg
        
    try:
        print("Running object_to_json...")
        subprocess.run(["python", "src/scripts/object_to_json.py", bbox_traj_path, bbox_json_filepath], check=True, text=True, capture_output=True)
        print("Bbox processing complete.")
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR (Bbox Chain):\nstdout: {e.stdout}\nstderr: {e.stderr}"
        print(error_msg)
        return False, error_msg
    except FileNotFoundError as e:
        error_msg = f"ERROR: Script not found: {e.filename}"
        print(error_msg)
        return False, error_msg

def run_final_projection():
    """Runs the final projection script to generate the video."""
    bbox_json_filepath = os.path.join(TEMP_DIR, "bbox.json")
    cam_first_frame_json = os.path.join(TEMP_DIR, "cam_json", "0000.json")
    cam_json_dir = os.path.join(TEMP_DIR, "cam_json")
    final_video_path = os.path.join(cam_json_dir, "0000_projection.mp4")

    if not os.path.exists(bbox_json_filepath):
        return None, f"ERROR: {bbox_json_filepath} not found. Please generate Bbox first."
    if not os.path.exists(cam_first_frame_json):
        return None, f"ERROR: {cam_first_frame_json} not found. Camera processing might have failed."

    try:
        print("Running projection.py...")
        subprocess.run([
            "python", "src/scripts/projection.py", 
            bbox_json_filepath, 
            cam_first_frame_json, 
            cam_json_dir
        ], check=True, text=True, capture_output=True)
        
        if os.path.exists(final_video_path):
            print(f"Final video generated: {final_video_path}")
            return final_video_path, None
        else:
            return None, f"ERROR: Video script ran but {final_video_path} was not found."
    
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR (Final Projection):\nstdout: {e.stdout}\nstderr: {e.stderr}"
        print(error_msg)
        return None, error_msg
    except FileNotFoundError as e:
        error_msg = f"ERROR: Script not found: {e.filename}"
        print(error_msg)
        return False, error_msg

def run_video_generation_script(prompt, src_video_path):
    """Runs the external video generation script using the trajectory video."""
    if not src_video_path or not os.path.exists(src_video_path):
        return None, "ERROR: Source video path is missing or invalid."

    output_path = os.path.join(TEMP_DIR, "out_video.mp4")
    
    # Construct command
    # Assuming the script takes --prompt, --src_video, and --output arguments
    cmd = [
        "python", "src/vace_lib/vace/vace_wan_inference.py",
        "--prompt", prompt,
        "--src_video", src_video_path,
        "--save_dir", TEMP_DIR,
        "--ckpt_dir", "src/vace_lib/models/Wan2.1-VACE-1.3B/"
    ]

    try:
        print(f"Running video generation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        if os.path.exists(output_path):
            return output_path, None
        else:
            return None, "ERROR: Video generation script ran but output file was not found."
            
    except subprocess.CalledProcessError as e:
        error_msg = f"ERROR (Video Gen Script):\nstdout: {e.stdout}\nstderr: {e.stderr}"
        print(error_msg)
        return None, error_msg
    except FileNotFoundError as e:
        return None, f"ERROR: Script not found: {e.filename}"

def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in
               ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg'])

def run_qwen_inference(
    message, generation_args=None, conversation_history=None, current_user_image=None
):
    """
    Runs Qwen model inference and streams text.
    Also handles image/video inputs.
    """
    streamed_text = ""
    user_uploaded_image_path = None

    qwen_images, qwen_videos = [], []
    if message and message.get("files"):
        for file_item in message["files"]:
            path = file_item["path"] if isinstance(file_item, dict) else file_item
            if is_video_file(path):
                qwen_videos.append(path)
            else:
                qwen_images.append(path)
                if user_uploaded_image_path is None:
                    user_uploaded_image_path = path
                    if current_user_image is not None:
                        current_user_image[0] = path

    current_user_content = []
    for image in qwen_images:
        current_user_content.append({"type": "image", "image": image})
    for vid in qwen_videos:
        current_user_content.append({"type": "video", "video": vid, "fps": 1.0})
    if message and message.get('text'):
        current_user_content.append({"type": "text", "text": message['text']})

    if not current_user_content:
        yield "", True, user_uploaded_image_path 
        return

    current_user_message = {"role": "user", "content": current_user_content}
    
    if conversation_history is None:
        conversation_history = []
    
    full_conversation = conversation_history + [current_user_message]
    
    prompt = processor.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(full_conversation)
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    assistant_response = ""
    for new_text in streamer:
        assistant_response += new_text
        streamed_text = assistant_response
        yield streamed_text, False, user_uploaded_image_path 
    
    if assistant_response.strip():
        conversation_history.append(current_user_message)
        conversation_history.append({"role": "assistant", "content": assistant_response.strip()})
    
    yield streamed_text, True, user_uploaded_image_path

# =============================================================================
# MAIN APP
# =============================================================================

def full_app():
    global processor, model, device
    
    conversation_history = []
    current_user_image = [None] 

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen Inference & Trajectory Generation")

        with gr.Row():
            with gr.Column(scale=2):
                live_output = gr.Textbox(label="Model / Script Output", lines=20)
                output_video = gr.Video(label="Trajectory Video (Bbox Projection)")
                
                # NEW: Video Generation Group (Hidden initially)
                with gr.Group(visible=False) as video_gen_group:
                    gr.Markdown("### 🎬 Video Generation Stage")
                    with gr.Row():
                        vid_prompt_input = gr.Textbox(
                            label="Video Prompt", 
                            placeholder="Describe the final video content...",
                            scale=3
                        )
                        run_vid_gen_btn = gr.Button("Generate Final Video", variant="primary", scale=1)
                    
                    final_generated_video = gr.Video(label="Final Generated Video")

            with gr.Column(scale=1):
                
                
                gr.Markdown(
                    """
                    ### 🛠️ Workflow Instructions

                    **1. Object Phase:** Describe the object's movement first, then click `📦 Generate Bbox`.

                    **2. Camera Phase:** Describe the camera trajectory, then click `🎥 Generate Camera`.

                    **3. Review:** A skeleton video and a new panel will appear. You can refine the camera text and regenerate, or use `🔄 Reset All` to restart.
                    
                    **4. Final Production:** If satisfied with the trajectory, enter a rich, descriptive prompt below and click `🎬 Generate Final Video` to generate the final video.
                    
                    *(⚠️ Note: Final generation involves model offloading and may take time depending on GPU availability.)*
                    """
                )
                
                chat_input = gr.MultimodalTextbox(
                    value="A man is running towards right.",
                    interactive=True, 
                    file_types=["image", "video"],
                    placeholder="A man is running towards right.", 
                    show_label=False, 
                    lines=10
                )

                reset_btn = gr.Button("🔄 Reset All", variant="secondary", size="sm")

                # =============================================================================
                # PROCESSING FUNCTION
                # =============================================================================
                
                def collect_all_inputs_base(message, generation_type):
                    
                    # Ensure temp directories exist
                    os.makedirs(TEMP_DIR, exist_ok=True)
                    os.makedirs(os.path.join(TEMP_DIR, "cam_json"), exist_ok=True)
                    
                    generation_args = {
                        "max_new_tokens": 1024,
                        "temperature": 0.2,
                        "do_sample": True,
                        "repetition_penalty": 1.0,
                    }
                    
                    final_qwen_output = ""
                    
                    # 1. RUN QWEN INFERENCE
                    if generation_type == "camera":
                        prompt_suffix = "  Generate Camera Tags: "
                    elif generation_type == "bbox":
                        prompt_suffix = ". Generate Object Trajectory: "
                    elif generation_type == "edit":
                        prompt_suffix = ". Generate Edit Tags: "
                        if conversation_history and conversation_history[-1]["role"] == "assistant":
                            print("added prefix\n\n")
                            last_response = conversation_history[-1]["content"]
                            # Eski cevap + Boşluk + Yeni talimat
                            message["text"] = last_response + " " + message["text"]
                    
                    message["text"] = message["text"] + prompt_suffix
                    
                    is_completed = False
                    
                    for text_chunk, completed, img_path in run_qwen_inference(
                        message, generation_args=generation_args, conversation_history=conversation_history, current_user_image=current_user_image
                    ):
                        final_qwen_output = text_chunk
                        is_completed = completed
                        # Yield text, no video yet, keep group hidden
                        yield final_qwen_output, None, gr.update(visible=False)

                    if not is_completed:
                        yield f"{final_qwen_output}\n\nERROR: Qwen inference did not complete.", None, gr.update(visible=False)
                        return

                    # 2. PROCESS QWEN OUTPUT (VIA SCRIPTS)
                    
                    if generation_type == "bbox":
                        # Generate and save Bbox
                        yield f"{final_qwen_output}\n\n**System:** Bbox text received. Running `object_to_json.py`...", None, gr.update(visible=False)
                        success, error_msg = process_bbox_text(final_qwen_output)
                        
                        if not success:
                            yield f"{final_qwen_output}\n\n{error_msg}", None, gr.update(visible=False)
                        else:
                            yield f"{final_qwen_output}\n\n**System:** `bbox.json` created successfully. You can now click 'Generate Camera'.", None, gr.update(visible=False)
                    
                    elif generation_type == "camera" or generation_type == "edit":
                        # Generate Camera, Save, and GENERATE TRAJECTORY VIDEO
                        yield f"{final_qwen_output}\n\n**System:** Camera text received. Running camera processing chain...", None, gr.update(visible=False)
                        
                        success, error_msg = process_camera_text(final_qwen_output)
                        
                        if not success:
                            yield f"{final_qwen_output}\n\n{error_msg}", None, gr.update(visible=False)
                            return

                        yield f"{final_qwen_output}\n\n**System:** Camera processing complete. Running `projection.py`...", None, gr.update(visible=False)

                        # Run projection script
                        video_path, error_msg = run_final_projection()
                        
                        if error_msg:
                            yield f"{final_qwen_output}\n\n{error_msg}", None, gr.update(visible=False)
                        else:
                            # SUCCESS: Show video AND make the new group visible
                            yield f"{final_qwen_output}\n\n**System:** Trajectory video generated successfully!\n{video_path}", video_path, gr.update(visible=True)
                

                def reset_all():
                    conversation_history.clear()
                    current_user_image[0] = None
                    
                    try:
                        if os.path.exists(TEMP_DIR):
                            files_to_delete = [
                                "cam.txt", "cam_tag.txt", "cam_traj.txt", "cam_traj_matrix.txt",
                                "bbox.txt", "bbox.json", "final_generated_video.mp4"
                            ]
                            for f in files_to_delete:
                                f_path = os.path.join(TEMP_DIR, f)
                                if os.path.exists(f_path):
                                    os.remove(f_path)
                            
                            cam_json_dir = os.path.join(TEMP_DIR, "cam_json")
                            if os.path.exists(cam_json_dir):
                                shutil.rmtree(cam_json_dir)
                            print("Temp files cleaned.")
                    except Exception as e:
                        print(f"Error cleaning temp files: {e}")

                    return (
                        gr.update(value="Camera zooms out and moves upward."),  # chat_input
                        "",      # live_output
                        None,    # output_video
                        gr.update(visible=False), # video_gen_group
                        None     # final_generated_video
                    )
                
                # Handler for the new Video Generation Button
                def on_generate_final_video(prompt, src_video):
                    if not src_video:
                        return None, "Please generate the trajectory video first."
                    
                    try:
                        model.to("cpu")
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        print(f"Error freeing GPU memory: {e}")
                        
                    
                    
                    video_path, error = run_video_generation_script(prompt, src_video)
                    
                    try:
                        model.to(device)
                    except Exception as e:
                        print(f"Error moving model to GPU: {e}")

                    if error:
                        return None # Or handle error display
                    else:
                        return video_path

                # Two separate generation buttons
                with gr.Row():
                    generate_camera_btn = gr.Button("🎥 Generate Camera", variant="primary", scale=1)
                    generate_bbox_btn = gr.Button("📦 Generate Bbox", variant="secondary", scale=1)
                    generate_edit_btn = gr.Button("🎥 Edit Camera", variant="tertiarty", scale=1)

                
                # The outputs for main generation now includes the group visibility
                outputs_list = [live_output, output_video, video_gen_group]
                
                # Camera generation button
                def camera_click(message):
                    yield from collect_all_inputs_base(message, "camera")
                
                generate_camera_btn.click(
                    fn=camera_click,
                    inputs=[chat_input],
                    outputs=outputs_list,
                    show_progress=True
                )

                # Bbox generation button
                def bbox_click(message):
                    yield from collect_all_inputs_base(message, "bbox")
                
                generate_bbox_btn.click(
                    fn=bbox_click,
                    inputs=[chat_input],
                    outputs=outputs_list,
                    show_progress=True
                )

                def edit_click(message):
                    yield from collect_all_inputs_base(message, "edit")

                generate_edit_btn.click(
                    fn=edit_click,
                    inputs=[chat_input],
                    outputs=outputs_list,
                    show_progress=True
                )
                # Chat input submit (Default to Camera)
                def chat_submit(message):
                    yield from collect_all_inputs_base(message, "camera")
                
                chat_input.submit(
                    fn=chat_submit,
                    inputs=[chat_input],
                    outputs=outputs_list,
                    show_progress=True
                )
                
                # Connect the new Video Gen button
                run_vid_gen_btn.click(
                    fn=on_generate_final_video,
                    inputs=[vid_prompt_input, output_video], # output_video acts as source input here
                    outputs=[final_generated_video],
                    show_progress=True
                )
                
                reset_btn.click(
                    fn=reset_all,
                    outputs=[chat_input, live_output, output_video, video_gen_group, final_generated_video],
                    show_progress=False
                )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8890, share=False)


def main(args):
    global processor, model, device
    device = args.device
    disable_torch_init()

    # Load Model
    processor, model = load_pretrained_model(
        model_base=args.model_base, model_path=args.model_path,
        device_map=args.device, model_name=get_model_name_from_path(args.model_path),
        load_4bit=args.load_4bit, load_8bit=args.load_8bit,
        device=args.device, use_flash_attn=not args.disable_flash_attention
    )

    full_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)
