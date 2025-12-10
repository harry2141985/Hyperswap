import os
from comfy.model_management import CPUState  # Импорт из того же файла

# Отключаем CUDA, чтобы избежать инициализации
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# Принудительно устанавливаем CPU режим
import comfy.model_management
comfy.model_management.cpu_state = CPUState.CPU

# import torch only AFTER disabling CUDA / forcing CPU
import torch

# add a sane default cap to avoid huge allocations
MAX_FRAME_LOAD_CAP = 300

import random
import sys
from typing import Sequence, Mapping, Any, Union
from PIL import Image
from huggingface_hub import hf_hub_download
import spaces

import subprocess, sys

import gradio
import gradio_client
import gradio as gr

print("gradio version:", gradio.__version__)
print("gradio_client version:", gradio_client.__version__)

hf_hub_download(repo_id="facefusion/models-3.3.0", filename="hyperswap_1a_256.onnx", local_dir="models/hyperswap")
hf_hub_download(repo_id="facefusion/models-3.3.0", filename="hyperswap_1b_256.onnx", local_dir="models/hyperswap")
hf_hub_download(repo_id="facefusion/models-3.3.0", filename="hyperswap_1c_256.onnx", local_dir="models/hyperswap")

hf_hub_download(repo_id="martintomov/comfy", filename="facedetection/yolov5l-face.pth", local_dir="models")
###hf_hub_download(repo_id="darkeril/collection", filename="detection_Resnet50_Final.pth", local_dir="models/facedetection")
hf_hub_download(repo_id="gmk123/GFPGAN", filename="parsing_parsenet.pth", local_dir="models/facedetection")

hf_hub_download(repo_id="MonsterMMORPG/tools", filename="1k3d68.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="MonsterMMORPG/tools", filename="2d106det.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="maze/faceX", filename="det_10g.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="typhoon01/aux_models", filename="genderage.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="maze/faceX", filename="w600k_r50.onnx", local_dir="models/insightface/models/buffalo_l")

hf_hub_download(repo_id="vladmandic/insightface-faceanalysis", filename="buffalo_l.zip", local_dir="models/insightface/models/buffalo_l")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    # Запускаем корутину и ждём её завершения
    loop.run_until_complete(init_extra_nodes())

import_custom_nodes()
from nodes import NODE_CLASS_MAPPINGS

# --- Глобальная загрузка моделей (один раз при старте) ---
loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
reactoroptions = NODE_CLASS_MAPPINGS["ReActorOptions"]()
vhs_videoinfoloaded = NODE_CLASS_MAPPINGS["VHS_VideoInfoLoaded"]()
reactorfaceswapopt = NODE_CLASS_MAPPINGS["ReActorFaceSwapOpt"]()
vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

# @spaces.GPU(duration=60)
def generate_image(source_image, input_video, input_index, input_faces_order, swap_model, pingpong, loop_count, select_every_nth, use_audio):
    with torch.inference_mode():
        loadimage_29 = loadimage.load_image(image=source_image)

        vhs_loadvideo_51 = vhs_loadvideo.load_video(
            video=input_video,
            force_rate=0,
            custom_width=640,        # downscale to reduce memory (tweak as needed)
            custom_height=360,
            frame_load_cap=MAX_FRAME_LOAD_CAP,  # limit frames loaded into memory
            skip_first_frames=0,
            select_every_nth=select_every_nth,
            format="AnimateDiff",
            unique_id=17765013700631265033,
        )

        reactoroptions_107 = reactoroptions.execute(
            input_faces_order=input_faces_order,
            input_faces_index=str(input_index),  # Преобразуем в строку
            detect_gender_input="no",
            source_faces_order="large-small",
            source_faces_index="0",
            detect_gender_source="no",
            console_log_level=1,
        )

        for q in range(1):
            vhs_videoinfoloaded_105 = vhs_videoinfoloaded.get_video_info(
                video_info=get_value_at_index(vhs_loadvideo_51, 3)
            )

            reactorfaceswapopt_106 = reactorfaceswapopt.execute(
                enabled=True,
                swap_model=swap_model, # Используем выбранную модель
                facedetection="YOLOv5l",
                face_restore_model="none",
                face_restore_visibility=1,
                codeformer_weight=0.5,
                input_image=get_value_at_index(vhs_loadvideo_51, 0),
                source_image=get_value_at_index(loadimage_29, 0),
                options=get_value_at_index(reactoroptions_107, 0),
            )

            # Формируем аргументы для combine_video
            combine_kwargs = dict(
                frame_rate=get_value_at_index(vhs_videoinfoloaded_105, 0),
                loop_count=loop_count,
                filename_prefix="vidswap",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=20,
                save_metadata=False,
                trim_to_audio=False,
                pingpong=pingpong,
                save_output=True,
                images=get_value_at_index(reactorfaceswapopt_106, 0),
                unique_id=17889577966051683261,
            )
            
            if use_audio:
                combine_kwargs["audio"] = get_value_at_index(vhs_loadvideo_51, 2)

            vhs_videocombine_28 = vhs_videocombine.combine_video(**combine_kwargs)

            saved_path = f"output/{vhs_videocombine_28['ui']['gifs'][0]['filename']}"
            return saved_path

def generate_image_faceswap(source_image, target_image, input_index, input_faces_order, swap_model):
    """Face swap between two images"""
    with torch.inference_mode():
        loadimage_source = loadimage.load_image(image=source_image)
        loadimage_target = loadimage.load_image(image=target_image)

        reactoroptions_img = reactoroptions.execute(
            input_faces_order=input_faces_order,
            input_faces_index=str(input_index),
            detect_gender_input="no",
            source_faces_order="large-small",
            source_faces_index="0",
            detect_gender_source="no",
            console_log_level=1,
        )

        reactorfaceswapopt_img = reactorfaceswapopt.execute(
            enabled=True,
            swap_model=swap_model,
            facedetection="YOLOv5l",
            face_restore_model="none",
            face_restore_visibility=1,
            codeformer_weight=0.5,
            input_image=get_value_at_index(loadimage_target, 0),
            source_image=get_value_at_index(loadimage_source, 0),
            options=get_value_at_index(reactoroptions_img, 0),
        )

        result_image = get_value_at_index(reactorfaceswapopt_img, 0)
        
        # Save result image
        os.makedirs("output", exist_ok=True)
        output_path = "output/imgswap_result.png"
        result_pil = Image.fromarray((result_image[0].cpu().numpy() * 255).astype("uint8"))
        result_pil.save(output_path)
        
        return output_path

if __name__ == "__main__":
    with gr.Blocks() as app:
        with gr.Tabs():
            # Video tab removed (video-based face swap UI is no longer present)
            
            with gr.Tab("Image Face Swap"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            source_image_img = gr.Image(label="Source Image (Face)", type="filepath")
                            target_image_img = gr.Image(label="Target Image (Body)", type="filepath")
                            swap_model_img = gr.Dropdown(
                                choices=["hyperswap_1a_256.onnx", "hyperswap_1b_256.onnx", "hyperswap_1c_256.onnx"],
                                value="hyperswap_1b_256.onnx",
                                label="Swap Model"
                            )
                            input_index_img = gr.Dropdown(choices=[0, 1, 2, 3, 4], value=0, label="Target Face Index")
                            input_faces_order_img = gr.Dropdown(
                                choices=[
                                    "left-right",
                                    "right-left",
                                    "top-bottom",
                                    "bottom-top",
                                    "large-small"
                                ],
                                value="large-small",
                                label="Target Faces Order"
                            )
                            
                            generate_img_btn = gr.Button("Generate Face Swap Image!")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Generated Image")

            generate_img_btn.click(
                fn=generate_image_faceswap,
                inputs=[source_image_img, target_image_img, input_index_img, input_faces_order_img, swap_model_img],
                outputs=[output_image]
            )

    app.launch(share=True)
