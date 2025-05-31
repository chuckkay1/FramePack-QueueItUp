import inspect
import io
import os
import sys
import re
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
from diffusers_helper.hf_login import login
import json
import traceback
from dataclasses import dataclass, asdict
from typing import Optional
import uuid
import configparser
import ast
import random
import gradio as gr # type: ignore
import torch # type: ignore
import traceback
import einops # type: ignore
import safetensors.torch as sf # type: ignore
from safetensors.torch import load_file # type: ignore
import numpy as np # type: ignore
import argparse
import math
from PIL import Image, ImageDraw, ImageFont # type: ignore
from PIL.PngImagePlugin import PngInfo # type: ignore
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers.utils import load_image # type: ignore
from diffusers import AutoencoderKLHunyuanVideo # type: ignore
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer # type: ignore
from transformers import SiglipImageProcessor, SiglipVisionModel # type: ignore
import shutil
import cv2 # type: ignore
import subprocess
import datetime
import math
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi, snapshot_download # type: ignore
import gc
import time


# ANSI color codes
YELLOW = '\033[93m'
RED = '\033[31m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'
try:
# Try to load arial font, fall back to default if not available
    font = ImageFont.truetype("arial.ttf", 16)
    small_font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


def debug_print(message):
    """Print debug messages in yellow color"""
    if getattr(Config, "DEBUG_MODE", True):
        print(f"{YELLOW}[DEBUG] {message}{RESET}")
    
def alert_print(message):
    """Print alert messages in red color"""
    print(f"{RED}[ALERT] {message}{RESET}")

def info_print(message):
    """Print info messages in green color"""
    print(f"{GREEN}[INFO] {message}{RESET}")

def success_print(message):
    """Print info messages in green color"""
    print(f"{BLUE}[INFO] {message}{RESET}")


# Create necessary folders and files at startup
def ensure_files_folders():
    # Define required folders
    folders = {
        'hf_download': os.path.join(os.path.dirname(__file__), 'hf_download'),
        'loras': os.path.join(os.path.dirname(__file__), 'loras'), 
        'outputs': os.path.join(os.path.dirname(__file__), 'outputs'),
        'job_history': os.path.join(os.path.dirname(__file__), 'job_history'),
        'temp_queue_images': os.path.join(os.path.dirname(__file__), 'temp_queue_images')
    }
    
    # Create folders if they don't exist
    for name, path in folders.items():
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created {name} folder: {path}")
        except Exception as e:
            print(f"Error creating {name} folder: {str(e)}")

    # Check and create required JSON files
    json_files = {
        'job_queue.json': os.path.join(os.path.dirname(__file__), 'job_queue.json'),
        'quick_prompts.json': os.path.join(os.path.dirname(__file__), 'quick_prompts.json')
    }

    for name, path in json_files.items():
        try:
            if not os.path.exists(path):
                # Create empty JSON file with empty list/dict
                with open(path, 'w') as f:
                    if name == 'job_queue.json':
                        json.dump([], f)
                    else:
                        json.dump({}, f)
                print(f"Created {name} file")
        except Exception as e:
            print(f"Error creating {name} file: {str(e)}")

# Run folder and file creation at startup
ensure_files_folders()

current_counts = {
    "completed": 0,
    "pending": 0,
    "failed": 0,
    "all": 0
}


# Check for ffmpeg and install if needed
def ensure_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("FFmpeg is already installed and available")
            return True
    except FileNotFoundError:
        print("FFmpeg not found, attempting to install imageio_ffmpeg...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio_ffmpeg"])
            print("Successfully installed imageio_ffmpeg")
            import imageio_ffmpeg # type: ignore
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_path and os.path.exists(ffmpeg_path):
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ['PATH']
                print(f"Added FFmpeg to PATH: {ffmpeg_dir}")
                return True
        except Exception as e:
            print(f"Error installing imageio_ffmpeg: {str(e)}")
            return False
    return True

# Run the check at startup
ensure_ffmpeg()

def ensure_peft():
    try:
        # try to import PEFT
        from peft import PeftModel, PeftConfig # type: ignore
        return PeftModel, PeftConfig
    except ImportError:
        # install it into this Python environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
        # now retry the import
        from peft import PeftModel, PeftConfig # type: ignore
        return PeftModel, PeftConfig

PeftModel, PeftConfig = ensure_peft()

# After imports, before other functions
def is_model_downloaded(model_value):
    """Check if a model is downloaded by checking for the LOCAL- prefix"""
    if isinstance(model_value, str):
        return model_value.startswith('LOCAL-')
    return False

def set_model_as_default(model_type, model_value):
    """Set a specific model as the default in settings.ini"""
    try:
        # First check if model is downloaded (has LOCAL- prefix)
        if not model_value.startswith('LOCAL-'):
            return f"Cannot set {model_type} as default - model must be downloaded first", None, None, None, None, None, None, None, None
            
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Remove the display prefix to get actual folder name
        actual_model = Config.model_name_mapping.get(model_value, model_value.replace('LOCAL-', '').replace(' - CURRENT DEFAULT MODEL', ''))
        
        # Update the config
        config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(actual_model)
        
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Update Config object
        setattr(Config, f'DEFAULT_{model_type.upper()}', actual_model)
        
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        # Return status message and individual model lists
        return (
            f"{model_type} set as default successfully. Restart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder'],
            models['image2image_model']
        )
    except Exception as e:
        alert_print(f"Error setting {model_type} as default: {str(e)}")
        return f"Error setting {model_type} as default: {str(e)}", None, None, None, None, None, None, None, None

def set_all_models_as_default(transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, image2image_model):
    """Set all models as default at once"""
    try:
        # Check if all models are downloaded
        models = [transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, image2image_model]
        model_types = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model']
        
        for model, model_type in zip(models, model_types):
            if not model.startswith('LOCAL-'):
                return f"Cannot set {model_type} as default - model must be downloaded first", None, None, None, None, None, None, None, None
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update all models
        success_messages = []
        for model, model_type in zip(models, model_types):
            actual_model = Config.model_name_mapping.get(model, model.replace('LOCAL-', '').replace(' - CURRENT DEFAULT MODEL', ''))
            config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(actual_model)
            setattr(Config, f'DEFAULT_{model_type.upper()}', actual_model)
            success_messages.append(f"{model_type}: {actual_model}")
            
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        return (
            "All models set as default successfully:\n" + "\n".join(success_messages) + "\n\nRestart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder'],
            models['image2image_model']
        )
    except Exception as e:
        alert_print(f"Error setting models as default: {str(e)}")
        return f"Error setting models as default: {str(e)}", None, None, None, None, None, None, None, None

def restore_model_default(model_type):
    """Restore a specific model to its original default in settings.ini"""
    try:
        # Get original defaults
        original_defaults = Config.get_original_defaults()
        original_value = original_defaults[f'DEFAULT_{model_type.upper()}']
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update the config
        config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(original_value)
        
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Update Config object
        setattr(Config, f'DEFAULT_{model_type.upper()}', original_value)
        
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        # Return status message and individual model lists
        return (
            f"{model_type} restored to original default successfully. Restart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder'],
            models['image2image_model']
        )
    except Exception as e:
        alert_print(f"Error restoring {model_type} default: {str(e)}")
        return f"Error restoring {model_type} default: {str(e)}", None, None, None, None, None, None, None, None


def download_model_from_huggingface(model_id):
    """Download a model from Hugging Face and return its local path"""
    try:
        # Handle our display name format
        if model_id.startswith('DOWNLOADED-MODEL-'):
            if hasattr(Config, 'model_name_mapping') and model_id in Config.model_name_mapping:
                model_id = Config.model_name_mapping[model_id]
        
        # Convert from org/model to models--org--model format
        if '/' in model_id:
            org, model = model_id.split('/')
            local_name = f"models--{org}--{model}"
        else:
            local_name = model_id
            
        hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download", "hub")
        model_path = os.path.join(hub_dir, local_name)
        
        # Get token from Config and set in environment
        token = Config._instance.HF_TOKEN if Config._instance else Config.HF_TOKEN
        if not token or token == "add token here":
            alert_print("No valid Hugging Face token found. Please add your token in the settings.")
            return None
            
        # Set token in environment
        os.environ['HF_TOKEN'] = token
        
        if not os.path.exists(model_path):
            debug_print(f"Downloading model {model_id}...")
            try:
                # First check if we can access the model
                api = HfApi()
                try:
                    # Try to get model info first
                    model_info = api.model_info(model_id, token=token)
                    debug_print(f"Model info: {model_info.modelId} - {model_info.tags if hasattr(model_info, 'tags') else 'No tags'}")
                    if hasattr(model_info, 'private') and model_info.private:
                        alert_print(f"Model {model_id} is private and cannot be accessed")
                        return None
                except Exception as e:
                    alert_print(f"Cannot access model {model_id}: {str(e)}")
                    return None
                
                # Use the appropriate model class based on the model type
                if "hunyuanvideo" in model_id.lower():
                    try:
                        AutoencoderKLHunyuanVideo.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        alert_print(f"Error downloading hunyuanvideo model: {str(e)}")
                        if os.path.exists(model_path):
                            shutil.rmtree(model_path, ignore_errors=True)
                        return None
                elif "flux_redux" in model_id.lower():
                    try:
                        SiglipImageProcessor.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                        SiglipVisionModel.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        alert_print(f"Error downloading flux_redux model: {str(e)}")
                        if os.path.exists(model_path):
                            shutil.rmtree(model_path, ignore_errors=True)
                        return None
                elif "framepack" in model_id.lower():
                    try:
                        HunyuanVideoTransformer3DModelPacked.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        alert_print(f"Error downloading framepack model: {str(e)}")
                        if os.path.exists(model_path):
                            shutil.rmtree(model_path, ignore_errors=True)
                        return None
                debug_print(f"Model downloaded to {model_path}")
            except Exception as e:
                alert_print(f"Error during download process: {str(e)}")
                if os.path.exists(model_path):
                    shutil.rmtree(model_path, ignore_errors=True)
                return None
                
        # Return display name if we have it
        display_name = next((k for k, v in Config.model_name_mapping.items() if v == local_name), None) if hasattr(Config, 'model_name_mapping') else None
        return display_name if display_name else local_name
        
    except Exception as e:
        alert_print(f"Error in download process: {str(e)}")
        traceback.print_exc()
        return None




def load_lora(
    state_dict: dict[str, torch.Tensor], lora_model: str, lora_weight: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.
    """
    lora_sd = load_file(lora_model)

    # Check the format of the LoRA file
    keys = list(lora_sd.keys())
    if keys[0].startswith("lora_unet_"):
        info_print(f"Musubi Tuner LoRA detected")
        return merge_musubi_tuner(lora_sd, state_dict, lora_weight, device)

    transformer_prefixes = ["diffusion_model", "transformer"]  # to ignore Text Encoder modules
    lora_suffix = None
    prefix = None
    for key in keys:
        if lora_suffix is None and "lora_A" in key:
            lora_suffix = "lora_A"
        if prefix is None:
            pfx = key.split(".")[0]
            if pfx in transformer_prefixes:
                prefix = pfx
        if lora_suffix is not None and prefix is not None:
            break

    if lora_suffix == "lora_A" and prefix is not None:
        success_print(f"Diffusion-pipe (?) LoRA detected")
        return merge_diffusion_pipe_or_something(lora_sd, state_dict, "lora_unet_", lora_weight, device)

    alert_print(f"LoRA file format not recognized: {os.path.basename(lora_model)}")
    return state_dict


def merge_diffusion_pipe_or_something(
    lora_sd: dict[str, torch.Tensor], state_dict: dict[str, torch.Tensor], prefix: str, lora_weight: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to the format used by the diffusion pipeline to Musubi Tuner.
    Copy from Musubi Tuner repo.
    """
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return merge_musubi_tuner(new_weights_sd, state_dict, lora_weight, device)


def merge_musubi_tuner(
    lora_sd: dict[str, torch.Tensor], state_dict: dict[str, torch.Tensor], lora_weight: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.
    """
    # Check LoRA is for FramePack or for HunyuanVideo
    is_hunyuan = False
    for key in lora_sd.keys():
        if "double_blocks" in key or "single_blocks" in key:
            is_hunyuan = True
            break
    if is_hunyuan:
        success_print("HunyuanVideo LoRA detected, converting to FramePack format")
        lora_sd = convert_hunyuan_to_framepack(lora_sd)

    # Merge LoRA weights into the state dict
    success_print(f"Merging LoRA weights into state dict. lora_weight: {lora_weight}")

    # Create module map
    name_to_original_key = {}
    for key in state_dict.keys():
        if key.endswith(".weight"):
            lora_name = key.rsplit(".", 1)[0]  # remove trailing ".weight"
            lora_name = "lora_unet_" + lora_name.replace(".", "_")
            if lora_name not in name_to_original_key:
                name_to_original_key[lora_name] = key

    # Merge LoRA weights
    keys = list([k for k in lora_sd.keys() if "lora_down" in k])
    for key in tqdm(keys, desc="Merging LoRA weights"):
        up_key = key.replace("lora_down", "lora_up")
        alpha_key = key[: key.index("lora_down")] + "alpha"

        # find original key for this lora
        module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
        if module_name not in name_to_original_key:
            debug_print(f"No module found for LoRA weight: {key}")
            continue

        original_key = name_to_original_key[module_name]

        down_weight = lora_sd[key]
        up_weight = lora_sd[up_key]

        dim = down_weight.size()[0]
        alpha = lora_sd.get(alpha_key, dim)
        scale = alpha / dim

        weight = state_dict[original_key]
        original_device = weight.device
        if original_device != device:
            weight = weight.to(device)  # to make calculation faster

        down_weight = down_weight.to(device)
        up_weight = up_weight.to(device)

        # W <- W + U * D
        if len(weight.size()) == 2:
            # linear
            if len(up_weight.size()) == 4:  # use linear projection mismatch
                up_weight = up_weight.squeeze(3).squeeze(2)
                down_weight = down_weight.squeeze(3).squeeze(2)
            weight = weight + lora_weight * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + lora_weight
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # logger.info(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + lora_weight * conved * scale

        weight = weight.to(original_device)  # move back to original device
        state_dict[original_key] = weight

    return state_dict


def convert_hunyuan_to_framepack(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert HunyuanVideo LoRA weights to FramePack format.
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        if "double_blocks" in key:
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
        elif "single_blocks" in key:
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
        else:
            print(f"Unsupported module name: {key}, only double_blocks and single_blocks are supported")
            continue

        if "QKVM" in key:
            # split QKVM into Q, K, V, M
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # copy QKVM weight or alpha to Q, K, V, M
                assert "alpha" in key or weight.size(1) == 3072, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # split QKVM weight into Q, K, V, M
                assert weight.size(0) == 21504, f"QKVM weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(f"Unsupported module name: {key}")
                continue
        elif "QKV" in key:
            # split QKV into Q, K, V
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # copy QKV weight or alpha to Q, K, V
                assert "alpha" in key or weight.size(1) == 3072, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # split QKV weight into Q, K, V
                assert weight.size(0) == 3072 * 3, f"QKV weight size mismatch: {key}. {weight.size()}"
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(f"Unsupported module name: {key}")
                continue
        else:
            # no split needed
            new_lora_sd[key] = weight

    return new_lora_sd



def get_available_models(include_online=False):
    """Get list of available models from hub directory and optionally from Hugging Face"""
    debug_print(f"Starting get_available_models with include_online={include_online}")
    hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download", "hub")
    debug_print(f"Hub directory: {hub_dir}")
    
    # Dictionary to store the mapping between display names and actual folder names
    name_mapping = {}
    
    models = {
        'text_encoder': [],
        'text_encoder_2': [],
        'tokenizer': [],
        'tokenizer_2': [],
        'vae': [],
        'feature_extractor': [],
        'image_encoder': [],
        'transformer': [],
        'image2image_model': []
    }
    
    # Get current defaults from Config
    user_defaults = {
        'transformer': Config.DEFAULT_TRANSFORMER,
        'text_encoder': Config.DEFAULT_TEXT_ENCODER,
        'text_encoder_2': Config.DEFAULT_TEXT_ENCODER_2,
        'tokenizer': Config.DEFAULT_TOKENIZER,
        'tokenizer_2': Config.DEFAULT_TOKENIZER_2,
        'vae': Config.DEFAULT_VAE,
        'feature_extractor': Config.DEFAULT_FEATURE_EXTRACTOR,
        'image_encoder': Config.DEFAULT_IMAGE_ENCODER,
        'image2image_model': Config.DEFAULT_IMAGE2IMAGE_MODEL
    }
    
    # Get original defaults
    original_defaults = Config.get_original_defaults()
    
    # Get local models
    if os.path.exists(hub_dir):
        debug_print("Scanning local models...")
        for item in os.listdir(hub_dir):
            if os.path.isdir(os.path.join(hub_dir, item)):
                debug_print(f"Found local model directory: {item}")
                # Create base display name with prefix
                display_name = f"LOCAL-{item}"
                # Store mapping
                name_mapping[display_name] = item
                
                # Map models to their correct categories using display name
                if "hunyuanvideo" in item.lower():
                    # Function to handle model categorization
                    def add_model_with_suffix(model_type):
                        is_user_default = item == user_defaults[model_type]
                        is_original_default = item == original_defaults[f'DEFAULT_{model_type.upper()}']
                        
                        if is_original_default and not is_user_default:
                            display_name_suffix = f"{display_name} - ORIGINAL DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        elif is_user_default:
                            display_name_suffix = f"{display_name} - CURRENT DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        else:
                            models[model_type].append(display_name)
                    
                    add_model_with_suffix('text_encoder')
                    add_model_with_suffix('text_encoder_2')
                    add_model_with_suffix('tokenizer')
                    add_model_with_suffix('tokenizer_2')    
                    add_model_with_suffix('vae')
                    add_model_with_suffix('image2image')
                elif "flux_redux" in item.lower():
                    def add_model_with_suffix(model_type):
                        is_user_default = item == user_defaults[model_type]
                        is_original_default = item == original_defaults[f'DEFAULT_{model_type.upper()}']
                        
                        if is_original_default and not is_user_default:
                            display_name_suffix = f"{display_name} - ORIGINAL DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        elif is_user_default:
                            display_name_suffix = f"{display_name} - CURRENT DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        else:
                            models[model_type].append(display_name)
                    
                    add_model_with_suffix('feature_extractor')
                    add_model_with_suffix('image_encoder')
                        
                elif "framepack" in item.lower():
                    is_user_default = item == user_defaults['transformer']
                    is_original_default = item == original_defaults['DEFAULT_TRANSFORMER']
                    
                    if is_original_default and not is_user_default:
                        display_name_suffix = f"{display_name} - ORIGINAL DEFAULT MODEL"
                        name_mapping[display_name_suffix] = item
                        models['transformer'].append(display_name_suffix)
                    elif is_user_default:
                        display_name_suffix = f"{display_name} - CURRENT DEFAULT MODEL"
                        name_mapping[display_name_suffix] = item
                        models['transformer'].append(display_name_suffix)
                    else:
                        models['transformer'].append(display_name)
    
    debug_print("Local models found:")
    for key, value in models.items():
        debug_print(f"  {key}: {value}")
    
    # Add online models if requested
    if include_online:
        debug_print("Online models requested, starting online search...")
        try:
            
            # Get token from Config and set in environment
            token = Config._instance.HF_TOKEN if Config._instance else Config.HF_TOKEN
            debug_print(f"Token status: {'Valid token found' if token and token != 'add token here' else 'No valid token'}")
            
            if token and token != "add token here":
                try:
                    # Set token in environment for the helper module
                    os.environ['HF_TOKEN'] = token
                    
                    debug_print("Attempting Hugging Face login...")
                    # Use the imported login function from diffusers_helper
                    login(token)
                    debug_print("Login successful, searching for models...")
                    
                    api = HfApi()
                    
                    # Convert generators to lists before processing
                    debug_print("Searching for hunyuanvideo models...")
                    hunyuan_models = list(api.list_models(search="hunyuanvideo", token=token))
                    debug_print(f"Found {len(hunyuan_models)} hunyuan models")
                    
                    debug_print("Searching for flux_redux models...")
                    flux_models = list(api.list_models(search="flux_redux", token=token))
                    debug_print(f"Found {len(flux_models)} flux models")
                    
                    debug_print("Searching for framepack models...")
                    framepack_models = list(api.list_models(search="framepack", token=token))
                    debug_print(f"Found {len(framepack_models)} framepack models")
                    
                    # Add online models to the lists
                    for model in hunyuan_models:
                        model_id = model.id
                        debug_print(f"Processing hunyuan model: {model_id}")
                        
                        # Check if this is a default model
                        is_user_default = (model_id == convert_model_path(user_defaults['text_encoder']) or
                                         model_id == convert_model_path(user_defaults['text_encoder_2']) or
                                         model_id == convert_model_path(user_defaults['tokenizer']) or
                                         model_id == convert_model_path(user_defaults['tokenizer_2']) or
                                         model_id == convert_model_path(user_defaults['vae']) or
                                         model_id == convert_model_path(user_defaults['image2image_model']))
                                         
                        is_original_default = (model_id == convert_model_path(original_defaults['DEFAULT_TEXT_ENCODER']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_TEXT_ENCODER_2']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_TOKENIZER']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_TOKENIZER_2']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_VAE']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_IMAGE2IMAGE_MODEL']))
                        
                        display_id = model_id
                        if is_original_default and not is_user_default:
                            display_id = f"{model_id} - ORIGINAL DEFAULT MODEL"
                        elif is_user_default:
                            display_id = f"{model_id} - CURRENT DEFAULT MODEL"
                            
                        if display_id not in models['text_encoder']:
                            models['text_encoder'].append(display_id)
                            models['text_encoder_2'].append(display_id)
                            models['tokenizer'].append(display_id)
                            models['tokenizer_2'].append(display_id)
                            models['vae'].append(display_id)
                            models['image2image_model'].append(display_id)
                    for model in flux_models:
                        model_id = model.id
                        debug_print(f"Processing flux model: {model_id}")
                        
                        # Check if this is a default model
                        is_user_default = (model_id == convert_model_path(user_defaults['feature_extractor']) or
                                         model_id == convert_model_path(user_defaults['image_encoder']))
                                         
                        is_original_default = (model_id == convert_model_path(original_defaults['DEFAULT_FEATURE_EXTRACTOR']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_IMAGE_ENCODER']))
                        
                        display_id = model_id
                        if is_original_default and not is_user_default:
                            display_id = f"{model_id} - ORIGINAL DEFAULT MODEL"
                        elif is_user_default:
                            display_id = f"{model_id} - CURRENT DEFAULT MODEL"
                            
                        if display_id not in models['feature_extractor']:
                            models['feature_extractor'].append(display_id)
                            models['image_encoder'].append(display_id)
                            
                    for model in framepack_models:
                        model_id = model.id
                        debug_print(f"Processing framepack model: {model_id}")
                        
                        # Check if this is a default model
                        is_user_default = model_id == convert_model_path(user_defaults['transformer'])
                        is_original_default = model_id == convert_model_path(original_defaults['DEFAULT_TRANSFORMER'])
                        
                        display_id = model_id
                        if is_original_default and not is_user_default:
                            display_id = f"{model_id} - ORIGINAL DEFAULT MODEL"
                        elif is_user_default:
                            display_id = f"{model_id} - CURRENT DEFAULT MODEL"
                            
                        if display_id not in models['transformer']:
                            models['transformer'].append(display_id)
                            
                except Exception as e:
                    alert_print(f"Error logging in to Hugging Face: {str(e)}")
                    debug_print(f"Login error details: {traceback.format_exc()}")
            else:
                alert_print("No valid Hugging Face token found. Please add your token in the settings.")
        except Exception as e:
            alert_print(f"Error fetching online models: {str(e)}")
            debug_print(f"Online fetch error details: {traceback.format_exc()}")
    
    debug_print("Returning final model lists")
    # Store the name mapping in a global variable or Config
    Config.model_name_mapping = name_mapping
    return models

# Path to settings file
INI_FILE = os.path.join(os.getcwd(), 'settings.ini')

# Path to the quick prompts JSON file
QUICK_LIST_FILE = os.path.join(os.getcwd(), 'quick_prompts.json')

# Initialize quick_prompts as a list with a default prompt
default_quick_prompt = {
    'prompt': "The girl dances gracefully, with clear movements, full of charm.",
    'n_prompt': "",
    'lora_model': "None",
    'lora_weight': 1.0,
    'video_length': 5.0,
    'job_name': "Job-",
    'use_teacache': True,
    'seed': -1,
    'steps': 25,
    'cfg': 1.0,
    'gs': 10.0,
    'rs': 0.0,
    'image_strength': 1.0,
    'mp4_crf': 16,
    'gpu_memory': 6.0,
    'keep_temp_png': True
}

# Initialize quick_prompts as a list
quick_prompts = []

# Load quick prompts from file if it exists
try:
    if os.path.exists(QUICK_LIST_FILE):
        with open(QUICK_LIST_FILE, 'r') as f:
            quick_prompts = json.load(f)
            if not isinstance(quick_prompts, list):
                quick_prompts = []
            # If the list is empty, add the default prompt
            if not quick_prompts:
                quick_prompts = [default_quick_prompt]
                with open(QUICK_LIST_FILE, 'w') as f:
                    json.dump(quick_prompts, f, indent=2)
    else:
        # Create quick_prompts.json with default prompt if it doesn't exist
        quick_prompts = [default_quick_prompt]
        with open(QUICK_LIST_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
except Exception as e:
    alert_print(f"Error loading quick prompts: {str(e)}")
    # Initialize with default prompt if there was an error
    quick_prompts = [default_quick_prompt]
    # Create quick_prompts.json with default prompt if there was an error
    with open(QUICK_LIST_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)

# Queue file path
QUEUE_JSON_FILE = os.path.join(os.getcwd(), 'job_queue.json')

# Temp directory for queue images
temp_queue_images = os.path.join(os.getcwd(), 'temp_queue_images')
os.makedirs(temp_queue_images, exist_ok=True)


def save_settings(config):
    """Save settings to settings.ini"""
    with open(INI_FILE, 'w') as f:
        config.write(f)

def save_job_defaults_from_ui(prompt, n_prompt, lora_model, lora_weight, video_length, job_name, use_teacache, seed, steps, cfg, gs, rs, image_strength, mp4_crf, gpu_memory, keep_temp_png):
    """Save Job Defaults from UI settings"""
    config = load_settings()
    
    # Ensure sections exist
    if 'Job Defaults' not in config:
        config['Job Defaults'] = {}
    if 'Model Defaults' not in config:
        config['Model Defaults'] = {}
    
    # Save Job Defaults with consistent casing - excluding Model Defaults
    section = config['Job Defaults']
    section['DEFAULT_PROMPT'] = repr(prompt)
    section['DEFAULT_N_PROMPT'] = repr(n_prompt)
    section['DEFAULT_LORA_MODEL'] = repr(lora_model)
    section['DEFAULT_LORA_WEIGHT'] = repr(lora_weight)
    section['DEFAULT_VIDEO_LENGTH'] = repr(video_length)
    section['DEFAULT_JOB_NAME'] = repr(job_name)
    section['DEFAULT_USE_TEACACHE'] = repr(use_teacache)
    section['DEFAULT_SEED'] = repr(seed)
    section['DEFAULT_STEPS'] = repr(steps)
    section['DEFAULT_CFG'] = repr(cfg)
    section['DEFAULT_GS'] = repr(gs)
    section['DEFAULT_RS'] = repr(rs)
    section['DEFAULT_IMAGE_STRENGTH'] = repr(image_strength)
    section['DEFAULT_MP4_CRF'] = repr(mp4_crf)
    section['DEFAULT_GPU_MEMORY'] = repr(gpu_memory)
    section['DEFAULT_KEEP_TEMP_PNG'] = repr(keep_temp_png)

    
    # Update Config instance with new job defaults
    Config.DEFAULT_PROMPT = prompt
    Config.DEFAULT_N_PROMPT = n_prompt
    Config.DEFAULT_LORA_MODEL = lora_model
    Config.DEFAULT_LORA_WEIGHT = lora_weight
    Config.DEFAULT_VIDEO_LENGTH = video_length
    Config.DEFAULT_JOB_NAME = job_name
    Config.DEFAULT_USE_TEACACHE = use_teacache
    Config.DEFAULT_SEED = seed
    Config.DEFAULT_STEPS = steps
    Config.DEFAULT_CFG = cfg
    Config.DEFAULT_GS = gs
    Config.DEFAULT_RS = rs
    Config.DEFAULT_IMAGE_STRENGTH = image_strength
    Config.DEFAULT_MP4_CRF = mp4_crf
    Config.DEFAULT_GPU_MEMORY = gpu_memory
    Config.DEFAULT_KEEP_TEMP_PNG = keep_temp_png
    
    save_settings(config)
    # debug_print(f"Saved settings:\n"
    #             f"  Prompt: {section['DEFAULT_PROMPT']}\n"
    #             f"  Negative Prompt: {section['DEFAULT_N_PROMPT']}\n" 
    #             f"  LoRA Model: {section['DEFAULT_LORA_MODEL']}\n"
    #             f"  LoRA Weight: {section['DEFAULT_LORA_WEIGHT']}\n"
    #             f"  Video Length: {section['DEFAULT_VIDEO_LENGTH']}\n"
    #             f"  Job Name: {section['DEFAULT_JOB_NAME']}\n"
    #             f"  Use TeaCache: {section['DEFAULT_USE_TEACACHE']}\n"
    #             f"  Seed: {section['DEFAULT_SEED']}\n"
    #             f"  Steps: {section['DEFAULT_STEPS']}\n"
    #             f"  CFG: {section['DEFAULT_CFG']}\n"
    #             f"  GS: {section['DEFAULT_GS']}\n"
    #             f"  RS: {section['DEFAULT_RS']}\n"
    #             f"  Image Strength: {section['DEFAULT_IMAGE_STRENGTH']}\n"
    #             f"  MP4 CRF: {section['DEFAULT_MP4_CRF']}\n"
    #             f"  GPU Memory: {section['DEFAULT_GPU_MEMORY']}\n"
    #             f"  Keep Temp PNG: {section['DEFAULT_KEEP_TEMP_PNG']}\n"
    return (
        prompt,
        n_prompt,
        lora_model,
        lora_weight,
        video_length,
        job_name,
        use_teacache,
        seed,
        steps,
        cfg,
        gs,
        rs,
        image_strength,
        mp4_crf,
        gpu_memory,
        keep_temp_png
    )

@dataclass
class Config:
    """Centralized configuration for default values"""
    _instance = None
    
    # Core inputs
    DEFAULT_PROMPT: str = None
    DEFAULT_N_PROMPT: str = None
    DEFAULT_LORA_MODEL: str = None
    DEFAULT_LORA_WEIGHT: float = None
    
    # Video settings
    DEFAULT_VIDEO_LENGTH: float = None
    
    # Job settings
    DEFAULT_JOB_NAME: str = None
    
    # Processing settings
    DEFAULT_USE_TEACACHE: bool = None
    DEFAULT_SEED: int = None
    DEFAULT_LATENT_WINDOW_SIZE: int = None
    DEFAULT_STEPS: int = None
    DEFAULT_CFG: float = None
    DEFAULT_GS: float = None
    DEFAULT_RS: float = None
    DEFAULT_IMAGE_STRENGTH: float = None  # Add Image Strength parameter
    DEFAULT_MP4_CRF: int = None
    DEFAULT_GPU_MEMORY: float = None
    
    # File retention settings
    DEFAULT_KEEP_TEMP_PNG: bool = None

    # Model Defaults
    DEFAULT_TRANSFORMER: str = None
    DEFAULT_TEXT_ENCODER: str = None
    DEFAULT_TEXT_ENCODER_2: str = None
    DEFAULT_TOKENIZER: str = None
    DEFAULT_TOKENIZER_2: str = None
    DEFAULT_VAE: str = None
    DEFAULT_FEATURE_EXTRACTOR: str = None
    DEFAULT_IMAGE_ENCODER: str = None
    DEFAULT_IMAGE2IMAGE_MODEL: str = None 

    # System defaults
    OUTPUTS_FOLDER: str = None
    JOB_HISTORY_FOLDER: str = None
    DEBUG_MODE: bool = None
    KEEP_COMPLETED_JOB: bool = None
    HF_TOKEN: str = None
    PREFIX_TIMESTAMP: bool = None
    PREFIX_SOURCE_IMAGE_NAME: bool = None
    REMOVE_HEXID_SUFFIX: bool = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


    @classmethod
    def get_original_defaults(cls):
        """Returns a dictionary of original default values - this is the single source of truth for defaults"""
        return {
            # Core inputs
            'DEFAULT_PROMPT': "The girl dances gracefully, with clear movements, full of charm.",
            'DEFAULT_N_PROMPT': "",
            'DEFAULT_LORA_MODEL': "None",
            'DEFAULT_LORA_WEIGHT': 1.0,
            # Video settings
            'DEFAULT_VIDEO_LENGTH': 5.0,
            # Job settings
            'DEFAULT_JOB_NAME': "Job-",
            # Processing settings
            'DEFAULT_USE_TEACACHE': True,
            'DEFAULT_SEED': -1,
            'DEFAULT_LATENT_WINDOW_SIZE': 9,
            'DEFAULT_STEPS': 25,
            'DEFAULT_CFG': 1.0,
            'DEFAULT_GS': 10.0,
            'DEFAULT_RS': 0.0,
            'DEFAULT_IMAGE_STRENGTH': 1.0,  # Default to no image to image generation
            'DEFAULT_MP4_CRF': 16,
            'DEFAULT_GPU_MEMORY': 6.0,
            # File retention settings
            'DEFAULT_KEEP_TEMP_PNG': True,
            # Model Defaults
            'DEFAULT_TRANSFORMER': "models--lllyasviel--FramePack_F1_I2V_HY_20250503",
            'DEFAULT_TEXT_ENCODER': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TEXT_ENCODER_2': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TOKENIZER': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TOKENIZER_2': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_VAE': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_FEATURE_EXTRACTOR': "models--lllyasviel--flux_redux_bfl",
            'DEFAULT_IMAGE_ENCODER': "models--lllyasviel--flux_redux_bfl",
            'DEFAULT_IMAGE2IMAGE_MODEL': "models--lllyasviel--flux_redux_bfl",
            # System settings
            'OUTPUTS_FOLDER': './outputs/',
            'JOB_HISTORY_FOLDER': './job_history/',
            'DEBUG_MODE': False,
            'KEEP_COMPLETED_JOB': True,
            'HF_TOKEN': 'add token here',
            'PREFIX_TIMESTAMP': False,
            'PREFIX_SOURCE_IMAGE_NAME': False,
            'REMOVE_HEXID_SUFFIX': False
        }

    @classmethod
    def from_settings(cls, config):
        """Create Config instance from settings.ini values"""
        instance = cls()
        
        # Load Job Defaults section
        section = config['Job Defaults']
        section_keys = {k.upper(): k for k in section.keys()}
        
        # Load all non-model values from Job Defaults
        for key, default_value in instance.get_original_defaults().items():
            if key.startswith('DEFAULT_') and not any(model_type in key.lower() for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model']):
                try:
                    # Look up the actual key in a case-insensitive way
                    actual_key = section_keys.get(key.upper(), key)
                    if actual_key not in section:
                        actual_key = section_keys.get(key.lower(), key)
                    
                    value = section.get(actual_key, str(default_value))
                    
                    # Handle different types appropriately
                    if isinstance(default_value, bool):
                        parsed_value = str(value).lower() in ('true', 't', 'yes', 'y', '1')
                    elif isinstance(default_value, int):
                        parsed_value = int(float(str(value).strip("'")))
                    elif isinstance(default_value, float):
                        parsed_value = float(str(value).strip("'"))
                    else:
                        parsed_value = str(value).strip("'")
                    
                    setattr(instance, key, parsed_value)
                except Exception as e:
                    alert_print(f"Error loading {key}: {str(e)}, using default value: {default_value}")
                    setattr(instance, key, default_value)
                    section[key] = str(default_value)
                    save_settings(config)

        # Load Model Defaults section
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
        model_section = config['Model Defaults']
        model_section_keys = {k.upper(): k for k in model_section.keys()}
        
        # Load model values from Model Defaults
        for key, default_value in instance.get_original_defaults().items():
            if key.startswith('DEFAULT_') and any(model_type in key.lower() for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model']):
                try:
                    # Look up the actual key in a case-insensitive way
                    actual_key = model_section_keys.get(key.upper(), key)
                    if actual_key not in model_section:
                        actual_key = model_section_keys.get(key.lower(), key)
                    
                    value = model_section.get(actual_key, str(default_value))
                    # Remove quotes if present
                    parsed_value = str(value).strip("'").strip('"')
                    if parsed_value.lower() == 'none':
                        parsed_value = default_value
                    setattr(instance, key, parsed_value)
                except Exception as e:
                    alert_print(f"Error loading model setting {key}: {str(e)}, using default value: {default_value}")
                    setattr(instance, key, default_value)
                    model_section[key] = str(default_value)
                    save_settings(config)

        # Load System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        section_keys = {k.upper(): k for k in section.keys()}
        
        # Load system values from settings, using defaults as fallback
        for key, default_value in instance.get_original_defaults().items():
            if not key.startswith('DEFAULT_'):
                try:
                    # Look up the actual key in a case-insensitive way
                    actual_key = section_keys.get(key.upper(), key)
                    if actual_key not in section:
                        actual_key = section_keys.get(key.lower(), key)
                    
                    value = section.get(actual_key, str(default_value))
                    
                    # Handle different types appropriately
                    if isinstance(default_value, bool):
                        parsed_value = str(value).lower() in ('true', 't', 'yes', 'y', '1')
                    elif isinstance(default_value, int):
                        parsed_value = int(float(str(value).strip("'")))
                    elif isinstance(default_value, float):
                        parsed_value = float(str(value).strip("'"))
                    else:
                        parsed_value = str(value).strip("'")
                    
                    setattr(instance, key, parsed_value)
                except Exception as e:
                    alert_print(f"Error loading system setting {key}: {str(e)}, using default value: {default_value}")
                    setattr(instance, key, default_value)
                    section[key] = str(default_value)
                    save_settings(config)

        return instance

    @classmethod
    def to_settings(cls, config):
        """Save Config instance values to settings.ini"""
        # Save Job Defaults section
        section = config['Job Defaults']
        job_defaults = [
            'DEFAULT_PROMPT', 'DEFAULT_N_PROMPT', 'DEFAULT_LORA_MODEL', 'DEFAULT_LORA_WEIGHT', 'DEFAULT_VIDEO_LENGTH',
            'DEFAULT_GS', 'DEFAULT_STEPS', 'DEFAULT_USE_TEACACHE', 'DEFAULT_SEED',
            'DEFAULT_CFG', 'DEFAULT_RS', 'DEFAULT_IMAGE_STRENGTH', 'DEFAULT_MP4_CRF', 'DEFAULT_GPU_MEMORY',
            'DEFAULT_KEEP_TEMP_PNG'
        ]
        for key in job_defaults:
            section[key] = repr(getattr(cls, key))

        # Save Model Defaults section
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
        section = config['Model Defaults']
        model_defaults = [
            'DEFAULT_TRANSFORMER', 'DEFAULT_TEXT_ENCODER', 'DEFAULT_TEXT_ENCODER_2',
            'DEFAULT_TOKENIZER', 'DEFAULT_TOKENIZER_2', 'DEFAULT_VAE',
            'DEFAULT_FEATURE_EXTRACTOR', 'DEFAULT_IMAGE_ENCODER', 'DEFAULT_IMAGE2IMAGE_MODEL'
        ]
        for key in model_defaults:
            section[key] = repr(getattr(cls, key))

        # Save System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        system_defaults = ['OUTPUTS_FOLDER', 'JOB_HISTORY_FOLDER', 'DEBUG_MODE', 'KEEP_COMPLETED_JOB', 'PREFIX_TIMESTAMP', 'PREFIX_SOURCE_IMAGE_NAME', 'REMOVE_HEXID_SUFFIX']
        for key in system_defaults:
            section[key] = str(getattr(cls, key))

        save_settings(config)

    @classmethod
    def get_default_prompt_tuple(cls):
        """Returns a tuple of all default values in the correct order"""
        return (
            cls.DEFAULT_PROMPT,
            cls.DEFAULT_N_PROMPT,
            cls.DEFAULT_LORA_MODEL,
            cls.DEFAULT_LORA_WEIGHT,
            cls.DEFAULT_VIDEO_LENGTH,
            cls.DEFAULT_JOB_NAME,
            cls.OUTPUTS_FOLDER,
            cls.JOB_HISTORY_FOLDER,
            cls.DEFAULT_USE_TEACACHE,
            cls.DEFAULT_SEED,
            cls.DEFAULT_STEPS,
            cls.DEFAULT_CFG,
            cls.DEFAULT_GS,
            cls.DEFAULT_RS,
            cls.DEFAULT_IMAGE_STRENGTH,  # Add Image Strength
            cls.DEFAULT_MP4_CRF,
            cls.DEFAULT_GPU_MEMORY,
            cls.KEEP_COMPLETED_JOB,
            cls.DEFAULT_KEEP_TEMP_PNG
        )

    @classmethod
    def get_default_prompt_dict(cls):
        """Returns a dictionary of default values for quick prompts"""
        return {
            'prompt': cls.DEFAULT_PROMPT,
            'n_prompt': cls.DEFAULT_N_PROMPT,
            'lora_model': cls.DEFAULT_LORA_MODEL,
            'lora_weight': cls.DEFAULT_LORA_WEIGHT,
            'video_length': cls.DEFAULT_VIDEO_LENGTH,
            'job_name': cls.DEFAULT_JOB_NAME,
            'outputs_folder': cls.OUTPUTS_FOLDER,
            'job_history_folder': cls.JOB_HISTORY_FOLDER,
            'use_teacache': cls.DEFAULT_USE_TEACACHE,
            'seed': cls.DEFAULT_SEED,
            'steps': cls.DEFAULT_STEPS,
            'cfg': cls.DEFAULT_CFG,
            'gs': cls.DEFAULT_GS,
            'rs': cls.DEFAULT_RS,
            'image_strength': cls.DEFAULT_IMAGE_STRENGTH,  # Add Image Strength
            'mp4_crf': cls.DEFAULT_MP4_CRF,
            'gpu_memory': cls.DEFAULT_GPU_MEMORY,
            'keep_completed_job': cls.KEEP_COMPLETED_JOB,
            'keep_temp_png': cls.DEFAULT_KEEP_TEMP_PNG
        }

def load_settings():
    """Load settings from settings.ini file and ensure all sections and values exist"""
    config = configparser.ConfigParser()
    
    # Get default values
    default_values = Config.get_original_defaults()
    
    # Create default sections if file doesn't exist
    if not os.path.exists(INI_FILE):
        # Split defaults into appropriate sections
        job_defaults = {k: v for k, v in default_values.items() 
                       if k.startswith('DEFAULT_') and not any(model_type in k.lower() 
                       for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model'])}
        
        model_defaults = {k: v for k, v in default_values.items() 
                         if k.startswith('DEFAULT_') and any(model_type in k.lower() 
                         for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model'])}
        
        system_defaults = {k: v for k, v in default_values.items() if not k.startswith('DEFAULT_')}
        
        # Add job-specific folder paths to Job Defaults
        job_defaults['DEFAULT_OUTPUTS_FOLDER'] = repr(default_values['OUTPUTS_FOLDER'])
        job_defaults['DEFAULT_JOB_HISTORY_FOLDER'] = repr(default_values['JOB_HISTORY_FOLDER'])
        
        config['Job Defaults'] = {k: repr(v) for k, v in job_defaults.items()}
        config['Model Defaults'] = {k: repr(v) for k, v in model_defaults.items()}
        config['System Defaults'] = {k: str(v) for k, v in system_defaults.items()}
        
        with open(INI_FILE, 'w') as f:
            config.write(f)
    else:
        # Read existing config
        config.read(INI_FILE)
        
        # Ensure Job Defaults section exists with all values
        if 'Job Defaults' not in config:
            config['Job Defaults'] = {}
        
        # Check and add any missing values in Job Defaults
        for key, value in default_values.items():
            if key.startswith('DEFAULT_') and not any(model_type in key.lower() for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model']):
                if key not in config['Job Defaults']:
                    config['Job Defaults'][key] = repr(value)
        
        # Add job-specific folder paths if missing
        if 'DEFAULT_OUTPUTS_FOLDER' not in config['Job Defaults']:
            config['Job Defaults']['DEFAULT_OUTPUTS_FOLDER'] = repr(default_values['OUTPUTS_FOLDER'])
        if 'DEFAULT_JOB_HISTORY_FOLDER' not in config['Job Defaults']:
            config['Job Defaults']['DEFAULT_JOB_HISTORY_FOLDER'] = repr(default_values['JOB_HISTORY_FOLDER'])
        
        # Ensure Model Defaults section exists with all values
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Check and add any missing model values
        for key, value in default_values.items():
            if key.startswith('DEFAULT_') and any(model_type in key.lower() for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder']):
                if key not in config['Model Defaults']:
                    config['Model Defaults'][key] = repr(value)
        
        # Ensure System Defaults section exists with all values
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        
        # Check and add any missing system values
        for key, value in default_values.items():
            if not key.startswith('DEFAULT_'):
                if key not in config['System Defaults']:
                    config['System Defaults'][key] = str(value)
        
        # Save any changes made to the config
        with open(INI_FILE, 'w') as f:
            config.write(f)
    
    return config

def save_settings_from_ui(outputs_folder, job_history_folder, debug_mode, keep_completed_job, prefix_timestamp, prefix_source_image_name, remove_hexid_suffix,
                         transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                         vae, feature_extractor, image_encoder, image2image_model):
    """Save settings from UI inputs"""
    settings_config = configparser.ConfigParser()
    
    # Create required sections
    settings_config['Job Defaults'] = {}
    settings_config['Model Defaults'] = {}
    settings_config['System Defaults'] = {}
    
    # System Defaults
    settings_config['System Defaults'] = {
        'OUTPUTS_FOLDER': repr(outputs_folder),
        'JOB_HISTORY_FOLDER': repr(job_history_folder),
        'DEBUG_MODE': repr(debug_mode),
        'KEEP_COMPLETED_JOB': repr(keep_completed_job),
        'PREFIX_TIMESTAMP': repr(prefix_timestamp),
        'PREFIX_SOURCE_IMAGE_NAME': repr(prefix_source_image_name),
        'REMOVE_HEXID_SUFFIX': repr(remove_hexid_suffix)
    }
    
    # Model Defaults
    settings_config['Model Defaults'] = {
        'DEFAULT_TRANSFORMER': repr(transformer),
        'DEFAULT_TEXT_ENCODER': repr(text_encoder),
        'DEFAULT_TEXT_ENCODER_2': repr(text_encoder_2),
        'DEFAULT_TOKENIZER': repr(tokenizer),
        'DEFAULT_TOKENIZER_2': repr(tokenizer_2),
        'DEFAULT_VAE': repr(vae),
        'DEFAULT_FEATURE_EXTRACTOR': repr(feature_extractor),
        'DEFAULT_IMAGE_ENCODER': repr(image_encoder),
        'DEFAULT_IMAGE2IMAGE_MODEL': repr(image2image_model)
    }
    
    # Update global Config object
    Config.OUTPUTS_FOLDER = outputs_folder
    Config.JOB_HISTORY_FOLDER = job_history_folder
    Config.DEBUG_MODE = debug_mode
    Config.KEEP_COMPLETED_JOB = keep_completed_job
    Config.PREFIX_TIMESTAMP = prefix_timestamp
    Config.PREFIX_SOURCE_IMAGE_NAME = prefix_source_image_name
    Config.REMOVE_HEXID_SUFFIX = remove_hexid_suffix
    Config.DEFAULT_TRANSFORMER = transformer
    Config.DEFAULT_TEXT_ENCODER = text_encoder
    Config.DEFAULT_TEXT_ENCODER_2 = text_encoder_2
    Config.DEFAULT_TOKENIZER = tokenizer
    Config.DEFAULT_TOKENIZER_2 = tokenizer_2
    Config.DEFAULT_VAE = vae
    Config.DEFAULT_FEATURE_EXTRACTOR = feature_extractor
    Config.DEFAULT_IMAGE_ENCODER = image_encoder
    Config.DEFAULT_IMAGE2IMAGE_MODEL = image2image_model
    
    # Load existing settings to preserve other values
    existing_config = load_settings()
    
    # Update with new settings
    for section in settings_config.sections():
        if section not in existing_config:
            existing_config[section] = {}
        for key, value in settings_config[section].items():
            existing_config[section][key] = value
    
    # Save updated settings
    save_settings(existing_config)
    return "Settings saved successfully. Restart required for changes to take effect."


def restore_job_defaults():
    """Restore Job Defaults to original values"""
    # Get the original defaults
    defaults = Config.get_original_defaults()
    debug_print("Restoring original defaults:")
    
    # Load the config file
    config = load_settings()
    if 'Job Defaults' not in config:
        config['Job Defaults'] = {}
    section = config['Job Defaults']
    
    # Update the section with all non-model default values
    for key, value in defaults.items():
        if (key.startswith('DEFAULT_') and 
            not any(model_type in key.lower() for model_type in 
                ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model'])):
            debug_print(f"  Setting {key} = {value}")
            # Format value based on type
            if isinstance(value, str):
                section[key.lower()] = repr(value)  # Use repr to properly quote strings
            elif isinstance(value, bool):
                section[key.lower()] = str(value)  # Booleans as 'True' or 'False'
            elif isinstance(value, (int, float)):
                section[key.lower()] = str(value)  # Numbers as plain strings
            else:
                section[key.lower()] = repr(value)  # Default to repr for other types
    
    # Save to settings.ini
    save_settings(config)
    debug_print("Saved original defaults to settings.ini")
    
    # Return values in the order expected by the UI
    return [
        defaults['DEFAULT_PROMPT'],
        defaults['DEFAULT_N_PROMPT'],
        defaults['DEFAULT_LORA_MODEL'],
        defaults['DEFAULT_LORA_WEIGHT'],
        defaults['DEFAULT_VIDEO_LENGTH'],
        defaults['DEFAULT_JOB_NAME'],
        defaults['DEFAULT_USE_TEACACHE'],
        defaults['DEFAULT_SEED'],
        defaults['DEFAULT_STEPS'],
        defaults['DEFAULT_CFG'],
        defaults['DEFAULT_GS'],
        defaults['DEFAULT_RS'],
        defaults['DEFAULT_IMAGE_STRENGTH'],
        defaults['DEFAULT_MP4_CRF'],
        defaults['DEFAULT_GPU_MEMORY'],
        defaults['DEFAULT_KEEP_TEMP_PNG']
    ]

def save_queue():
    try:
        jobs = [job.to_dict() for job in job_queue]
        with open(QUEUE_JSON_FILE, 'w') as f:
            json.dump(jobs, f, indent=4)
            debug_print(f"Queue saved to {QUEUE_JSON_FILE} (called from {sys._getframe().f_back.f_code.co_name})")
        return True
    except Exception as e:
        alert_print(f"Error saving queue: {str(e)}")
        traceback.print_exc()
        return False

def load_queue():
    try:
        if os.path.exists(QUEUE_JSON_FILE):
            try:
                with open(QUEUE_JSON_FILE, 'r') as f:
                    jobs = json.load(f)
                    debug_print(f"Queue loaded from {QUEUE_JSON_FILE} (called from {sys._getframe().f_back.f_code.co_name})")
            except json.JSONDecodeError as e:
                alert_print(f"Error reading queue file (corrupted JSON): {str(e)}")
                sys.exit(1)

            # Clear existing queue and load valid jobs from file
            job_queue.clear()
            valid_jobs = []
            for job_data in jobs:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    valid_jobs.append(job)
                else:
                    alert_print(f"Skipping invalid job data: {job_data}")
            
            # Only update queue if we have valid jobs
            if valid_jobs:
                job_queue.extend(valid_jobs)
                debug_print(f"Loaded {len(valid_jobs)} valid jobs")
            
            return job_queue
        else:
            alert_print("No queue file found")
            return []
    except Exception as e:
        alert_print(f"Error loading queue: {str(e)}")
        debug_print(f"Error details: {traceback.format_exc()}")
        return []

def setup_local_variables():
    """Set up local variables from Config values"""
    global job_history_folder, outputs_folder, debug_mode, keep_completed_job, prefix_timestamp, prefix_source_image_name, remove_hexid_suffix

    
    job_history_folder = Config.JOB_HISTORY_FOLDER
    outputs_folder = Config.OUTPUTS_FOLDER
    debug_mode = Config.DEBUG_MODE
    keep_completed_job = Config.KEEP_COMPLETED_JOB
    prefix_timestamp = Config.PREFIX_TIMESTAMP
    prefix_source_image_name = Config.PREFIX_SOURCE_IMAGE_NAME
    remove_hexid_suffix = Config.REMOVE_HEXID_SUFFIX

# Initialize settings first
settings_config = load_settings()
Config = Config.from_settings(settings_config)
print("Loaded settings into Config:")


# Create necessary directories using values from Config
os.makedirs(Config.OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(Config.JOB_HISTORY_FOLDER, exist_ok=True)


# Initialize job queue as a list
job_queue = []

# Set up local variables
setup_local_variables()

# Initialize quick prompts
DEFAULT_PROMPTS = [
    Config.get_default_prompt_dict(),
    {
        'prompt': 'A character doing some simple body movements.',
        'n_prompt': '',
        'lora_model': Config.DEFAULT_LORA_MODEL,
        'lora_weight': Config.DEFAULT_LORA_WEIGHT,
        'job_name': Config.DEFAULT_JOB_NAME,
        'length': Config.DEFAULT_VIDEO_LENGTH,
        'gs': Config.DEFAULT_GS,
        'steps': Config.DEFAULT_STEPS,
        'use_teacache': Config.DEFAULT_USE_TEACACHE,
        'seed': Config.DEFAULT_SEED,
        'cfg': Config.DEFAULT_CFG,
        'rs': Config.DEFAULT_RS,
        'image_strength': Config.DEFAULT_IMAGE_STRENGTH,
        'mp4_crf': Config.DEFAULT_MP4_CRF,
        'gpu_memory': Config.DEFAULT_GPU_MEMORY,
        'keep_temp_png': Config.DEFAULT_KEEP_TEMP_PNG
    }
]

# Load existing prompts or create the file with defaults
if os.path.exists(QUICK_LIST_FILE):
    with open(QUICK_LIST_FILE, 'r') as f:
        quick_prompts = json.load(f)
else:
    quick_prompts = DEFAULT_PROMPTS.copy()
    with open(QUICK_LIST_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)

@dataclass
class QueuedJob:
    # Required parameters (no defaults) first
    prompt: str
    n_prompt: str
    lora_model: str
    lora_weight: float
    video_length: float
    job_name: str
    use_teacache: bool
    seed: int
    steps: int
    cfg: float
    gs: float
    rs: float
    image_strength: float
    mp4_crf: float
    gpu_memory: float
    image_path: str
    # Optional parameters (with defaults) after
    outputs_folder: str = str(Config.OUTPUTS_FOLDER)
    job_history_folder: str = str(Config.JOB_HISTORY_FOLDER)
    latent_window_size: int = Config.DEFAULT_LATENT_WINDOW_SIZE
    keep_completed_job: bool = Config.KEEP_COMPLETED_JOB
    keep_temp_png: bool = False
    thumbnail: str = ""
    source_name: str = ""
    status: str = "pending"
    change_job_name: str = None  # Add this line for the new attribute
    error_message: str = "None"

    def to_dict(self):
        try:
            # Always include all fields with their current values
            return {
                'prompt': self.prompt,
                'n_prompt': self.n_prompt,
                'lora_model': self.lora_model,
                'lora_weight': self.lora_weight,
                'video_length': self.video_length,
                'job_name': self.job_name,
                'use_teacache': self.use_teacache,
                'seed': self.seed,
                'steps': self.steps,
                'cfg': self.cfg,
                'gs': self.gs,
                'rs': self.rs,
                'image_strength': getattr(self, 'image_strength', 1.0),  # Always include, default to 1.0 if not set
                'mp4_crf': self.mp4_crf,
                'gpu_memory': self.gpu_memory,
                'image_path': self.image_path,
                'outputs_folder': str(self.outputs_folder),
                'job_history_folder': str(self.job_history_folder),
                'latent_window_size': self.latent_window_size,
                'keep_completed_job': self.keep_completed_job,
                'keep_temp_png': self.keep_temp_png,
                'thumbnail': self.thumbnail,
                'source_name': self.source_name,
                'status': self.status,
                'error_message': self.error_message
            }
        except Exception as e:
            alert_print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            # Define the expected order of keys and their default values
            expected_keys = {
                'prompt': '',
                'n_prompt': Config.DEFAULT_N_PROMPT,
                'lora_model': Config.DEFAULT_LORA_MODEL,
                'lora_weight': Config.DEFAULT_LORA_WEIGHT,
                'video_length': Config.DEFAULT_VIDEO_LENGTH,
                'job_name': '',
                'use_teacache': Config.DEFAULT_USE_TEACACHE,
                'seed': Config.DEFAULT_SEED,
                'steps': Config.DEFAULT_STEPS,
                'cfg': Config.DEFAULT_CFG,
                'gs': Config.DEFAULT_GS,
                'rs': Config.DEFAULT_RS,
                'image_strength': Config.DEFAULT_IMAGE_STRENGTH,
                'mp4_crf': Config.DEFAULT_MP4_CRF,
                'gpu_memory': Config.DEFAULT_GPU_MEMORY,
                'image_path': '',
                'outputs_folder': str(Config.OUTPUTS_FOLDER),
                'job_history_folder': str(Config.JOB_HISTORY_FOLDER),
                'latent_window_size': Config.DEFAULT_LATENT_WINDOW_SIZE,
                'keep_completed_job': Config.KEEP_COMPLETED_JOB,
                'keep_temp_png': Config.DEFAULT_KEEP_TEMP_PNG,
                'thumbnail': '',
                'source_name': "",
                'status': 'pending',
                'error_message': "None"
            }

            # Create a new dictionary with keys in the correct order
            ordered_data = {}
            
            # First, add all expected keys with their values from data or defaults
            for key, default_value in expected_keys.items():
                if key in data:
                    try:
                        if isinstance(default_value, bool):
                            ordered_data[key] = bool(data[key])
                        elif isinstance(default_value, int):
                            ordered_data[key] = int(float(str(data[key]).strip("'")))
                        elif isinstance(default_value, float):
                            ordered_data[key] = float(str(data[key]).strip("'"))
                        else:
                            ordered_data[key] = str(data[key]).strip("'")
                    except (ValueError, TypeError):
                        # If conversion fails, use the default value
                        ordered_data[key] = default_value
                else:
                    # If key is missing, use the default value
                    ordered_data[key] = default_value

            # Create the job with the ordered data
            return cls(**ordered_data)

        except Exception as e:
            alert_print(f"Error creating job from dict: {str(e)}")
            debug_print(f"Problem data: {data}")
            debug_print(f"Error details: {traceback.format_exc()}")
            return None


def job_name_prefix(name):
    # Remove the suffix after the last '-' or '_'
    for sep in ['-', '_']:
        if sep in name:
            name = name.rsplit(sep, 1)[0]
    return name


def save_image_to_temp(image: np.ndarray, job_name: str) -> str:
    """Save image to temp directory and return the path"""
    try:
        # Handle Gallery tuple format
        if isinstance(image, tuple):
            image = image[0]  # Get the file path from the tuple
        if isinstance(image, str):
            # If it's a path, open the image
            pil_image = Image.open(image)
            # Only convert if it's RGBA
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
        else:
            # If it's already a numpy array
            pil_image = Image.fromarray(image)
            # Only convert if it's RGBA
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            
        filename = f"queue_image_{job_name}.png"
        filepath = os.path.join(temp_queue_images, filename)
        # Save image
        pil_image.save(filepath)
        return filepath
    except Exception as e:
        alert_print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return ""

def add_to_queue(prompt, n_prompt, lora_model, lora_weight, input_image, video_length, job_name, create_job_outputs_folder, create_job_history_folder, use_teacache, seed, steps, cfg, gs, rs, image_strength, mp4_crf, gpu_memory, create_job_keep_completed_job, keep_temp_png, status="pending", source_name=""):
    """Add a new job to the queue"""
    try:
        # Ensure lora_model is always a string
        if not isinstance(lora_model, str):
            lora_model = "None"
        # Set default values for optional parameters
        if create_job_outputs_folder is None:
            create_job_outputs_folder = Config.OUTPUTS_FOLDER
        if create_job_history_folder is None:
            create_job_history_folder = Config.JOB_HISTORY_FOLDER
        if create_job_keep_completed_job is None:
            create_job_keep_completed_job = Config.KEEP_COMPLETED_JOB
            
        hex_id = uuid.uuid4().hex[:8]
        job_name = f"{job_name}_{hex_id}"

        # Handle text-to-video case
        
        if input_image is None:
            job = QueuedJob(
                prompt=prompt,
                n_prompt=n_prompt,
                lora_model=lora_model,
                lora_weight=lora_weight,
                video_length=video_length,
                job_name=job_name,
                outputs_folder=create_job_outputs_folder,
                job_history_folder=create_job_history_folder,
                use_teacache=use_teacache,
                seed=seed,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                image_strength=image_strength,  # Add Image Strength
                mp4_crf=mp4_crf,
                gpu_memory=gpu_memory,
                keep_completed_job=create_job_keep_completed_job,
                keep_temp_png=keep_temp_png,
                image_path="text2video",
                thumbnail="",
                source_name="",
                status=status
            )
            # Find the first completed job
            insert_index = len(job_queue)
            for i, existing_job in enumerate(job_queue):
                if existing_job.status == "completed":
                    insert_index = i
                    break
            job_queue.insert(insert_index, job)
            job.thumbnail = create_thumbnail(job, status_change=True)  
            save_queue()
            debug_print(f"Total jobs in the queue:{len(job_queue)}")
            return job_name

        # Handle image-to-video case
        if isinstance(input_image, np.ndarray):
            # Save the input image
            image_path = save_image_to_temp(input_image, job_name)
            if image_path == "text2video":
                return None

            job = QueuedJob(
                prompt=prompt,
                n_prompt=n_prompt,
                lora_model=lora_model,
                lora_weight=lora_weight,
                video_length=video_length,
                job_name=job_name,
                outputs_folder=create_job_outputs_folder,
                job_history_folder=create_job_history_folder,
                use_teacache=use_teacache,
                seed=seed,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                image_strength=image_strength,  # Add Image Strength
                mp4_crf=mp4_crf,
                gpu_memory=gpu_memory,
                keep_completed_job=create_job_keep_completed_job,
                keep_temp_png=keep_temp_png,
                image_path=image_path,
                thumbnail="",
                source_name=source_name,
                status=status
            )
            # Find the first completed job
            insert_index = len(job_queue)
            for i, existing_job in enumerate(job_queue):
                if existing_job.status == "completed":
                    insert_index = i
                    break
            job_queue.insert(insert_index, job)
            job.thumbnail = create_thumbnail(job, status_change=False)
            save_queue()
            debug_print(f"Total jobs in the queue:{len(job_queue)}")
            return job_name
        else:
            alert_print("Invalid input image format")
            return None
    except Exception as e:
        alert_print(f"Error adding to queue: {str(e)}")
        return None

def create_thumbnail(job, status_change=False):
    # Print caller info
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller_info = inspect.getframeinfo(caller_frame)
    debug_print(f"create_thumbnail() called from {caller_info.function} at line {caller_info.lineno}")
    """Create a thumbnail for a job"""
    # Add status
    status_color = {
        "pending": "yellow",
        "processing": "blue",
        "completed": "green",
        "failed": "red"
    }.get(job.status, "white")
    
    # Initialize status overlay
    status_overlay = "RUNNING" if job.status == "processing" else ("DONE" if job.status == "completed" else job.status.upper())
    
    try:
        # Try to load arial font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            try:
                # DejaVuSans ships with Pillow and is usually available
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except (OSError, IOError):
                # Final fallback to a simple built-in bitmap font
                font = ImageFont.load_default()

        # Handle text-to-video case (job.image_path is text2video)
        if job.image_path == "text2video":
            debug_print(f"created a blank thumbnail for Text2Video job {job.job_name} here is the path {job.image_path}")
            if not job.thumbnail or status_change:  # Create new thumbnail if none exists or status changed
                # Create a text-to-video thumbnail
                img = Image.new('RGB', (200, 200), color='black')
                draw = ImageDraw.Draw(img)
                # Calculate text positions using the same approach as the example
                text1 = "Text to Video"
                text2 = "Generation"
                text3 = status_overlay
                
                # Get text sizes
                text1_bbox = draw.textbbox((0, 0), text1, font=font)
                text2_bbox = draw.textbbox((0, 0), text2, font=font)
                text3_bbox = draw.textbbox((0, 0), text3, font=font)
                
                # Calculate positions to center text
                x1 = (200 - (text1_bbox[2] - text1_bbox[0])) // 2
                x2 = (200 - (text2_bbox[2] - text2_bbox[0])) // 2
                x3 = (200 - (text3_bbox[2] - text3_bbox[0])) // 2
                
                # Add text-to-video indicator with calculated positions
                draw.text((x1, 80), text1, fill='white', font=font)
                draw.text((x2, 100), text2, fill='white', font=font)
                draw.text((x3, 120), text3, fill=status_color, font=font)
                
                # Save thumbnail
                thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
                img.save(thumbnail_path)
                job.thumbnail = thumbnail_path
                save_queue()
                debug_print(f"saved thumbnail for thumb {job.job_name}")
            return job.thumbnail

        # Handle missing image-based cases 
        if job.image_path != "text2video" and not os.path.exists(job.image_path) and not job.thumbnail:
            # Create missing image thumbnail
            img = Image.new('RGB', (200, 200), color='black')
            draw = ImageDraw.Draw(img)
            
            # Calculate text position for "MISSING IMAGE"
            text = "MISSING IMAGE"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            x = (200 - (text_bbox[2] - text_bbox[0])) // 2
            y = (200 - (text_bbox[3] - text_bbox[1])) // 2
            
            # Add black outline to make text more readable
            for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text((x+offset[0], y+offset[1]), text, font=font, fill=(255,255,255))
            draw.text((x, y), text, fill='red', font=font)
            
            # Save thumbnail
            thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
            img.save(thumbnail_path)
            job.thumbnail = thumbnail_path
            save_queue()
            debug_print(f"saved thumbnail for thumb {job.job_name}")
            return thumbnail_path

        # Normal case - create thumbnail from existing image add overlay
        if job.image_path != "text2video":
            img = Image.open(job.image_path)
            width, height = img.size
            new_height = 200
            new_width = int((new_height / height) * width)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Create a new image with padding
            new_img = Image.new('RGB', (200, 200), color='black')
            new_img.paste(img, ((200 - img.width) // 2, (200 - img.height) // 2))
            # Add status text if provided
            if status_change:
                draw = ImageDraw.Draw(new_img)
                # Calculate text position for status overlay
                text_bbox = draw.textbbox((0, 0), status_overlay, font=font)
                x = (200 - (text_bbox[2] - text_bbox[0])) // 2
                y = (200 - (text_bbox[3] - text_bbox[1])) // 2
                
                # Add black outline to make text more readable
                for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    draw.text((x+offset[0], y+offset[1]), status_overlay, font=font, fill=(0,0,0))
                draw.text((x, y), status_overlay, fill=status_color, font=font)
            
            # Save thumbnail
            thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
            new_img.save(thumbnail_path)
            job.thumbnail = thumbnail_path
            save_queue()
            debug_print(f"saved thumbnail for thumb_{job.job_name}")
        return thumbnail_path
    except Exception as e:
        alert_print(f"Error creating thumbnail: {str(e)}")
        return ""

# Global mapping from thumbnail path to caption
thumbnail_captions = {}

def update_queue_display():
    global thumbnail_captions
    try:
        queue_data = []
        thumbnail_captions = {}
        for job in job_queue:
            # Only check for missing images if the job is not being deleted
            if job.status != "deleting":
                # Check if both queue image and thumbnail are missing
                queue_image_missing = not os.path.exists(job.image_path) if job.image_path else True
                thumbnail_missing = not os.path.exists(job.thumbnail) if job.thumbnail else True
                if queue_image_missing and thumbnail_missing:
                    new_thumbnail = create_thumbnail(job, status_change=False)
                elif not job.thumbnail and job.image_path:
                    job.thumbnail = create_thumbnail(job, status_change=False)

            if job.thumbnail:
                caption = f"{job.prompt} \n Negative: {job.n_prompt} \n Lora: {job.lora_model}\nLength: {job.video_length}s\nGS: {job.gs}"
                thumbnail_captions[job.thumbnail] = caption
                queue_data.append(job.thumbnail)
        current_counts = count_jobs_by_status()
        return gr.update(value=queue_data, label=f"{current_counts['pending']} Jobs pending in the Queue")
    except Exception as e:
        alert_print(f"Error updating queue display: {str(e)}")
        traceback.print_exc()
        return gr.update(value=[], label="0 Jobs pending in the Queue")

def update_queue_table():
    """Update the queue table display with current jobs from JSON"""
    data = []
    for job in job_queue:
        # Add job data to display
        if not job.thumbnail or not os.path.exists(job.thumbnail):
            create_thumbnail(job, status_change=True)
            
        try:
            # Read the image and convert to base64
            with open(job.thumbnail, "rb") as img_file:
                import base64
                img_data = base64.b64encode(img_file.read()).decode()
            
            # Set background color based on status
            bg_color = {
                "pending": "#ffffff",    # White
                "processing": "#fffde6",  # Light yellow
                "completed": "#e6ffe6",   # Light green
                "failed": "#ffe6e6"       # Light red
            }.get(job.status, "#ffffff")  # Default white

            img_md = f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;"><div style="text-align: center; font-size: 0.8em; color: #666; margin-bottom: 5px;">{job.status}</div><div style="text-align: center; font-size: 0.8em; color: #666;">{job.job_name}</div><div style="text-align: center; font-size: 0.8em; color: #666;">seed: {job.seed}</div><div style="text-align: center; font-size: 0.8em; color: #666;">Length: {job.video_length:.1f}s</div><img src="data:image/png;base64,{img_data}" alt="Input" style="max-width:100px; max-height:100px; display: block; margin: auto; object-fit: contain; transform: scale(0.75); transform-origin: top left;" /></div>'
        except Exception as e:
            alert_print(f"Error converting image to base64: {str(e)}")
            img_md = ""

        # Use full prompt text without truncation
        if job.error_message == "None":
            prompt_cell = (
                f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 5px;">'
                f'<span style="white-space: normal; word-wrap: break-word; display: block; width: 100%;">'
                f'Prompt: {job.prompt}<br>'
                f'Negative prompt: {job.n_prompt}<br>'
                f'Lora Model: {job.lora_model}{f" - Lora Weight: {job.lora_weight}" if job.lora_model != "None" else ""}<br>'
                f'Image Strength (AKA Denoising): {"FALSE" if job.image_strength == 1.0 else f"TRUE - {job.image_strength * 100:.0f}%"}<br>'
                f'Settings: Seed: {job.seed} Steps: {job.steps}   Teacache: {job.use_teacache} - MP4 Compression: {job.mp4_crf} - GPU Memory: {job.gpu_memory}<br>'
                f'Scaleing:   GS: {job.gs} - RS: {job.rs} - CFG {job.cfg}'
                f'</span></div>'
            )
        else:
            prompt_cell = f"Error message: {job.error_message}"

        # Add edit button for pending jobs
        edit_button = "✎" if job.status in ["pending", "completed", "failed"] else ""
        top_button = "⏫️"
        up_button = "⬆️"
        down_button = "⬇️"
        bottom_button = "⏬️"
        remove_button = "❌"
        copy_button = "📋"

        data.append([
            img_md,           # Input thumbnail with ID, length, and status
            top_button,
            up_button,
            down_button,
            bottom_button,
            remove_button,
            copy_button,
            edit_button,
            prompt_cell      # Prompt
        ])
    
    # Return DataFrame with container style to force full height
    return gr.DataFrame(
        value=data, 
        visible=True, 
        elem_classes=["gradio-dataframe"],
        elem_id="queue_table"
    )



def cleanup_orphaned_files():

    print ("Clean up any temp files that don't correspond to jobs in the queue")
    try:
        # Get all job files from queue
        job_files = set()
        for job in job_queue:
            if job.image_path:
                job_files.add(job.image_path)
            if job.thumbnail:
                job_files.add(job.thumbnail)
        
        # Get all files in temp directory
        temp_files = set()
        for root, _, files in os.walk(temp_queue_images):
            for file in files:
                temp_files.add(os.path.join(root, file))
        
        # Find orphaned files (in temp but not in queue)
        orphaned_files = temp_files - job_files
        
        # Delete orphaned files
        for file in orphaned_files:
            try:
                os.remove(file)
            except Exception as e:
                alert_print(f"Error deleting file {file}: {str(e)}")
    except Exception as e:
        alert_print(f"Error in cleanup_orphaned_files: {str(e)}")
        traceback.print_exc()


def reset_processing_jobs():
    info_print ("Reset any processing to pending and move them to top of queue")
    global job_queue
    try:
        # First load the queue from JSON
        load_queue()
        
        # Remove completed jobs if keep_completed_job is False
        if not keep_completed_job:
            completed_jobs_to_remove = [
                job for job in job_queue 
                if job.status == "completed" 
                and (
                    (hasattr(job, 'keep_completed_job') and not job.keep_completed_job) or
                    (not hasattr(job, 'keep_completed_job') and not keep_completed_job)
                )
            ]

            # Count jobs that will be removed
            completed_jobs_count = len(completed_jobs_to_remove)

            # Remove the jobs that meet our criteria
            if completed_jobs_count > 0:
                job_queue = [job for job in job_queue if job not in completed_jobs_to_remove]
                debug_print(f"Removed {completed_jobs_count} completed jobs from queue")
        
        # Find all processing jobs and move them to top
        processing_jobs = []
        for job in job_queue:
            if job.status == "processing":
                debug_print(f"Found job {job.job_name} with status {job.status}")
                clean_up_temp_mp4png(job)
                mark_job_pending(job)
                processing_jobs.append(job)
        
        # Remove these jobs from their current positions
        for job in processing_jobs:
            if job in job_queue:
                job_queue.remove(job)
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(processing_jobs):
            job_queue.insert(0, job)
            debug_print(f"marked previously aborted job as pending and Moved job {job.job_name} to top of queue")
        
        # Find all failed jobs and move them to top
        failed_jobs = []
        for job in job_queue:
            if job.status == "failed":
                debug_print(f"Found job {job.job_name} with status {job.status}")
                mark_job_failed(job)
                failed_jobs.append(job)
        
        # Remove these jobs from their current positions
        for job in failed_jobs:
            if job in job_queue:
                job_queue.remove(job)
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(failed_jobs):
            job_queue.insert(0, job)
            debug_print(f"marked previously aborted job as pending and Moved job {job.job_name} to top of queue")
        
        save_queue()
        debug_print(f"{len(processing_jobs)} aborted jobs found and moved to the top as pending")
        debug_print(f"{len(failed_jobs)} failed jobs found and moved to the top as pending")
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
    except Exception as e:
        alert_print(f"Error resetting processing jobs: {str(e)}")

# Quick prompts management functions
def get_default_prompt():
    try:
        if quick_prompts and len(quick_prompts) > 0:
            # Get the first quick prompt
            prompt_data = quick_prompts[0]
            
            # Return values in the correct order, using Config defaults for missing keys
            return (
                prompt_data.get('prompt', Config.DEFAULT_PROMPT),
                prompt_data.get('n_prompt', Config.DEFAULT_N_PROMPT),
                prompt_data.get('lora_model', Config.DEFAULT_LORA_MODEL),
                prompt_data.get('lora_weight', Config.DEFAULT_LORA_WEIGHT),
                prompt_data.get('video_length', Config.DEFAULT_VIDEO_LENGTH),
                prompt_data.get('job_name', Config.DEFAULT_JOB_NAME),
                prompt_data.get('use_teacache', Config.DEFAULT_USE_TEACACHE),
                prompt_data.get('seed', Config.DEFAULT_SEED),
                prompt_data.get('steps', Config.DEFAULT_STEPS),
                prompt_data.get('cfg', Config.DEFAULT_CFG),
                prompt_data.get('gs', Config.DEFAULT_GS),
                prompt_data.get('rs', Config.DEFAULT_RS),
                prompt_data.get('image_strength', Config.DEFAULT_IMAGE_STRENGTH),
                prompt_data.get('mp4_crf', Config.DEFAULT_MP4_CRF),
                prompt_data.get('gpu_memory', Config.DEFAULT_GPU_MEMORY),
                prompt_data.get('outputs_folder', Config.OUTPUTS_FOLDER),
                prompt_data.get('job_history_folder', Config.JOB_HISTORY_FOLDER),
                prompt_data.get('keep_completed_job', Config.KEEP_COMPLETED_JOB),
                prompt_data.get('keep_temp_png', Config.DEFAULT_KEEP_TEMP_PNG)
            )
        return Config.get_default_prompt_tuple()
    except Exception as e:
        alert_print(f"Error getting default prompt: {str(e)}")
        return Config.get_default_prompt_tuple()

def save_quick_prompt(prompt_text, n_prompt_text_value, lora_model_value, lora_weight_value, video_length_value, job_name_value, use_teacache_value, seed_value, steps_value, cfg_value, gs_value, rs_value, image_strength_value,
 mp4_crf_value, gpu_memory_value, outputs_folder_value, job_history_folder_value, keep_completed_job_value,
  keep_temp_png_value):
    """Save the current prompt and settings to the quick prompts list"""
    global quick_prompts
    
    # If job name is blank, use first 20 chars of prompt
    if not job_name_value or job_name_value.strip() == "":
        job_name_value = prompt_text[:20].strip()
    
    # Create the current structure
    current_structure = {
        'prompt': prompt_text,
        'n_prompt': n_prompt_text_value,
        'lora_model': lora_model_value,
        'lora_weight': lora_weight_value,
        'video_length': video_length_value,
        'job_name': job_name_value,
        'use_teacache': use_teacache_value,
        'seed': seed_value,
        'steps': steps_value,
        'cfg': cfg_value,
        'gs': gs_value,
        'rs': rs_value,
        'image_strength': image_strength_value,
        'mp4_crf': mp4_crf_value,
        'gpu_memory': gpu_memory_value,
        'outputs_folder': outputs_folder_value,
        'job_history_folder': job_history_folder_value,
        'keep_completed_job': keep_completed_job_value,
        'keep_temp_png': keep_temp_png_value
    }
    
    # Check if a prompt with this job name already exists
    existing_index = next((i for i, item in enumerate(quick_prompts) if item['job_name'] == job_name_value), None)
    
    if existing_index is not None:
        # Update existing prompt
        quick_prompts[existing_index] = current_structure
    else:
        # Add new prompt
        quick_prompts.append(current_structure)
    
    # Save to file
    with open(QUICK_LIST_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)
    
    # Update quick list choices
    quick_list_choices = [item['job_name'] for item in quick_prompts]    
    current_choice = f"{job_name_value}"
    
    # Return updated values for UI components in order matching the UI layout
    return [
        prompt_text,  # Keep the current prompt value
        n_prompt_text_value,  # Keep the current n_prompt value
        gr.update(choices=quick_list_choices, value=current_choice),  # Update the quick list dropdown
        lora_model_value,  # Keep the current lora model
        lora_weight_value,  # Keep the current lora weight
        video_length_value,  # Keep the current video length
        job_name_value,  # Keep the current job name
        outputs_folder_value,  # Keep the current outputs folder
        job_history_folder_value,  # Keep the current job history folder
        use_teacache_value,  # Keep the current teacache setting
        seed_value,  # Keep the current seed
        steps_value,  # Keep the current steps
        cfg_value,  # Keep the current cfg
        gs_value,  # Keep the current gs
        rs_value,  # Keep the current rs
        image_strength_value,  # Keep the current image strength
        mp4_crf_value,  # Keep the current mp4 crf
        gpu_memory_value,  # Keep the current gpu memory
        keep_completed_job_value,  # Keep the current keep completed job setting
        keep_temp_png_value  # Keep the current keep temp png setting
    ]

def delete_quick_prompt(prompt_text):
    """Delete the selected prompt from the quick prompts list"""
    global quick_prompts
    
    # Extract job name from the display format (job_name - prompt_preview...)
    job_name = prompt_text.split(" - ")[0] if " - " in prompt_text else prompt_text
    
    # Find and remove the prompt with matching job name
    quick_prompts = [item for item in quick_prompts if item['job_name'] != job_name]
    
    # Save to file
    with open(QUICK_LIST_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)
    
    # Update quick list choices
    quick_list_choices = [item['job_name'] for item in quick_prompts]    
    # Return updated values for UI components in order
    return [
        Config.DEFAULT_PROMPT,  # prompt
        Config.DEFAULT_N_PROMPT,  # n_prompt
        gr.update(choices=quick_list_choices, value=None),  # quick_list
        Config.DEFAULT_LORA_MODEL,  # lora_model
        Config.DEFAULT_LORA_WEIGHT,  # lora_weight
        Config.DEFAULT_VIDEO_LENGTH,  # video_length
        Config.DEFAULT_JOB_NAME,  # job_name
        Config.DEFAULT_USE_TEACACHE,  # use_teacache
        Config.DEFAULT_SEED,  # seed
        Config.DEFAULT_STEPS,  # steps
        Config.DEFAULT_CFG,  # cfg
        Config.DEFAULT_GS,  # gs
        Config.DEFAULT_RS,  # rs
        Config.DEFAULT_IMAGE_STRENGTH,  # image_strength
        Config.DEFAULT_MP4_CRF,  # mp4_crf
        Config.DEFAULT_GPU_MEMORY,  # gpu_memory
        Config.OUTPUTS_FOLDER,  # outputs_folder
        Config.JOB_HISTORY_FOLDER,  # job_history_folder
        Config.KEEP_COMPLETED_JOB,  # keep_completed_job
        Config.DEFAULT_KEEP_TEMP_PNG  # keep_temp_png
    ]

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

def convert_model_path(path):
    """Convert from directory name format to huggingface format"""
    # Remove any default model suffixes if present
    if " - USERS DEFAULT MODEL" in path:
        path = path.replace(" - USERS DEFAULT MODEL", "")
    if " - ORIGINAL DEFAULT MODEL" in path:
        path = path.replace(" - ORIGINAL DEFAULT MODEL", "")
    
    # First check if this is a display name with our prefix
    if path.startswith('DOWNLOADED-MODEL-'):
        # Get the actual folder name from our mapping
        if hasattr(Config, 'model_name_mapping') and path in Config.model_name_mapping:
            path = Config.model_name_mapping[path]
    
    # Then do the normal conversion
    if path.startswith('models--'):
        # Convert from "models--org--model" to "org/model"
        parts = path.split('--')
        if len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
    return path

# Update model loading code
text_encoder = LlamaModel.from_pretrained(convert_model_path(Config.DEFAULT_TEXT_ENCODER), subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained(convert_model_path(Config.DEFAULT_TEXT_ENCODER_2), subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained(convert_model_path(Config.DEFAULT_TOKENIZER), subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained(convert_model_path(Config.DEFAULT_TOKENIZER_2), subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained(convert_model_path(Config.DEFAULT_VAE), subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained(convert_model_path(Config.DEFAULT_FEATURE_EXTRACTOR), subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained(convert_model_path(Config.DEFAULT_IMAGE_ENCODER), subfolder='image_encoder', torch_dtype=torch.float16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(convert_model_path(Config.DEFAULT_TRANSFORMER), torch_dtype=torch.bfloat16).cpu()


vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()
# image2image_model.eval()  # Flux pipeline doesn't have eval() method

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)   
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)   
    vae.to(gpu)
    transformer.to(gpu)


def cleanup_temp_i2i(job):
    # If image_strength is e.g. 0.75, turn it into 75
    job_name = job.job_name
    image_strength = job.image_strength
    strength_pct = int(image_strength * 100)

    files = [
        f"new_input_image_{job_name}.png",
        f"1secondtempvideo_{job_name}.mp4",
        f"{strength_pct}%_noise_added_{job_name}.png",
        f"original_input_image_{job_name}.png"
    ]

    for fname in files:
        path = os.path.join(temp_queue_images, fname)
        try:
            os.unlink(path)  # same as os.remove
        except FileNotFoundError:
            # file was already gone; ignore
            pass


def clean_up_temp_mp4png(job):
    job_name = job.job_name

    #Deletes all '<job_name>_<n>.mp4' in outputs_folder except the one with the largest n. Also deletes the '<job_name>.png' file if keep_temp_png is false.
    

    # Delete the PNG file
    png_path = os.path.join(job.job_history_folder if hasattr(job, 'job_history_folder') else Config.JOB_HISTORY_FOLDER, f'{job_name}.png')
    try:
        if os.path.exists(png_path) and not job.keep_temp_png:
            os.remove(png_path)
            debug_print(f"Deleted job_history PNG file for {job_name}")
    except OSError as e:
        alert_print(f"Failed to delete job_history PNG file for {job_name}: {e}")

    #rename the job history PNG file with prefix and suffix like we do with the output MP4 files
    try:
        if os.path.exists(png_path) and job.keep_temp_png:           
            # Get the current timestamp if enabled
            timestamp_prefix = f"{generate_timestamp().rsplit('_', 2)[0]}_" if prefix_timestamp else ""
            # Get source name prefix if enabled 
            source_prefix = f"{job.source_name}_" if prefix_source_image_name and hasattr(job, 'source_name') else ""
            
            # Remove hex ID suffix if present
            base_name = re.sub(r'_[0-9a-f]{8}$', '', job_name)
            
            # Create the new filename
            # Build filename based on enabled prefixes
            parts = []
            if prefix_timestamp:
                parts.append(timestamp_prefix)
                debug_print(f"added timestamp prefix to new_filename")
            if prefix_source_image_name and hasattr(job, 'source_name'):
                parts.append(source_prefix)
                debug_print(f"added source image name prefix to new_filename")
            if hasattr(job, 'prefix') and job.prefix:
                parts.append(f"{job.prefix}_")
            # Only remove hex suffix if both remove_hexid_suffix and PREFIX_TIMESTAMP are true
            if remove_hexid_suffix and prefix_timestamp:
                parts.append(base_name)  # base_name already has suffix removed
            else:
                parts.append(job_name)  # Use full job_name with suffix
            if hasattr(job, 'suffix') and job.suffix:
                parts.append(f"_{job.suffix}")
            parts.append(".png")
            new_filename = "".join(parts)
            new_path = os.path.join(job.job_history_folder if hasattr(job, 'job_history_folder') else Config.JOB_HISTORY_FOLDER, new_filename)
            os.rename(png_path, new_path)
            debug_print(f"Successfully renamed job history PNG file to {new_filename}")
    except OSError as e:
        alert_print(f"Failed to rename job history PNG file for {job_name}: {e}")

   
    # grab the trailing number from mp4
    pattern = re.compile(rf'^{re.escape(job_name)}_(\d+)\.mp4$')
    candidates = []
    # scan directory
    for fname in os.listdir(Config.OUTPUTS_FOLDER): 
        m = pattern.match(fname)
        if m:
            frame_count = int(m.group(1))
            candidates.append((frame_count, fname))

    if not candidates:
        return  # nothing to clean up

    # find the highest frame‐count
    highest_count, highest_fname = max(candidates, key=lambda x: x[0])

    # delete all but the highest
    for count, fname in candidates:
        if count != highest_count and (fname.endswith('.mp4')):
            path = os.path.join(Config.OUTPUTS_FOLDER, fname)
            try:
                os.remove(path)
            except OSError as e:
                alert_print(f"Failed to delete unneeded mp4 chunk {fname}: {e}")
        debug_print(f"deleted unneeded mp4 chunks for {job_name}")

    # Rename the remaining MP4 to {job_name}.mp4
    if highest_fname:
        old_path = os.path.join(Config.OUTPUTS_FOLDER, highest_fname)
        new_path = os.path.join(Config.OUTPUTS_FOLDER, f"{job_name}.mp4")
        try:
            if os.path.exists(new_path):
                debug_print(f"Overwriting existing file {job_name}.mp4")
                os.remove(new_path)  # Remove existing file if it exists
            debug_print(f"Renaming {highest_fname} to {job_name}")
            os.rename(old_path, new_path)
            debug_print(f"Successfully saved {new_path}")
        except OSError as e:
            alert_print(f"Failed to rename {highest_fname} to {job_name}.mp4: {e}")



def add_meta_data_to_mp4(worker_job, worker_seed, output_filename):

    # … after rendering your MP4 to final_mp4 add metadata …
    try:
        # 1) make sure ffmpeg is available
        subprocess.run(['ffmpeg','-version'], check=True, stdout=subprocess.DEVNULL)

        final_mp4 = output_filename
        pass1_mp4 = f"{output_filename}_pass1.mp4"
        pass2_mp4 = f"{output_filename}_pass2.mp4"


        
        
            
        fields = [
            ("prompt", worker_job.prompt),
            ("n_prompt", worker_job.n_prompt),
            ("lora_model", str(worker_job.lora_model)),
            ("lora_weight", str(worker_job.lora_weight)),
            ("video_length", str(worker_job.video_length)),
            ("job_name", worker_job.job_name),
            ("use_teacache", str(worker_job.use_teacache)),
            ("seed", str(worker_seed)),
            ("steps", str(worker_job.steps)),
            ("cfg", str(worker_job.cfg)),
            ("gs", str(worker_job.gs)),
            ("rs", str(worker_job.rs)),
            ("image_strength", str(worker_job.image_strength)),
            ("mp4_crf", str(worker_job.mp4_crf)),
            ("gpu_memory", str(worker_job.gpu_memory)),
            ("source_name", str(getattr(worker_job, 'source_name', '')))
        ]        
        
        comment_parts = [
            f"{value[0]}: {value[1]}" 
            for value in fields
        ]
        comment_str = "; ".join(comment_parts)

        # --- PASS 1: write uppercase Comments for WMP ---
        cmd1 = [
            "ffmpeg", "-i", final_mp4,
            "-c", "copy", 
            "-movflags", "+use_metadata_tags",
            "-metadata", f"Comments={comment_str}",
            pass1_mp4
        ]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)

        # --- PASS 2: pull in that Comments + add each key=value ---
        meta_args = []
        for field in fields:
            meta_args += ['-metadata', f"{field[0]}={field[1]}"]

        cmd2 = [
            "ffmpeg", "-i", pass1_mp4,
            "-map_metadata", "0",
            "-c", "copy",
            "-movflags", "+use_metadata_tags",
        ] + meta_args + [pass2_mp4]

        subprocess.run(cmd2, check=True, capture_output=True, text=True)

        # overwrite original and clean up
        os.replace(pass2_mp4, final_mp4)
        os.remove(pass1_mp4)
        debug_print(f"Successfully added Metadata to {final_mp4}")

    except subprocess.CalledProcessError as e:
        alert_print(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        alert_print(f"Metadata embedding failed: {e}")


def move_mp4_to_custom_output_folder(job):
    # Rename the remaining MP4 to {job_name}.mp4
    old_path = os.path.join(Config.OUTPUTS_FOLDER, f"{job.job_name}.mp4")
    if hasattr(job, 'outputs_folder') and job.outputs_folder != Config.OUTPUTS_FOLDER:
        try:
            os.makedirs(job.outputs_folder, exist_ok=True)

            # Start with base job name
            output_filename = job.job_name
            
            # Remove hexid suffix only if both timestamp and remove_hexid are enabled
            if prefix_timestamp and remove_hexid_suffix:
                output_filename = output_filename[:-9]
                debug_print(f"removed hex id suffix from {output_filename}")

            # Add source image name prefix if configured
            if prefix_source_image_name and hasattr(job, 'source_name'):
                source_prefix = os.path.splitext(job.source_name)[0]
                output_filename = f"{source_prefix}_{output_filename}"                
                debug_print(f"added source image name prefix to {output_filename}")

            # Add timestamp prefix if configured 
            if prefix_timestamp:
                timestamp = generate_timestamp().rsplit('_', 2)[0]  # Remove milliseconds portions
                output_filename = f"{timestamp}_{output_filename}"
                debug_print(f"added timestamp prefix to {output_filename}")
            custom_path = os.path.join(job.outputs_folder, f"{output_filename}.mp4")
            
            debug_print(f"Moving video {output_filename}.mp4 to user output folder: {custom_path}")
            
            # If a file already exists at the destination, remove it
            if os.path.exists(custom_path):
                os.remove(custom_path)
                
            # Move the file to the custom location
            shutil.move(old_path, custom_path)
            debug_print(f"Successfully moved {output_filename}.mp4 to user output folder: {custom_path}")
        except Exception as e:
            alert_print(f"Failed to move video to custom output folder {job.outputs_folder}: {e}")
            alert_print("Video will remain in default output folder")
            traceback.print_exc()

def count_jobs_by_status():
    """Count jobs by their status, ignoring 'processing' and unknown statuses"""
    global current_counts
    counts = {
        "completed": 0,
        "pending": 0,
        "failed": 0,
        "all": len(job_queue)
    }
    for job in job_queue:
        if job.status in ("completed", "pending", "failed"):
            counts[job.status] += 1
    current_counts = counts  # Update the global variable
    return counts



def mark_job_processing(job):
    #Mark a job as processing and update its thumbnail
    job.status = "processing"
    
    # Delete existing thumbnail if it exists
    if job.thumbnail and os.path.exists(job.thumbnail):
        os.remove(job.thumbnail)

    job.thumbnail = create_thumbnail(job, status_change=True)

    
    # Move job to top of queue
    if job in job_queue:
        job_queue.remove(job)
        job_queue.insert(0, job)
        
    save_queue()
    return update_queue_table(), update_queue_display()


def mark_job_completed(completed_job):
    #Mark a job as completed and update its thumbnail
    completed_job.status = "completed"
    # Move completed_job to the top of completed jobs
    if completed_job in job_queue:
        job_queue.remove(completed_job)
        # Find the first completed job
        insert_index = len(job_queue)
        for i, existing_job in enumerate(job_queue):
            if existing_job.status == "completed":
                insert_index = i
                break
        job_queue.insert(insert_index, completed_job)
        save_queue()

    if completed_job.thumbnail and os.path.exists(completed_job.thumbnail):
        os.remove(completed_job.thumbnail)
    mp4_path = os.path.join(Config.OUTPUTS_FOLDER, f"{completed_job.job_name}.mp4")
    extract_thumb_from_processing_mp4(completed_job, mp4_path, job_percentage = 100)
    completed_job.thumbnail = thumbnail
    if completed_job.image_path != "text2video":
        completed_job.thumbnail = create_thumbnail(completed_job, status_change=True)
    save_queue()
    return update_queue_table(), update_queue_display()

def mark_job_failed(job):
    #Mark a job as completed and update its thumbnail
    try:
        job.status = "failed"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new thumbnail with failed status
        if job.image_path:
            job.thumbnail = create_thumbnail(job, status_change=True)
        
        # Move job to top of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.insert(0, job)
            
        save_queue()
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error marking job as failed: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

def mark_job_pending(job):
    #Mark a job as pending and update its thumbnail
    try:
        job.status = "pending"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new clean thumbnail
        if job.image_path:
            job.thumbnail = create_thumbnail(job, status_change=True)
            
        save_queue()
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error marking job as pending: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()


@torch.no_grad()
def make_img2img(worker_input_image, next_job, worker_prompt, worker_cfg, worker_n_prompt, 
    worker_image_strength, worker_job_name, worker_seed, worker_use_teacache, worker_gpu_memory, worker_steps, 
    worker_latent_window_size, worker_gs, worker_rs, worker_mp4_crf):
    #High-quality image-to-image generator with denoising capabilities using the same models as worker function
    #total_latent_sections = 1
    total_latent_sections = (1 * 30) / (worker_latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    noising_value = round(1 - worker_image_strength, 2)  # e.g. if worker_image_strength is 0.7, noising_value would be 0.3
############  New image to image code below  #######################################
    worker_gs_factor = worker_gs + (noising_value * 10)

    worker_prompt = worker_prompt + ", high quality video, focused on subject"
    worker_n_prompt = worker_n_prompt + ", static, noise, cartoon, grainy, unrealistic"

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )





        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(worker_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if worker_n_prompt is None or worker_n_prompt == "":
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(worker_n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        input_image_np = np.array(worker_input_image)
        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(temp_queue_images, f'original_input_image_{worker_job_name}.png'))
        
        
 
        # Load the original image and add noise
        original_image = Image.open(os.path.join(temp_queue_images, f'original_input_image_{worker_job_name}.png'))
        original_array = np.array(original_image).astype(np.float32)
        
        # Generate random noise with same shape as image
        noise = np.random.normal(0, 1, original_array.shape).astype(np.float32)
        
        # Mix original image and noise according to noising_value (1 - worker_image_strength)
        noised_array = original_array * worker_image_strength + noise * (1 - worker_image_strength) * 255
        noised_array = np.clip(noised_array, 0, 255).astype(np.uint8)
        
        # Save the noised image
        Image.fromarray(noised_array).save(os.path.join(temp_queue_images, f'{noising_value*100}%_noise_added_{worker_job_name}.png'))

        # Use the noised array directly instead of loading the saved image
        input_image_np = noised_array

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]


        # VAE encoding


        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)



        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
 

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        state_dict = transformer.state_dict()

        # Load LoRA if set
        if next_job.lora_model and next_job.lora_model != "None":
            try:
                lora_model = os.path.join(lora_path, next_job.lora_model)
                debug_print(f"Loading hunyaun videoLoRA from {lora_model}")
                state_dict = load_lora(state_dict, lora_model, next_job.lora_weight, device=gpu)
                gc.collect()
                debug_print("LoRA loaded successfully")
            except Exception as e:
                alert_print(f"Error loading LoRA: {str(e)}")
                traceback.print_exc()
                raise
        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        rnd = torch.Generator("cpu").manual_seed(worker_seed)

        # Initialize history latents with improved structure
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=worker_gpu_memory)
            if worker_use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=worker_steps+10)
            else:
                transformer.initialize_teacache(enable_teacache=False)
            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')


            
            indices = torch.arange(0, sum([1, 16, 2, 1, worker_latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, worker_latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
            
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames= worker_latent_window_size * 4 - 3,
                real_guidance_scale=worker_cfg,
                distilled_guidance_scale=worker_gs_factor,  #increase the GS value by a factor of the image strength
                guidance_rescale=1,#worker_rs,
                # shift=3.0,
                num_inference_steps=worker_steps,  
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = worker_latent_window_size * 2
                overlapped_frames = worker_latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)


            if not high_vram:
                unload_complete_models()

            temp_1secondtempvideo = os.path.join(temp_queue_images, f'1secondtempvideo_{worker_job_name}.mp4')
            save_bcthw_as_mp4(history_pixels, temp_1secondtempvideo, fps=30, crf=worker_mp4_crf)
            
            # Extract last frame from the generated MP4
            cap = cv2.VideoCapture(temp_1secondtempvideo)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, output = cap.read()
            cap.release()

#################End of new image to image code#########################################################
############ make sure whatever image to image code pruduces the image as the variable output  #########
            if ret:
                # Convert BGR to RGB
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                # Save the last frame as PNG
                new_input_image = os.path.join(temp_queue_images, f'new_input_image_{worker_job_name}.png')
                Image.fromarray(output_rgb).save(new_input_image)

            temp_video_path = os.path.join(temp_queue_images, f'1secondtempvideo_{worker_job_name}.mp4')
            denoised_result = np.array(new_input_image)
            starting_image = create_still_frame_video(next_job, temp_video_path, denoised_result)


        return new_input_image, starting_image
        
    except Exception as e:
        alert_print(f"Error in denoising: {str(e)}")
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            ) 
        new_input_image = "failed"
        starting_image = ""
        return new_input_image, starting_image


            # # # Initialize RealESRGAN if available
            # # dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # # rrgan = None
            # # if ensure_package('realesrgan'):
                # # from realesrgan import RealESRGAN
                # # try:
                    # # rrgan = RealESRGAN(dev, scale=2)
                    # # rrgan.load_weights('RealESRGAN_x2plus')
                    # # print("DEBUG: RealESRGAN ready for enhancement")
                # # except Exception as e:
                    # # print(f"WARNING: RealESRGAN init/load error ({e}); skipping super-resolution.")
                    # # rrgan = None
            # # else:
                # # rrgan = None

            # # # ... inside your processing loop ...
            # if ret:
                # # 1) Convert BGR -> RGB and create PIL image
                # output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                # pil_img = Image.fromarray(output_rgb)

                # # 2) Save original 'before' image
                # before_path = os.path.join(temp_queue_images, f"before_image_{worker_job_name}.png")
                # pil_img.save(before_path)
                # print(f"DEBUG: Saved before-image: {before_path}")

                # orig_size = pil_img.size

                # # SECTION 3: OpenCV DENOISE PASS
                # print("DEBUG [Section 3]: Starting OpenCV denoising...")
                # try:
                    # proc_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    # denoised_np = cv2.fastNlMeansDenoisingColored(
                        # proc_np, None,
                        # h=30, hColor=30,
                        # templateWindowSize=7,
                        # searchWindowSize=21
                    # )
                    # denoised_img = Image.fromarray(cv2.cvtColor(denoised_np, cv2.COLOR_BGR2RGB))
                    # print("DEBUG [Section 3]: OpenCV denoising applied successfully")
                # except Exception as e:
                    # print(f"ERROR [Section 3]: OpenCV denoise failed ({e})")
                    # denoised_img = pil_img
                # # ─── Section 4: Super-resolution enhancement ───
                # try:
                    # # dynamic import only when needed
                    # import importlib
                    # rr_module = importlib.import_module("realesrgan")
                    # RealESRGAN = getattr(rr_module, "RealESRGAN")
                    # print("DEBUG: RealESRGAN imported successfully")

                    # # initialize and load weights
                    # rrgan = RealESRGAN(device="cuda", scale=2)
                    # rrgan.load_weights("RealESRGAN_x2plus")
                    # print("DEBUG: RealESRGAN weights loaded")

                    # # run enhancement on the denoised or original PIL image
                    # enhanced_large = rrgan.enhance(pil_img)
                    # enhanced_large.save(os.path.join(temp_queue_images, f"enhanced_large_{worker_job_name}.png"))
                    # print("DEBUG: Saved enhanced_large file")

                    # # resize back to original size and overwrite new_input_image
                    # restored = enhanced_large.resize(orig_size, resample=Image.LANCZOS)
                    # restored.save(new_input_image)
                    # success_print(f"new enhanced image saved: {new_input_image}")

                # except ImportError:
                    # print("DEBUG: realesrgan not installed; skipping super-resolution")
                # except TypeError as e:
                    # print(f"DEBUG: RealESRGAN registry error, skipping super-resolution ({e})")
                # except Exception as e:
                    # print(f"DEBUG: Unexpected error during RealESRGAN enhancement; skipping ({e})")


            # temp_video_path = os.path.join(temp_queue_images, f'1secondtempvideo_{worker_job_name}.mp4')
            # denoised_result = np.array(new_input_image)
            # starting_image = create_still_frame_video(next_job, temp_video_path, denoised_result)


        # return new_input_image, starting_image
        
    # except Exception as e:
        # alert_print(f"Error in denoising: {str(e)}")
        # if not high_vram:
            # unload_complete_models(
                # text_encoder, text_encoder_2, image_encoder, vae, transformer
            # ) 
        # new_input_image = "failed"
        # starting_image = ""
        # return new_input_image, starting_image



@torch.no_grad()
def worker(worker_job):
    global transformer, worker_input_image
    """Worker function to process a job"""
    global stream
    # Create job output and history folders if they don't exist
    try:
        if not os.path.exists(worker_job.job_history_folder):
            debug_print(f"Creating job history folder: {worker_job.job_history_folder}") 
            os.makedirs(worker_job.job_history_folder, exist_ok=True)
    except Exception as e:
        alert_print(f"Error creating job folders: {str(e)}")
        traceback.print_exc()
        raise

    debug_print(f"Starting worker for job {worker_job.job_name}")
    
    # prepare input_image & Handle text-to-video case
    if worker_job.image_path == "text2video":
        worker_input_image = None
    else:
        try:
            worker_input_image = np.array(Image.open(worker_job.image_path))
        except Exception as e:
            alert_print(f"ERROR loading image: {str(e)}")
            traceback.print_exc()
            raise

    # Extract all values from worker_job at the start
    # Core inputs
    worker_prompt = worker_job.prompt if hasattr(worker_job, 'prompt') else None
    worker_n_prompt = worker_job.n_prompt if hasattr(worker_job, 'n_prompt') else None
    worker_lora_model = worker_job.lora_model if hasattr(worker_job, 'lora_model') else None
    worker_lora_weight = worker_job.lora_weight if hasattr(worker_job, 'lora_weight') else None
    
    # Video settings
    worker_video_length = (worker_job.video_length) if hasattr(worker_job, 'video_length') else (Config.DEFAULT_VIDEO_LENGTH)
    
    # Job settings
    worker_job_name = worker_job.job_name
    worker_outputs_folder = worker_job.outputs_folder if hasattr(worker_job, 'outputs_folder') else Config.OUTPUTS_FOLDER
    worker_job_history_folder = worker_job.job_history_folder if hasattr(worker_job, 'job_history_folder') else Config.JOB_HISTORY_FOLDER
    
    # Processing settings
    worker_use_teacache = worker_job.use_teacache if hasattr(worker_job, 'use_teacache') else Config.DEFAULT_USE_TEACACHE
    checked_seed = worker_job.seed if hasattr(worker_job, 'seed') else Config.DEFAULT_SEED
    worker_seed = checked_seed if checked_seed != -1 else random.randint(0, 2**32 - 1)
    worker_latent_window_size = worker_job.latent_window_size if hasattr(worker_job, 'latent_window_size') else Config.DEFAULT_LATENT_WINDOW_SIZE
    worker_steps = worker_job.steps if hasattr(worker_job, 'steps') else Config.DEFAULT_STEPS
    worker_cfg = worker_job.cfg if hasattr(worker_job, 'cfg') else Config.DEFAULT_CFG
    worker_gs = worker_job.gs if hasattr(worker_job, 'gs') else Config.DEFAULT_GS
    worker_rs = worker_job.rs if hasattr(worker_job, 'rs') else Config.DEFAULT_RS
    worker_image_strength = worker_job.image_strength if hasattr(worker_job, 'image_strength') else 1.0
    worker_mp4_crf = worker_job.mp4_crf if hasattr(worker_job, 'mp4_crf') else Config.DEFAULT_MP4_CRF
    worker_gpu_memory = worker_job.gpu_memory if hasattr(worker_job, 'gpu_memory') else Config.DEFAULT_GPU_MEMORY
    
    # File retention settings
    worker_keep_temp_png = worker_job.keep_temp_png if hasattr(worker_job, 'keep_temp_png') else Config.DEFAULT_KEEP_TEMP_PNG
    worker_keep_completed_job = worker_job.keep_completed_job if hasattr(worker_job, 'keep_completed_job') else Config.KEEP_COMPLETED_JOB

    debug_print(f"Job {worker_job_name} initial seed value: {checked_seed}")
    debug_print(f"Generated new random seed for job {worker_job_name}: {worker_seed}") if checked_seed == -1 else None

    # Save the input image with metadata
    metadata = PngInfo()
    fields = [
        ("prompt", worker_prompt),
        ("n_prompt", worker_n_prompt),
        ("lora_model", str(worker_lora_model)),
        ("lora_weight", str(worker_lora_weight)),
        ("video_length", str(worker_video_length)),
        ("job_name", worker_job_name),
        ("use_teacache", str(worker_use_teacache)),
        ("seed", str(worker_seed)),
        ("steps", str(worker_steps)),
        ("cfg", str(worker_cfg)),
        ("gs", str(worker_gs)),
        ("rs", str(worker_rs)),
        ("image_strength", str(worker_image_strength)),
        ("mp4_crf", str(worker_mp4_crf)),
        ("gpu_memory", str(worker_gpu_memory)),
        ("source_name", str(getattr(worker_job, 'source_name', '')))
    ]
    for key, value in fields:
        metadata.add_text(key, value)

    job_percentage = 0

    # Pre-process input image if image_strength < 1.0


    # total_latent_sections = (worker_video_length * 30) / (worker_latent_window_size * 4)
    # total_latent_sections = int(max(round(total_latent_sections), 1))
    # target total frames minus 1 seed frame:
    target_frames = ((worker_video_length * 30) - 1)
    total_latent_sections = max(math.ceil(target_frames / (worker_latent_window_size * 4)), 1)

    
    
    # Convert RGBA images to RGB if needed
    if worker_input_image is not None:
        if worker_input_image.shape[-1] == 4:  # Check if image has alpha channel
            debug_print("Converting RGBA input image to RGB")
            # Convert numpy array from RGBA to RGB by dropping alpha channel
            worker_input_image = worker_input_image[..., :3]

    preview_image = None #important to set this to none here so if not a image to image job it will use none on the first round of preview images
    if worker_input_image is not None and worker_image_strength < 1.0:
        debug_print(f"Pre-processing input image with image_strength {worker_image_strength}")
        
        # send to image to image function, Get the denoised image and update worker_input_image
        new_input_image, starting_image = make_img2img(worker_input_image, worker_job, worker_prompt, worker_cfg, worker_n_prompt, worker_image_strength, worker_job_name, worker_seed, worker_use_teacache, worker_gpu_memory, worker_steps, worker_latent_window_size, worker_gs, worker_rs, worker_mp4_crf)
        
        worker_prompt = worker_prompt + ", high quality video, focused on subject"
        worker_n_prompt = worker_n_prompt + ", static, noise, cartoon, grainy, unrealistic"

        
        if new_input_image == "failed":
            e = "Error creating Image2Image"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worker_job.error_message = f"[{timestamp}] - {str(e)}"
            save_queue()
            alert_print(f"job updated to include error message: {e}")
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
            stream.output_queue.push(('failed', (job_percentage, worker_job)))
            return


        else:
            worker_input_image = np.array(Image.open(new_input_image))
            stream.output_queue.push(('file', (starting_image, 0)))

    job_type = "Text 2 Video" if worker_job.image_path == "text2video" else ("Image to Image to Video" if worker_image_strength < 1.0 else "Image to Video")
    job_desc = f"""<br>
Job Name:      {worker_job_name} Job Type: {job_type} {"" if worker_image_strength == 1.0 else f"Initial Image Strength: {worker_image_strength * 100:.0f}%"}<br>
Source Image:  {worker_job.source_name}<br>
Output folder: {worker_outputs_folder}<br>
Prompt:        {worker_prompt} - Negative Prompt: {worker_n_prompt}<br>
Lora Model:    {worker_lora_model}{f" - Lora Weight: {worker_lora_weight}" if worker_lora_model != "None" else ""}<br>
Settings:      Seed: {worker_seed} - Steps: {worker_steps} - Teacache: {worker_use_teacache} - MP4 Compression: {worker_mp4_crf}<br>
CFG Scaleing:  GS: {worker_gs} - RS: {worker_rs} - CFG: {worker_cfg}<br>
"""
    try:
        job_percentage = 0
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        if worker_lora_model and worker_lora_model != "None":
            
            state_dict = transformer.state_dict()
            try:
                lora_model = os.path.join(lora_path, worker_lora_model)
                info_print(f"Loading LoRA from {lora_model}")

                state_dict = load_lora(state_dict, lora_model, worker_lora_weight, device=gpu)
                gc.collect()

                success_print("LoRA loaded successfully")
            except Exception as e:
                alert_print(f"Error loading LoRA: {str(e)}")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                worker_job.error_message = f"[{timestamp}] - {str(e)}"
                save_queue()
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                stream.output_queue.push(('failed', (job_percentage, worker_job)))
                raise


        
        # Text encoding
        stream.output_queue.push(('progress', (preview_image, "Text encoding...", make_progress_bar_html(0, "Step Progress"), job_desc)))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(worker_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # if worker_cfg == 1:
        if worker_n_prompt is None or worker_n_prompt == "":
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(worker_n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)


        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            
        # Processing input image
        stream.output_queue.push(('progress', (preview_image, "Image processing...", make_progress_bar_html(0, "Step Progress"), job_desc)))

        # Handle text-to-video case
        if worker_input_image is None:
            # Create a blank image for text-to-video with default resolution
            default_resolution = 640  # Default resolution for text-to-video
            input_image_np = np.zeros((default_resolution, default_resolution, 3), dtype=np.uint8)
            height = width = default_resolution
        else:
            # Handle image-to-video case
            input_image_np = np.array(worker_input_image)
            H, W, C = input_image_np.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(worker_job_history_folder, f'{worker_job_name}.png'), pnginfo=metadata)
            debug_print(f"saved input image to {worker_job_history_folder}/{worker_job_name}.png")
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (preview_image, "VAE encoding...", make_progress_bar_html(0, "Step Progress"), job_desc)))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)
        # CLIP Vision
        stream.output_queue.push(('progress', (preview_image, "CLIP Vision encoding...", make_progress_bar_html(0, "Step Progress"), job_desc)))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (preview_image, "Start sampling...", make_progress_bar_html(0, "Step Progress"), job_desc)))

        rnd = torch.Generator("cpu").manual_seed(worker_seed)

        # Initialize history latents with improved structure
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1
        current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
        job_percentage = int((current_time / worker_video_length) * 100)
        previous_total_generated_latent_frames = 0
        for section_index in range(total_latent_sections):
            previous_total_generated_latent_frames= total_generated_latent_frames
            debug_print("in callback check for flags")

            if stream.input_queue.top() == 'abort':
                alert_print(f"Worker for section_index {section_index} in range {total_latent_sections} - Received abort signal, stopping processing")
                stream.output_queue.push(('abort', (job_percentage, worker_job)))
                success_print("before worker callback User abort_deletes the task, keping whatever output video has been made so far.")
                return

            if stream.input_queue.top() == 'abort_delete':
                alert_print(f"Worker for section_index {section_index} in range {total_latent_sections} - Received abort_delete signal, stopping processing")
                stream.output_queue.push(('abort_delete', (job_percentage, worker_job)))
                success_print("before worker callback User abort_deletes the task, deleting whatever output video has been made so far.")
                return
                
            if stream.input_queue.top() == 'failed':
                e = f"Error in section_index {section_index} in range {total_latent_sections} - had a failed job, stopping processing"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                worker_job.error_message = f"[{timestamp}] - {str(e)}"
                save_queue()
                alert_print(f"job updated to include error message: {e}")
                alert_print(f"{e}")
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                stream.output_queue.push(('failed', (job_percentage, worker_job)))
                return
                
            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=worker_gpu_memory)

            if worker_use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=worker_steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)
            def callback(d):
                current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
                job_percentage = int((current_time / worker_video_length) * 100)
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'abort':
                    # stream.input_queue.pop()
                    stream.output_queue.push(('abort', (job_percentage, worker_job)))
                    success_print("In worker callback User aborts the task, keping whatever output video has been made so far.")
                    raise KeyboardInterrupt('User aborts the task, keeping whatever output video has been made so far.')
                    
                if stream.input_queue.top() == 'abort_delete':
                    # stream.input_queue.pop()
                    stream.output_queue.push(('abort_delete', (job_percentage, worker_job)))
                    success_print("In worker callback User aborts the task, deleting whatever output video has been made so far.")
                    raise KeyboardInterrupt('User aborts the task, deleting whatever output video has been made so far.')
                    
                current_step = d['i'] + 1
                step_percentage = int(100.0 * current_step / worker_steps)
                step_desc = f'this is Step {current_step} of {worker_steps} in processing the next chunk of video.'
                step_progress = make_progress_bar_html(step_percentage, f'Step Progress: {step_percentage}%')


                job_desc = f"""<br>
Job Progress:   {job_percentage}% - So far {current_time:.2f} seconds of {worker_video_length} seconds have been completed<br>
Job Name:      {worker_job_name} Job Type: {job_type} {"" if worker_image_strength == 1.0 else f"Initial Image Strength: {worker_image_strength * 100:.0f}%"}<br>
Source Image:  {worker_job.source_name}<br>
Output folder: {worker_outputs_folder}<br>
Prompt:        {worker_prompt} - Negative Prompt: {worker_n_prompt}<br>
Lora Model:    {worker_lora_model}{f" - Lora Weight: {worker_lora_weight}" if worker_lora_model != "None" else ""}<br>
Settings:      Seed: {worker_seed} - Steps: {worker_steps} - Teacache: {worker_use_teacache} - MP4 Compression: {worker_mp4_crf}<br>
CFG Scaleing:  GS: {worker_gs} - RS: {worker_rs} - CFG: {worker_cfg}<br>"""

                stream.output_queue.push(('progress', (preview, step_desc, step_progress, job_desc)))
                return # callback return
            indices = torch.arange(0, sum([1, 16, 2, 1, worker_latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, worker_latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=worker_latent_window_size * 4 - 3,
                real_guidance_scale=worker_cfg,
                distilled_guidance_scale=worker_gs,
                guidance_rescale=worker_rs,
                # shift=3.0,
                num_inference_steps=worker_steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = worker_latent_window_size * 2
                overlapped_frames = worker_latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            previous_filename = os.path.join(outputs_folder, f'{worker_job_name}_{previous_total_generated_latent_frames}.mp4')
            output_filename = os.path.join(outputs_folder, f'{worker_job_name}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=worker_mp4_crf)
            if os.path.exists(previous_filename):
                try:
                    os.remove(previous_filename)
                    debug_print (f"Deleted old mp4 chunk {previous_filename}")
                except:
                    traceback.print_exc()    

            debug_print (f"adding metadata to {output_filename}")
            add_meta_data_to_mp4(worker_job, worker_seed, output_filename)
            debug_print (f"added metadata to {output_filename}")
    
            # Calculate current progress percentage
            current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
            job_percentage = min(int((current_time / worker_video_length) * 100), 100)
            stream.output_queue.push(('file', (output_filename, job_percentage)))

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
        info_print("done if not in ['abort', 'abort_delete', 'failed']")


    except:

        traceback.print_exc()
        
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
    # except Exception as e:
        # #Capture the traceback as a string
        # error_text = io.StringIO()
        # traceback.print_exc(file=error_text)
        # error_text = error_text.getvalue()
        # traceback.print_exc()  # Also print to console
        
        # if not high_vram:
            # unload_complete_models(
                # text_encoder, text_encoder_2, image_encoder, vae, transformer
            # )
        # if stream.input_queue.top() not in ['abort', 'abort_delete']:
            # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # if hasattr(worker_job, 'error_message') and worker_job.error_message:
                # worker_job.error_message += f"\n\n[{timestamp}] - {error_text}"
            # else:
                # worker_job.error_message = f"[{timestamp}] - {error_text}"
            # save_queue()
            # alert_print(f"job updated to include error message: {error_text}")
            # stream.output_queue.push(('failed', job_percentage, worker_job))
            # return
            # raise KeyboardInterrupt('ERRORS OCCURED WITH THE JOB')
    info_print("done if not in ['abort', 'abort_delete', 'failed']")
    # need something here

    completed_job = worker_job
    success_print(f"job {worker_job.job_name} completed")
    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    stream.output_queue.push(('done',( completed_job,output_filename)))




def create_still_frame_video(job, output_path, add_noise_to_image):
    """Convert a still image into a single-frame MP4 video, or create text2video start frame"""
    image_path = job.image_path
    try:
        if add_noise_to_image:
            if os.path.join(temp_queue_images, f"new_input_image_{job.job_name}.png"):
                img = cv2.imread (os.path.join(temp_queue_images, f"new_input_image_{job.job_name}.png"))
                if img is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                text = f"THIS IS YOUR NEW\nINPUT IMAGE\nCREATION OF THIS VIDEO\nHAS BEGUN\n1st video segment\n will appear here\nwhen step progress\n reaches 100 percent"

        elif image_path == "text2video":
            # Create black background image
            img = np.zeros((512, 512, 3), dtype=np.uint8)  # Black background
            text = "TEXT TO VIDEO\nCREATION HAS BEGUN\n1st video segment\n will appear here\nwhen step progress\n reaches 100 percent"
        else:
            if job.image_strength < 1.0:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                text = f"IMAGE TO IMAGE TO VIDEO\nCreation has begun\n the image will BE DENOISED\n with image stregth of {job.image_strength * 100:.0f}%\n then the 1st video segment\n will appear here\nwhen step progress\n reaches 100 percent"
            if job.image_strength == 1.0:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                text = "IMAGE TO VIDEO\nCREATION HAS BEGUN\n1st video segment\n will appear here\nwhen step progress\n reaches 100 percent"



        # Get image dimensions
        height, width = img.shape[:2]
        
        # Ensure dimensions are even (required by H.264)
        width = width - (width % 2)  # Make width even
        height = height - (height % 2)  # Make height even
        if width != img.shape[1] or height != img.shape[0]:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = min(width, height) * 0.002  # Scale font based on image size
        thickness = 2
        color = (0, 255, 255)  # Yellow in BGR
        
        # Split text into lines
        lines = text.split('\n')
        
        # Get size of each line
        line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
        line_heights = [size[1] for size in line_sizes]
        max_line_width = max(size[0] for size in line_sizes)
        total_text_height = sum(line_heights) + (len(lines) - 1) * 10  # 10 pixels between lines
        
        # Calculate starting Y position to center text block
        y_start = (height - total_text_height) // 2
        
        # Draw each line with black border
        for i, line in enumerate(lines):
            # Calculate x position to center this line
            text_size = line_sizes[i]
            x = (width - text_size[0]) // 2
            y = y_start + sum(line_heights[:i+1]) + i * 10
            
            # Draw black border
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                cv2.putText(img, line, (x+dx, y+dy), font, font_scale, (0,0,0), thickness+1, cv2.LINE_AA)
            
            # Draw yellow text
            cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Create a temporary PNG for ffmpeg input
        temp_png = output_path + "_temp.png"
        cv2.imwrite(temp_png, img)


        try:
            # FFmpeg command to create video directly from the image
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-loop', '1',  # Loop the input
                '-i', temp_png,  # Input image
                '-t', '1',  # Duration in seconds
                '-r', '30',  # Frame rate
                '-c:v', 'libx264',  # Use H.264 codec
                '-preset', 'medium',  # Encoding speed preset
                '-tune', 'stillimage',  # Optimize for still image
                '-pix_fmt', 'yuv420p',  # Required for compatibility
                '-movflags', '+faststart',  # Enable fast start
                output_path
            ]
            
            # Run ffmpeg with CREATE_NO_WINDOW flag on Windows
            if sys.platform.startswith('win'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                try:
                    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, startupinfo=startupinfo)
                except subprocess.CalledProcessError as e:
                    alert_print(f"FFmpeg error while creating preview video: {e.stderr}")
                    if os.path.exists(temp_png):
                        os.remove(temp_png)
                    return None
            else:
                try:
                    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    alert_print(f"Error creating still frame video: {e.stderr}")
                    if os.path.exists(temp_png):
                        os.remove(temp_png)
                    return None
            
            # Clean up temp file
            if os.path.exists(temp_png):
                os.remove(temp_png)
            success_print (f"created still frame video {output_path}")
            return output_path
            
        except FileNotFoundError:
            alert_print("FFmpeg not found. Please install FFmpeg to see the input image preview.")
            if os.path.exists(temp_png):
                os.remove(temp_png)
            return None
        except subprocess.SubprocessError as e:
            alert_print(f"FFmpeg error while creating preview video: {str(e)}")
            if os.path.exists(temp_png):
                os.remove(temp_png)
            return None
            
    except Exception as e:
        alert_print(f"Error creating still frame video preview image: {str(e)}")
        traceback.print_exc()
        return None


def extract_thumb_from_processing_mp4(job, mp4_path, job_percentage):
    status_overlay = "PROCESSING" if job.status=="processing" else "DONE" if job.status=="completed" else job.status.upper()
    status_color = {
        "pending": (0, 255, 255),  # BGR for yellow
        "processing": (255, 0, 0),  # BGR for blue
        "completed": (0, 255, 0),   # BGR for green
        "failed": (0, 0, 255)       # BGR for red
    }.get(job.status, (255, 255, 255))  # BGR for white

    if os.path.exists(mp4_path):

        cap = cv2.VideoCapture(mp4_path)
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Seek to middle frame
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            # target thumbnail size
            THUMB_SIZE = 200

            # get frame dims
            h, w = frame.shape[:2]

            # scale so that the larger dimension becomes THUMB_SIZE
            scale = THUMB_SIZE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # resize the frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # create black background and center the resized frame
            thumb = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
            x_off = (THUMB_SIZE - new_w) // 2
            y_off = (THUMB_SIZE - new_h) // 2
            thumb[y_off : y_off + new_h, x_off : x_off + new_w] = resized

            # Overlay centered status text, 
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2

            if job.status == "processing":
                text1 = f"{job_percentage}%"
                text2 = "PROCESSED"
                
                # Calculate text sizes for both lines
                (text1_w, text1_h), _ = cv2.getTextSize(text1, font, scale, thickness)
                (text2_w, text2_h), _ = cv2.getTextSize(text2, font, scale, thickness)
                
                x1 = (thumb.shape[1] - text1_w) // 2
                x2 = (thumb.shape[1] - text2_w) // 2
                y1 = (thumb.shape[0] - text2_h) // 2  # Center vertically
                y2 = y1 + text2_h + 10  # Add small gap between lines
                
                # Add black outline and text for both lines
                for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    cv2.putText(thumb, text1, (x1+offset[0], y1+offset[1]), font, scale, (255,255,255), thickness+2, cv2.LINE_AA)
                    cv2.putText(thumb, text2, (x2+offset[0], y2+offset[1]), font, scale, (255,255,255), thickness+2, cv2.LINE_AA)
                
                # Add colored text
                cv2.putText(thumb, text1, (x1, y1), font, scale, status_color, thickness, cv2.LINE_AA)
                cv2.putText(thumb, text2, (x2, y2), font, scale, status_color, thickness, cv2.LINE_AA)
            else:
                # Single line for other statuses
                text = status_overlay
                (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                x = (thumb.shape[1] - text_w) // 2
                y = (thumb.shape[0] + text_h) // 2
                
                # Add black outline
                for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    cv2.putText(thumb, text, (x+offset[0], y+offset[1]), font, scale, (255,255,255), thickness+2, cv2.LINE_AA)
                
                # Add colored text
                cv2.putText(thumb, text, (x, y), font, scale, status_color, thickness, cv2.LINE_AA)

            thumb_path = os.path.join(temp_queue_images, f'thumb_{job.job_name}.png')
            cv2.imwrite(thumb_path, thumb)
        cap.release()
        debug_print (f"extracted thumb from {mp4_path}")    
    return(job, thumb_path)

def process():
    global stream

    load_queue()
    # First check for pending jobs
    pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]

    if not pending_jobs:
        return
    # Process first pending job
    pending_job = pending_jobs[0]
    debug_print(f"start button clicked, job starting {pending_job.job_name}")

    success_print(f"marking job {pending_job.job_name} as processing")
    queue_table_update, queue_display_update = mark_job_processing(pending_job)
    save_queue()
    
    # Create starting video
    temp_video_path = os.path.join(temp_queue_images, f"temp_start_{pending_job.job_name}.mp4")
    starting_image = create_still_frame_video(pending_job, temp_video_path, add_noise_to_image = None)
    # Start processing


     # Initial yield 
    yield (
        gr.update(interactive=True),      # queue_button
        gr.update(interactive=False),     # start_button
        gr.update(interactive=True),      # abort_button
        gr.update(interactive=True),      # abort_delete_button
        None,                             # preview_image
        gr.update(visible=True, label="input image", value=starting_image),  # still frame video
        "",                               # progress_desc1
        "",                               # progress_bar1
        "",                               # progress_desc2
        update_queue_display(),           # queue_display
        update_queue_table()             # queue_table
    )
    
    debug_print ("innitial yield")
    stream = AsyncStream() 
    debug_print ("innitializing stream = AsyncStream")
    debug_print ("called async_run worker, pending_job")

    async_run(worker, pending_job)    ###### the first run is needed to start the stream all later runs will be dunt in the while true loop. pending job is the one that will be processed
    
    debug_print ("called async_run worker, pending_job after clicking start button")
    while True:
        try:
            flag, data = stream.output_queue.next()
            if flag == 'file':
                debug_print("[DEBUG] Process - file flag")
                try:
                    output_filename, job_percentage = data
                except (ValueError, TypeError):
                    debug_print(f"Invalid data format received: {data}")
                    continue
                
                # Find the currently processing job
                processing_jobs = [job for job in job_queue if job.status.lower() == "processing"]
                if not processing_jobs:
                    debug_print("No processing jobs found, skipping file update")
                    continue
                    
                file_job = processing_jobs[0]
                try:
                    extract_thumb_from_processing_mp4(file_job, output_filename, job_percentage)
                except Exception as e:
                    debug_print(f"Error extracting thumbnail: {str(e)}")
                    continue

                yield (
                    gr.update(interactive=True),    # queue_button
                    gr.update(interactive=False),   # start_button
                    gr.update(interactive=True),    # abort_button
                    gr.update(interactive=True),    # abort_delete_button
                    gr.update(),        # preview_image (File Output: visible)
                    gr.update(visible=True, label=f"Video Output {job_percentage} % complete", value=output_filename), # result_video
                    gr.update(),    # keep last step progress
                    gr.update(),       # keep last step progress bar
                    gr.update(),     # keep last job progress
                    update_queue_display(),         # queue_display
                    update_queue_table()           # queue_table
                )
                print(f"PROGRESS: {GREEN}{job_percentage} %{RESET} done \n JOB NAME: {YELLOW}{file_job.job_name}{RESET} \n  {GREEN}Video: {output_filename}{RESET}")
                if os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception as e:
                        alert_print(f"Error removing temp video file {temp_video_path}: {str(e)}")
          #=======================PROGRESS STAGE=============

            if flag == 'progress':
                preview, step_desc, step_progress, job_desc = data
                yield (
                    gr.update(interactive=True),    # queue_button
                    gr.update(interactive=False),   # start_button
                    gr.update(interactive=True),    # abort_button
                    gr.update(interactive=True),    # abort_delete_button
                    gr.update(visible=True, value=preview), # preview_image
                    gr.update(),                    # leave result_video as is
                    step_desc,                      # progress_desc1
                    step_progress,                  # progress_bar1 
                    job_desc,                       # progress_desc2 
                    gr.update(),                    # queue_display (no update)
                    gr.update()                     # queue_table (no update)
                )

        
          #=======================failed STAGE=============

            if flag == 'failed':
                job_percentage, failed_job = data   
                debug_print(f"worker callback - failed flag recieved for job {failed_job.job_name}, job percentage {job_percentage}, {flag}")    
                mark_job_failed(failed_job)
                alert_print(f"saveing failed job to outputs folder {failed_job.outputs_folder}")                        
                mp4_path = os.path.join(Config.OUTPUTS_FOLDER, f"{failed_job.job_name}.mp4")
                extract_thumb_from_processing_mp4(failed_job, mp4_path, job_percentage)
                move_mp4_to_custom_output_folder(failed_job)
                if image_strength != 1: cleanup_temp_i2i(failed_job)
                alert_print(f"failed job video saved to outputs folder {failed_job.outputs_folder}")
                queue_table_update, queue_display_update = mark_job_failed(failed_job)
                save_queue()
                    
                alert_print("the job was marked as failed")
                yield (
                    gr.update(interactive=True),      # queue_button
                    gr.update(interactive=True),      # start_button
                    gr.update(interactive=False),     # abort_button
                    gr.update(interactive=False),     # abort_delete_button
                    None,                             # preview_image
                    gr.update(visible=True),          # result_video
                    "Job failed and marked failed at the top of the queue",  # progress_desc1
                    "",                               # progress_bar1
                    "Job Failed",                     # progress_desc2
                    update_queue_display(),           # queue_display
                    update_queue_table()             # queue_table
                )
                
                # Check for next pending job
                failed_job = None
                next_job = None
                pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                if pending_jobs:
                    next_job = pending_jobs[0]
                if next_job:
                    info_print(f"there are {len(pending_jobs)} pending jobs remaining in the queue")
                    info_print(f"{RESET} {BLUE} Next job {YELLOW}{next_job.job_name}{GREEN} is sent to the worker and marked as Processing ")   
                    
                    queue_table_update, queue_display_update = mark_job_processing(next_job)
                    info_print(f"{RESET} {BLUE} Next job {YELLOW}{next_job.job_name}{GREEN} is is marked as Processing ")    

                    save_queue()
                    
                    # Create starting video for next job
                    temp_video_path = os.path.join(temp_queue_images, f"temp_start_{next_job.job_name}.mp4")
                    starting_image = create_still_frame_video(next_job, temp_video_path, add_noise_to_image=None)
                                   
                    yield (
                        gr.update(interactive=True),      # queue_button
                        gr.update(interactive=False),     # start_button
                        gr.update(interactive=True),      # abort_button
                        gr.update(interactive=True),      # abort_delete_button
                        None,          # preview_image
                        gr.update(visible=True, label=f"input image", value=starting_image),  # still frame video
                        "",          # progress_desc1
                        "",          # progress_bar1
                        "",          # progress_desc2
                        update_queue_display(),           # queue_display
                        update_queue_table()             # queue_table
                    )

                    debug_print ("called async_run worker, next_job after a failed job")

                    async_run(worker, next_job)

                else:
                    debug_print("No more pending jobs to process")
                    yield (
                        gr.update(interactive=True),   # queue_button (always enabled)
                        gr.update(interactive=True),   # start_button
                        gr.update(interactive=False),  # abort_button
                        gr.update(interactive=False),  # abort_delete_button
                        None,  # preview_image
                        gr.update(output_filename),  # show result_video with final file
                        "No more pending jobs to process",  # progress_desc1 (step progress)
                        None,  # progress_bar1 (step progress)
                        "Job Finished Successfully",    # progress_desc2 (job progress)
                        update_queue_display(),        # queue_display
                        update_queue_table()         # queue_table
                    )

                    stream = None
                    return


          #=======================Abort STAGE=============

            if flag == 'abort':
                job_percentage, aborted_job = data
                debug_print(f"worker sent - abort flag recieved for job {aborted_job.job_name}, job percentage {job_percentage}, {flag}")    
                if aborted_job:
                    clean_up_temp_mp4png(aborted_job)
                    if image_strength != 1: cleanup_temp_i2i(aborted_job)
                    mp4_path = os.path.join(Config.OUTPUTS_FOLDER, f"{aborted_job.job_name}.mp4")
                    debug_print(f"trying to save aborted job video to outputs folder {aborted_job.outputs_folder}")                        
                    mp4_path = os.path.join(Config.OUTPUTS_FOLDER, f"{aborted_job.job_name}.mp4")
                    extract_thumb_from_processing_mp4(aborted_job, mp4_path, job_percentage)
                    move_mp4_to_custom_output_folder(aborted_job)
                    debug_print(f"aborted job video saved to outputs folder {aborted_job.outputs_folder}")
                    queue_table_update, queue_display_update = mark_job_pending(aborted_job)

                    
                    yield (
                        gr.update(interactive=True),    # queue_button
                        gr.update(interactive=True),    # start_button
                        gr.update(interactive=False),   # abort_button
                        gr.update(interactive=False),    # abort_delete_button
                        gr.update(visible=False),       # preview_image (Abort: hidden)
                        gr.update(label=f"Video aborted {job_percentage}% complete video saved"),       # result_video (Aborted)
                        "Job Aborted and marked pending at the top of the queue",# progress_desc1 (step progress)

                        "",                            # progress_bar1 (step progress)
                        "Processing stopped",          # progress_desc2 (job progress)
                        update_queue_display(),         # queue_display
                        update_queue_table()           # queue_table
                    )
                    save_queue()
                    stream = None
                    return 


          #=======================Abort and Delete STAGE=============

            if flag == 'abort_delete':
                job_percentage, aborted_job = data
                debug_print(f"worker callback - abort_delete flag recieved for job {aborted_job.job_name}, job percentage {job_percentage}, {flag}")    
                if aborted_job:
                    clean_up_temp_mp4png(aborted_job)
                    if image_strength != 1: cleanup_temp_i2i(aborted_job)
                    mp4_path = os.path.join(Config.OUTPUTS_FOLDER, f"{aborted_job.job_name}.mp4")
                    
                    if os.path.exists(mp4_path):    
                        os.remove(mp4_path)                               
                        debug_print(f"aborted job video deleted from {mp4_path}")
                    queue_table_update, queue_display_update = mark_job_pending(aborted_job)
                    info_print(f"job abort_and_delete_process not saving aborted job video  for{aborted_job.job_name} marked pending ")                        


                    
                    yield (
                        gr.update(interactive=True),    # queue_button
                        gr.update(interactive=True),    # start_button
                        gr.update(interactive=False),   # abort_button
                        gr.update(interactive=False),    # abort_delete_button
                        gr.update(visible=False),       # preview_image (Abort: hidden)
                        gr.update(label=f"Video aborted and deleted"),       # result_video (Aborted)
                        "Job Aborted and marked pending at the top of the queue",# progress_desc1 (step progress)

                        "",                            # progress_bar1 (step progress)
                        "Processing stopped",          # progress_desc2 (job progress)
                        update_queue_display(),         # queue_display
                        update_queue_table()           # queue_table
                    )
                    save_queue()
                    stream = None
                    return 

          #=======================Completed Job STAGE=============

            if flag == 'done':
                completed_job, output_filename = data
                cleanup_job = completed_job
                debug_print(f"worker callback - done flag recieved for job {cleanup_job.job_name}, {flag}")    
                success_print(f"completed job recieved at done flag job name {cleanup_job.job_name}")
                print(f"{GREEN}100% COMPLETED{RESET} - {BLUE}JOB NAME: {cleanup_job.job_name}{RESET}")

                # previous job completed
                clean_up_temp_mp4png(cleanup_job)
                queue_table_update, queue_display_update = mark_job_completed(cleanup_job)
                move_mp4_to_custom_output_folder(cleanup_job)
                if image_strength != 1: cleanup_temp_i2i(cleanup_job)
                debug_print(f"extracting thumb from completed job {cleanup_job.job_name}.mp4 and moved video to the jobs custom outputs folder {cleanup_job.outputs_folder}")
                save_queue()

                # Check for next pending job
                cleanup_job = None
                next_job = None
                pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                if pending_jobs:
                    next_job = pending_jobs[0]
                if next_job:
                    info_print(f"there are {len(pending_jobs)} pending jobs remaining in the queue")
                    info_print(f"{RESET} {BLUE} Next job {YELLOW}{next_job.job_name}{GREEN} is sent to the worker and marked as Processing ")    
                    mark_job_processing(next_job)
                    save_queue()
                    
                    # Create starting video for next job
                    temp_video_path = os.path.join(temp_queue_images, f"temp_start_{next_job.job_name}.mp4")
                    starting_image = create_still_frame_video(next_job, temp_video_path, add_noise_to_image=None)
                                   
                    yield (
                        gr.update(interactive=True),      # queue_button
                        gr.update(interactive=False),     # start_button
                        gr.update(interactive=True),      # abort_button
                        gr.update(interactive=True),      # abort_delete_button
                        None,          # preview_image
                        gr.update(visible=True, label=f"input image", value=starting_image),  # still frame video
                        "",          # progress_desc1
                        "",          # progress_bar1
                        "",          # progress_desc2
                        update_queue_display(),           # queue_display
                        update_queue_table()             # queue_table
                    )

                    debug_print ("called async_run worker, next_job after a completed job")

                    async_run(worker, next_job)

                else:
                    debug_print("No more pending jobs to process")
                    yield (
                        gr.update(interactive=True),   # queue_button (always enabled)
                        gr.update(interactive=True),   # start_button
                        gr.update(interactive=False),  # abort_button
                        gr.update(interactive=False),  # abort_delete_button
                        None,  # preview_image
                        gr.update(output_filename),  # show result_video with final file
                        "No more pending jobs to process",  # progress_desc1 (step progress)
                        None,  # progress_bar1 (step progress)
                        "Job Finished Successfully",    # progress_desc2 (job progress)
                        update_queue_display(),        # queue_display
                        update_queue_table()         # queue_table
                    )

                    stream = None
                    return

        except Exception as e:
            alert_print(f"Error in process loop: {str(e)}")
            return

def abort_process():
    debug_print(f"abort button clicked")

    """Handle abort generation button click - stop all processes and change all processing jobs to pending jobs"""
    stream.input_queue.push('abort')
    return (
        update_queue_table(),  # dataframe
        update_queue_display(),  # gallery
        gr.update(interactive=True),  # queue_button
        gr.update(interactive=True),  # start_button
        gr.update(interactive=False),  # abort_button
        gr.update(interactive=False)  # abort_delete_button
    )

def abort_and_delete_process():
    debug_print(f"abort and delete button clicked")

    """Handle abort generation button click - stop all processes and change all processing jobs to pending jobs"""
    stream.input_queue.push('abort_delete')
    return (
        update_queue_table(),  # dataframe
        update_queue_display(),  # gallery
        gr.update(interactive=True),  # queue_button
        gr.update(interactive=True),  # start_button
        gr.update(interactive=False),  # abort_button
        gr.update(interactive=False)  # abort_delete_button
    )
    

def add_to_queue_handler(input_image, prompt, n_prompt, lora_model, lora_weight, video_length, job_name, create_job_outputs_folder, create_job_history_folder, use_teacache, seed, steps, cfg, gs, rs, image_strength, mp4_crf, gpu_memory, create_job_keep_completed_job, keep_temp_png, status="pending", source_name=""):
    """Handle adding a new job to the queue"""

    try:
        # Ensure lora_model is always a string
        if not isinstance(lora_model, str):
            lora_model = "None"

        if prompt is None and input_image is None:
            return (
                update_queue_table(),         # queue_table
                update_queue_display(),       # queue_display
                gr.update(interactive=True)   # queue_button (always enabled)
            )
        if not job_name:  # This will catch both None and empty string
            job_name = "job"  # Remove the underscore here since we add it later
        # Handle text-to-video case (no input image)
        if input_image is None:
            job_name = add_to_queue(
                prompt=prompt,
                n_prompt=n_prompt,
                lora_model=lora_model,
                lora_weight=lora_weight,
                input_image=None,  # Pass None for text-to-video
                video_length=video_length,
                job_name=job_name,
                create_job_outputs_folder=create_job_outputs_folder,
                create_job_history_folder=create_job_history_folder,
                use_teacache=use_teacache,
                seed=seed,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                image_strength=image_strength,
                mp4_crf=mp4_crf,
                gpu_memory=gpu_memory,
                create_job_keep_completed_job=create_job_keep_completed_job,
                keep_temp_png=keep_temp_png,
                status="pending",
                source_name=""
            )
            save_queue()
            return (
                update_queue_table(),         # queue_table
                update_queue_display(),       # queue_display
                gr.update(interactive=True)   # queue_button (always enabled)
            )

        # Handle image-to-video cases
        if isinstance(input_image, list):
            # Multiple images case
            original_job_name = job_name  # Store the original job name prefix
            for img_tuple in input_image:
                original_basename = os.path.splitext(os.path.basename(img_tuple[0]))[0]
                input_image_np = np.array(Image.open(img_tuple[0]))  # Convert to numpy array
                # Add job for each image, using original job name prefix
                job_name = add_to_queue(
                    prompt=prompt,
                    n_prompt=n_prompt,
                    lora_model=lora_model,
                    lora_weight=lora_weight,
                    input_image=input_image_np,
                    video_length=video_length,
                    job_name=original_job_name,  # Use original prefix each time
                    create_job_outputs_folder=create_job_outputs_folder,
                    create_job_history_folder=create_job_history_folder,
                    use_teacache=use_teacache,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    gs=gs,
                    rs=rs,
                    image_strength=image_strength,
                    mp4_crf=mp4_crf,
                    gpu_memory=gpu_memory,
                    create_job_keep_completed_job=create_job_keep_completed_job,
                    keep_temp_png=keep_temp_png,
                    status="pending",
                    source_name=original_basename
                )
                # Create thumbnail for the job
                job = next((job for job in job_queue if job.job_name == job_name), None)
                if job and job.image_path:
                    job.thumbnail = create_thumbnail(job, status_change=True)
                    save_queue()
        else:
            # Single image case
            original_basename = os.path.splitext(os.path.basename(input_image[0]))[0]
            input_image_np = np.array(Image.open(input_image[0]))  # Convert to numpy array
            # Add single image job
            job_name = add_to_queue(
                prompt=prompt,
                n_prompt=n_prompt,
                lora_model=lora_model,
                lora_weight=lora_weight,
                input_image=input_image_np,
                video_length=video_length,
                job_name=job_name,
                create_job_outputs_folder=create_job_outputs_folder,
                create_job_history_folder=create_job_history_folder,
                use_teacache=use_teacache,
                seed=seed,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                image_strength=image_strength,
                mp4_crf=mp4_crf,
                gpu_memory=gpu_memory,
                create_job_keep_completed_job=create_job_keep_completed_job,
                keep_temp_png=keep_temp_png,
                status="pending",
                source_name=original_basename
            )

            job = next((job for job in job_queue if job.job_name == job_name), None)
            if job and job.image_path:
                job.thumbnail = create_thumbnail(job, status_change=True)
                save_queue()  # Save after changing statuses

        return (
            update_queue_table(),         # queue_table
            update_queue_display(),       # queue_display
            gr.update(interactive=True)   # queue_button (always enabled)
        )
    except Exception as e:
        alert_print(f"Error in add_to_queue_handler: {str(e)}")
        traceback.print_exc()
        return (
            update_queue_table(),         # queue_table
            update_queue_display(),       # queue_display
            gr.update(interactive=True)   # queue_button (always enabled)
        )

def create_default_settings():
    """Create default settings.ini file"""
    config = configparser.ConfigParser()
    config['Job Defaults'] = {}
    config['System Defaults'] = {}
    config['Model Defaults'] = {}
    save_settings(config)






def restore_model_default(model_type):
    """Restore a specific model to its original default in settings.ini"""
    try:
        # Get original defaults
        original_defaults = Config.get_original_defaults()
        original_value = original_defaults[f'DEFAULT_{model_type.upper()}']
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update the config
        config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(original_value)
        
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Update Config object
        setattr(Config, f'DEFAULT_{model_type.upper()}', original_value)
        
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        # Return status message and individual model lists
        return (
            f"{model_type} restored to original default successfully. Restart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder'],
            models['image2image_model']
        )
    except Exception as e:
        alert_print(f"Error restoring {model_type} default: {str(e)}")
        return f"Error restoring {model_type} default: {str(e)}", None, None, None, None, None, None, None, None



def delete_all_jobs():
    """Delete all jobs from the queue"""
    global job_queue
    job_queue = []
    save_queue()
    return #update_queue_table(), update_queue_display()


def move_job_to_top(job_name):
    """Move a job to the top of the queue, maintaining processing job at top"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_name == job_name:
                current_index = i
                break
        
        if current_index is None:
            return# update_queue_table(), update_queue_display()
        
        # Get the job
        job = job_queue[current_index]
        
        # Remove from current position
        job_queue.pop(current_index)
        
        # Find the appropriate insert position based on job status
        insert_index = 0
        if job.status == "pending":
            # For pending jobs, find the first pending job
            for i, existing_job in enumerate(job_queue):
                if existing_job.status == "pending":
                    insert_index = i
                    break
                elif existing_job.status != "processing":
                    insert_index = i
                    break
        else:
            # For other jobs, find first non-processing job
            for i, existing_job in enumerate(job_queue):
                if existing_job.status != "processing":
                    insert_index = i
                    break
        
        # Insert the job at the found index
        job_queue.insert(insert_index, job)
        save_queue()
        
        # return #update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error moving job to top: {str(e)}")
        traceback.print_exc()
        # return #update_queue_table(), update_queue_display()

def move_job_to_bottom(job_name):
    """Move a job to the bottom of the queue, maintaining completed jobs at bottom"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_name == job_name:
                current_index = i
                break
        
        if current_index is None:
            return #update_queue_table(), update_queue_display()
        
        # Get the job
        job = job_queue[current_index]
        
        # Remove from current position
        job_queue.pop(current_index)
        
        # Find the first completed job
        insert_index = len(job_queue)
        for i, existing_job in enumerate(job_queue):
            if existing_job.status == "completed":
                insert_index = i
                break
        
        # Insert the job at the found index
        job_queue.insert(insert_index, job)
        save_queue()
        
        # return #update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error moving job to bottom: {str(e)}")
        traceback.print_exc()
        # return #update_queue_table(), update_queue_display()

def move_job(job_name, direction):
    """Move a job up or down one position in the queue while maintaining sorting rules"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_name == job_name:
                current_index = i
                break
        
        if current_index is None:
            return #update_queue_table(), update_queue_display()
        
        # Get the job
        job = job_queue[current_index]
        
        # Calculate new index based on direction and sorting rules
        if direction == 'up':
            if current_index == 0:  # Already at top
                return
                
            # Get the job above
            prev_job = job_queue[current_index - 1]
            
            # Check movement rules
            # No jobs can move above processing jobs
            if prev_job.status == "processing":
                return
                
            # Move is allowed - swap positions
            new_index = current_index - 1
            
        else:  # direction == 'down'
            if current_index >= len(job_queue) - 1:  # Already at bottom
                return
                
            # Get the job below
            next_job = job_queue[current_index + 1]
            
            # Don't allow moving below completed jobs
            if job.status == "pending" and next_job.status == "completed":
                return
                
            # Move is allowed - swap positions
            new_index = current_index + 1
        
        # Perform the move
        job_queue.pop(current_index)
        job_queue.insert(new_index, job)
        save_queue()
        
        # return #update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error moving job: {str(e)}")
        traceback.print_exc()
        # return #update_queue_table(), update_queue_display()

def remove_job(job_name):
    """Delete a job from the queue and its associated files"""
    try:
        # Find and remove job from queue
        for job in job_queue:
            if job.job_name == job_name:
                # Delete associated files
                if job.image_path and os.path.exists(job.image_path):
                    os.remove(job.image_path)
                if os.path.exists(job.thumbnail):
                    os.remove(job.thumbnail)
                job_queue.remove(job)
                break
        save_queue()
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
        # return #update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error deleting job: {str(e)}")
        traceback.print_exc()
        # return #update_queue_table(), update_queue_display()

def handle_queue_action(evt: gr.SelectData):
    """Handle queue action button clicks"""
    # Default empty return values
    empty_values = (
        "",  # prompt
        "",  # n_prompt
        None, #lora_model
        None,  # lora_weight 
        None,  # video_length
        None,  # seed
        None,  # use_teacache
        None,  # steps
        None,  # cfg
        None,  # gs
        None,  # rs
        None,  # image_strength
        None,  # mp4_crf
        None,  # gpu_memory
        None,  # keep_temp_png
        "",  # outputs_folder
        "",  # job_history_folder
        None,  # keep_completed_job
        "",  # job_name
        "",  # change_job_name
        gr.update(visible=False),  # edit group visibility
        gr.update(visible=True),  # edit_job_table_group visibility
        gr.update(),  # queue_table
        gr.update()  # queue_display
    )

    if evt.index is None or evt.value not in ["⏫️", "⬆️", "⬇️", "⏬️", "❌", "📋", "✎"]:
        # Only return updates for components that need to be cleared
        return empty_values
    
    row_index, col_index = evt.index
    button_clicked = evt.value
    try:
        job = job_queue[row_index]
    except IndexError as e:
        return empty_values
    
    # Handle simple queue operations that only need to update the queue display
    if button_clicked in ["⏫️", "⬆️", "⬇️", "⏬️", "❌", "📋"]:
        if button_clicked == "⏫️":
            move_job_to_top(job.job_name)
            return *empty_values[:-2], update_queue_table(), update_queue_display()
            
        elif button_clicked == "⬆️":
            move_job(job.job_name, 'up')
            return *empty_values[:-2], update_queue_table(), update_queue_display()
        elif button_clicked == "⬇️":
            move_job(job.job_name, 'down')
            return *empty_values[:-2], update_queue_table(), update_queue_display()
        elif button_clicked == "⏬️":
            move_job_to_bottom(job.job_name)
            return *empty_values[:-2], update_queue_table(), update_queue_display()
        elif button_clicked == "❌":
            remove_job(job.job_name)
            return *empty_values[:-2], update_queue_table(), update_queue_display()
        elif button_clicked == "📋":
            copy_job(job.job_name)
            return *empty_values[:-2], update_queue_table(), update_queue_display()

    # Handle edit action
    elif button_clicked == "✎":
        if job.status in ["pending", "completed", "failed"]:
            # LoRA dropdown logic (same as prepare_metadata_job_edit)
            lora_value = str(getattr(job, "lora_model", Config.DEFAULT_LORA_MODEL))
            lora_weight_value = float(getattr(job, "lora_weight", Config.DEFAULT_LORA_WEIGHT))
            lora_choices_actual = lora_choices.copy()
            lora_display_value = lora_value
            if lora_value != "None" and lora_value not in lora_choices_actual:
                lora_display_value = f"{lora_value} - missing"
                lora_choices_actual = [lora_display_value] + lora_choices_actual
            return (
                job.prompt,  # prompt
                job.n_prompt,  # n_prompt
                gr.update(value=lora_display_value, choices=lora_choices_actual),  # lora_model
                job.lora_weight,  # lora_weight
                job.video_length,  # video_length
                job.seed,  # seed
                job.use_teacache,  # use_teacache
                job.steps,  # steps
                job.cfg,  # cfg
                job.gs,  # gs
                job.rs,  # rs
                job.image_strength,  # image_strength
                job.mp4_crf,  # mp4_crf
                job.gpu_memory,  # gpu_memory
                job.keep_temp_png,  # keep_temp_png
                job.outputs_folder,  # outputs_folder
                job.job_history_folder,  # job_history_folder
                job.keep_completed_job,  # keep_completed_job
                job.job_name,  # job_name
                job.job_name,  # change_job_name
                gr.update(visible=True),  # edit_job_group
                gr.update(visible=False),  # edit_job_table_group
                gr.update(),  # queue_table
                gr.update()  # queue_display
            )
    
    return empty_values

def generate_new_job_name(old_job_name):
    """Helper function to generate new job name while preserving the prefix"""
    # Find the last separator (either hyphen or underscore)
    hyphen_index = old_job_name.rfind('-')
    underscore_index = old_job_name.rfind('_')
    last_separator_index = max(hyphen_index, underscore_index)
    
    if last_separator_index > 0:
        # Keep everything before the last separator
        prefix = old_job_name[:last_separator_index]
    else:
        # If no separator found, use the whole name as prefix
        prefix = old_job_name
        
    # Generate new hex and add with hyphen
    new_hex = uuid.uuid4().hex[:8]
    return f"{prefix}-{new_hex}"

def copy_job(job_name):
    """Create a copy of a job and insert it below the original"""
    try:
        # Find the job
        original_job = next((j for j in job_queue if j.job_name == job_name), None)
        if not original_job:
            return
            
        # Create a new job ID by keeping the prefix and adding a new hex suffix
        # Get prefix and generate new job name
        new_job_name = generate_new_job_name(original_job.job_name.rsplit('-', 1)[0])
        
        # Copy the image file
        if os.path.exists(original_job.image_path):
            new_image_path = os.path.join(temp_queue_images, f"queue_image_{new_job_name}.png")
            shutil.copy2(original_job.image_path, new_image_path)
        else:
            new_image_path = "text2video"
            
        # Create new job with copied parameters
        new_job = QueuedJob(
            prompt=original_job.prompt,
            n_prompt=original_job.n_prompt,
            image_path=new_image_path,
            lora_model=original_job.lora_model,
            lora_weight=original_job.lora_weight,
            video_length=original_job.video_length,
            job_name=new_job_name,
            seed=original_job.seed,
            use_teacache=original_job.use_teacache,
            steps=original_job.steps,
            cfg=original_job.cfg,
            gs=original_job.gs,
            rs=original_job.rs,
            image_strength=original_job.image_strength,
            status="pending",
            thumbnail="",
            mp4_crf=original_job.mp4_crf,
            gpu_memory=original_job.gpu_memory,
            keep_temp_png=original_job.keep_temp_png,
            outputs_folder=original_job.outputs_folder,
            job_history_folder=original_job.job_history_folder,
            keep_completed_job=original_job.keep_completed_job,
            source_name=original_job.source_name
        )
        
        # Find the original job's index
        original_index = job_queue.index(original_job)
        
        # Insert the new job right after the original
        job_queue.insert(original_index + 1, new_job)
        if new_image_path:
            new_job.thumbnail = create_thumbnail(new_job, status_change=True)

        queue_table_update, queue_display_update = mark_job_pending(new_job)
        save_queue()
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
        
    except Exception as e:
        alert_print(f"Error copying job: {str(e)}")
        traceback.print_exc()

css = make_progress_bar_css() + """
.gradio-gallery-container {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}
.gradio-gallery-container::-webkit-scrollbar {
    width: 8px !important;
}
.gradio-gallery-container::-webkit-scrollbar-track {
    background: #f0f0f0 !important;
}
.gradio-gallery-container::-webkit-scrollbar-thumb {
    background-color: #666 !important;
    border-radius: 4px !important;
}
.input-gallery,
.queue-gallery {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}
.input-gallery > div,
.queue-gallery > div {
    height: 100% !important;
    overflow-y: auto !important;
}
.input-gallery .gallery-container,
.queue-gallery .gallery-container {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}

/* Hide DataFrame headers */
.gradio-dataframe thead {
    display: none !important;
}

/* Fix widths for action columns */
.gradio-dataframe td:nth-child(1) { 
    width: 100px !important; 
    max-width: 100px !important; 
    padding: 0 !important; 
    text-align: left !important;
}

.gradio-dataframe td:nth-child(2),
.gradio-dataframe td:nth-child(3),
.gradio-dataframe td:nth-child(4),
.gradio-dataframe td:nth-child(5),
.gradio-dataframe td:nth-child(6),
.gradio-dataframe td:nth-child(7),
.gradio-dataframe td:nth-child(8) {
    width: 16px !important;
    min-width: 16px !important;
    max-width: 16px !important;
    text-align: left !important;
    padding: 0 !important;
}

.gradio-dataframe td:nth-child(9) { 
    width: 300px !important; 
    min-width: 300px !important; 
    text-align: left !important; 
}
"""


def edit_job(
    job_name,change_job_name, prompt, n_prompt, lora_model, lora_weight, video_length, outputs_folder, job_history_folder, use_teacache, seed, steps, cfg, gs, rs, image_strength, mp4_crf, gpu_memory, keep_completed_job, keep_temp_png
):
    #Edit a job's parameters
    try:
        # Find the job  need to add a gr field next to job_name called new_job_name make  job_name not visable  make new_job_name labled as jon_name visible and interactive
        for job in job_queue:
            if job.job_name == job_name:
                # Only allow editing if job is pending or completed
                if job.status not in ("pending", "completed", "failed"):
                    return update_queue_table(), update_queue_display(), gr.update(visible=False), gr.update(visible=True)
 
                # Update job parameters
                job.prompt = prompt
                job.n_prompt = n_prompt
                job.lora_model = lora_model
                job.lora_weight = lora_weight
                job.video_length = video_length
                job.outputs_folder = outputs_folder
                job.job_history_folder = job_history_folder
                job.use_teacache = use_teacache
                job.seed = seed
                job.steps = steps
                job.cfg = cfg
                job.gs = gs
                job.rs = rs
                job.image_strength = image_strength  # Add Image Strength
                job.mp4_crf = mp4_crf
                job.gpu_memory = gpu_memory
                job.keep_completed_job = keep_completed_job
                job.keep_temp_png = keep_temp_png
                job.error_message = "None"




                #added this if so that if job_name has changed then give it a new suffix hex and copy the the image_path to the new name     
                if job_name != change_job_name: #check for syntax
                    job.change_job_name = change_job_name
                    change_job_name = generate_new_job_name(job.change_job_name)
                    new_image_path = os.path.join(temp_queue_images, f"queue_image_{change_job_name}.png")
                    if job.image_path != "text2video":
                        shutil.copy2(job.image_path, new_image_path)
                        job.image_path = new_image_path
                    job.job_name = change_job_name
                    job.change_job_name = None
                    save_queue()
                
                # If job was completed or failed, change to pending and move it   
                if job.status == "completed" or job.status == "failed":   
                    job.status = "pending"
                    save_queue()

                    move_job_to_top(job)
                #job.thumbnail = create_thumbnail(job, status_change=True)
                queue_table_update, queue_display_update = mark_job_pending(job)
                break
        
        return update_queue_table(), update_queue_display(), gr.update(visible=False), gr.update(visible=True)  # Hide edit group, show table
    except Exception as e:
        alert_print(f"Error editing job: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display(), gr.update(visible=False), gr.update(visible=True)  # Hide edit group, show table

def delete_completed_jobs():
    """Delete all completed jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "completed"]
    save_queue()
    counts = count_jobs_by_status()
    return update_queue_table(), update_queue_display()

def delete_pending_jobs():
    """Delete all pending jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "pending"]
    save_queue()
    counts = count_jobs_by_status()
    return update_queue_table(), update_queue_display()

def delete_failed_jobs():
    """Delete all failed jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "failed"]
    save_queue()
    counts = count_jobs_by_status()
    return update_queue_table(), update_queue_display()

def hide_edit_window():
    #Hide the edit window without saving changes
    return (
        gr.update(visible=False),  # edit_job_group
        gr.update(visible=True),   # edit_job_table_group
        gr.update(visible=True)   # update_queue_table
    )

def save_system_settings(new_outputs_folder, new_job_history_folder, new_debug_mode, new_keep_completed_job, new_prefix_timestamp, new_prefix_source_image_name, new_remove_hexid_suffix):
    """Save system settings to settings.ini and update runtime values"""
    try:
        # Declare globals first
        global job_history_folder, outputs_folder, debug_mode, keep_completed_job, prefix_timestamp, prefix_source_image_name, remove_hexid_suffix
        
        config = load_settings()
    
        if 'System Defaults' not in config:
            config['System Defaults'] = {}

        # Update the settings
        config['System Defaults']['OUTPUTS_FOLDER'] = repr(new_outputs_folder)
        config['System Defaults']['JOB_HISTORY_FOLDER'] = repr(new_job_history_folder)
        config['System Defaults']['DEBUG_MODE'] = repr(new_debug_mode)
        config['System Defaults']['KEEP_COMPLETED_JOB'] = repr(new_keep_completed_job)
        config['System Defaults']['PREFIX_TIMESTAMP'] = repr(new_prefix_timestamp)
        config['System Defaults']['PREFIX_SOURCE_IMAGE_NAME'] = repr(new_prefix_source_image_name)
        config['System Defaults']['REMOVE_HEXID_SUFFIX'] = repr(new_remove_hexid_suffix)
    
        # Save to file
        save_settings(config)
        
            # Update Config object
        Config.OUTPUTS_FOLDER = new_outputs_folder
        Config.JOB_HISTORY_FOLDER = new_job_history_folder
        Config.DEBUG_MODE = new_debug_mode
        Config.KEEP_COMPLETED_JOB = new_keep_completed_job
        Config.PREFIX_TIMESTAMP = new_prefix_timestamp
        Config.PREFIX_SOURCE_IMAGE_NAME = new_prefix_source_image_name
        Config.REMOVE_HEXID_SUFFIX = new_remove_hexid_suffix
            
        # Update global variables
        job_history_folder = new_job_history_folder  # Fixed: was incorrectly set to outputs_folder
        outputs_folder = new_outputs_folder
        debug_mode = new_debug_mode
        keep_completed_job = new_keep_completed_job
        prefix_timestamp = new_prefix_timestamp
        prefix_source_image_name = new_prefix_source_image_name
        remove_hexid_suffix = new_remove_hexid_suffix
        
        # Create directories if they don't exist
        os.makedirs(outputs_folder, exist_ok=True)
        os.makedirs(job_history_folder, exist_ok=True)
        
        return "System settings saved and runtime values updated! New directories created if needed."
    except Exception as e:
        alert_print(f"Error saving system settings: {str(e)}")
        return f"Error saving system settings: {str(e)}"

def restore_system_settings():
    """Restore system settings to original defaults"""
    try:
        # Get original defaults
        defaults = Config.get_original_defaults()
    
        # Load current config
        config = load_settings()
        
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
                
        # Update with original values - format consistently
        for key, value in defaults.items():
            if not key.startswith('DEFAULT_'):  # Only process system settings
                if isinstance(value, str):
                    config['System Defaults'][key] = repr(value)  # Use repr to properly quote strings
                elif isinstance(value, bool):
                    config['System Defaults'][key] = str(value)  # Booleans as 'True' or 'False'
                elif isinstance(value, (int, float)):
                    config['System Defaults'][key] = str(value)  # Numbers as plain strings
                else:
                    config['System Defaults'][key] = repr(value)  # Default to repr for other types
        
        # Save to file
        save_settings(config)
        
        # Update Config object with system settings
        Config.OUTPUTS_FOLDER = defaults['OUTPUTS_FOLDER']
        Config.JOB_HISTORY_FOLDER = defaults['JOB_HISTORY_FOLDER']
        Config.DEBUG_MODE = bool(defaults['DEBUG_MODE'])
        Config.KEEP_COMPLETED_JOB = bool(defaults['KEEP_COMPLETED_JOB'])
        Config.PREFIX_TIMESTAMP = bool(defaults['PREFIX_TIMESTAMP'])
        Config.PREFIX_SOURCE_IMAGE_NAME = bool(defaults['PREFIX_SOURCE_IMAGE_NAME'])
        Config.REMOVE_HEXID_SUFFIX = bool(defaults['REMOVE_HEXID_SUFFIX'])
        
        # Update global variables
        global job_history_folder, outputs_folder, debug_mode, keep_completed_job, prefix_timestamp, prefix_source_image_name, remove_hexid_suffix
        job_history_folder = defaults['JOB_HISTORY_FOLDER']
        outputs_folder = defaults['OUTPUTS_FOLDER']
        debug_mode = bool(defaults['DEBUG_MODE'])
        keep_completed_job = bool(defaults['KEEP_COMPLETED_JOB'])
        prefix_timestamp = bool(defaults['PREFIX_TIMESTAMP'])
        prefix_source_image_name = bool(defaults['PREFIX_SOURCE_IMAGE_NAME'])
        remove_hexid_suffix = bool(defaults['REMOVE_HEXID_SUFFIX'])
        
        # Create directories if they don't exist
        os.makedirs(outputs_folder, exist_ok=True)
        os.makedirs(job_history_folder, exist_ok=True)
        
        # Return values to update UI - ensure proper types
        return (
            str(defaults['OUTPUTS_FOLDER']),  # string
            str(defaults['JOB_HISTORY_FOLDER']),  # string
            bool(defaults['DEBUG_MODE']),  # bool
            bool(defaults['KEEP_COMPLETED_JOB']),  # bool
            bool(defaults['PREFIX_TIMESTAMP']),  # bool
            bool(defaults['PREFIX_SOURCE_IMAGE_NAME']),  # bool
            bool(defaults['REMOVE_HEXID_SUFFIX']),  # bool
            gr.update(value="System settings restored and runtime values updated! New directories created if needed when job is processed.")  # Use gr.update for Markdown
        )
    except Exception as e:
        alert_print(f"Error restoring system settings: {str(e)}")
        return None, None, False, False, False, gr.update(value=f"Error restoring system settings: {str(e)}")




def set_all_models_as_default(transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, image2image_model):
    """Set all models as default at once"""
    try:
        # Check if all models are downloaded
        models = [transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, image2image_model]
        model_types = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model']
        
        for model, model_type in zip(models, model_types):
            if not model.startswith('LOCAL-'):
                return f"Cannot set {model_type} as default - model must be downloaded first", None, None, None, None, None, None, None, None
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update all models
        success_messages = []
        for model, model_type in zip(models, model_types):
            actual_model = Config.model_name_mapping.get(model, model.replace('LOCAL-', '').replace(' - CURRENT DEFAULT MODEL', ''))
            config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(actual_model)
            setattr(Config, f'DEFAULT_{model_type.upper()}', actual_model)
            success_messages.append(f"{model_type}: {actual_model}")
            
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        return (
            "All models set as default successfully:\n" + "\n".join(success_messages) + "\n\nRestart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder'],
            models['image2image_model']
        )
    except Exception as e:
        alert_print(f"Error setting models as default: {str(e)}")
        return f"Error setting models as default: {str(e)}", None, None, None, None, None, None, None, None

def restore_all_model_defaults():
    """Restore all models to their original defaults at once"""
    try:
        # Get original defaults
        original_defaults = Config.get_original_defaults()
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update all model defaults
        success_messages = []
        model_types = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'feature_extractor', 'image_encoder', 'image2image_model']
        
        for model_type in model_types:
            original_value = original_defaults[f'DEFAULT_{model_type.upper()}']
            config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(original_value)
            setattr(Config, f'DEFAULT_{model_type.upper()}', original_value)
            success_messages.append(f"{model_type}: {original_value}")
            
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        return (
            "All models restored to original defaults:\n" + "\n".join(success_messages) + "\n\nRestart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder'],
            models['image2image_model']
        )
    except Exception as e:
        alert_print(f"Error restoring model defaults: {str(e)}")
        return f"Error restoring model defaults: {str(e)}", None, None, None, None, None, None, None, None
        

def enable_edit_metadata_button(file_path):
    """Enable the edit metadata button if the file has metadata"""
    if not file_path:
        return gr.update(visible=False, interactive=False)
    
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.png':
            # For PNG files, check image metadata
            with Image.open(file_path) as img:
                metadata = img.info
                has_metadata = any(field in metadata for field in ["prompt", "video_length", "seed"])

        elif ext == '.mp4':
            # For MP4 files, check ffprobe metadata
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    mp4_data = json.loads(result.stdout)
                    metadata = mp4_data.get('format', {}).get('tags', {})
                    
                    # Check both individual fields and Comments field
                    has_metadata = False
                    
                    # Check individual fields
                    if any(field in metadata for field in ["prompt", "video_length", "seed"]):
                        has_metadata = True
                    # Check Comments field for required fields
                    elif "Comments" in metadata:
                        comment = metadata["Comments"]
                        if any(field in comment for field in ["Prompt:", "Length:", "Seed:"]):
                            has_metadata = True
                            
                    debug_print(f"MP4 metadata found: {has_metadata}")
                    debug_print(f"MP4 metadata fields: {list(metadata.keys())}")
                    if "Comments" in metadata:
                        debug_print(f"MP4 Comments: {metadata['Comments']}")
                else:
                    debug_print(f"ffprobe failed with return code {result.returncode}")
                    has_metadata = False
            except Exception as e:
                debug_print(f"Error reading MP4 metadata: {str(e)}")
                has_metadata = False
        else:
            has_metadata = False
            
        debug_print(f"File {file_path} has metadata: {has_metadata}")
        return gr.update(visible=True, interactive=has_metadata)
    except Exception as e:
        debug_print(f"Error checking metadata: {str(e)}")
        return gr.update(visible=False, interactive=False)



def prepare_metadata_job_edit(file_path):
    if not file_path:
        return (
            gr.update(visible=False),  # edit_group
            "", # prompt
            "", # n_prompt
            gr.update(value=Config.DEFAULT_LORA_MODEL, choices=lora_choices, interactive=True), # lora_model
            Config.DEFAULT_LORA_WEIGHT, # lora_weight
            Config.DEFAULT_VIDEO_LENGTH, # video_length
            "", # job_name
            Config.OUTPUTS_FOLDER, # outputs_folder
            Config.JOB_HISTORY_FOLDER, # job_history_folder
            Config.DEFAULT_USE_TEACACHE, # use_teacache
            Config.DEFAULT_SEED, # seed
            Config.DEFAULT_STEPS, # steps
            Config.DEFAULT_CFG, # cfg
            Config.DEFAULT_GS, # gs
            Config.DEFAULT_RS, # rs
            Config.DEFAULT_IMAGE_STRENGTH, # image_strength
            Config.DEFAULT_MP4_CRF, # mp4_crf
            Config.DEFAULT_GPU_MEMORY, # gpu_memory
            Config.KEEP_COMPLETED_JOB, # keep_completed_job
            Config.DEFAULT_KEEP_TEMP_PNG, # keep_temp_png
        )
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Initialize metadata dictionary
        metadata = {}
        
        if ext == '.png':
            # Handle PNG file
            temp_input = os.path.join(temp_queue_images, "temp_input.png")
            # Retry logic for copying the file
            for attempt in range(5):
                try:
                    shutil.copy2(file_path, temp_input)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)

            # Retry logic for opening the image
            for attempt in range(5):
                try:
                    with Image.open(temp_input) as orig_img:
                        metadata = orig_img.info.copy()  # Make a copy of the metadata

                        # Check if image has required metadata
                        required_fields = ["prompt", "video_length", "seed"]
                        has_metadata = any(field in metadata for field in required_fields)
                        debug_print(f"Raw PNG metadata: {metadata}")
                        debug_print(f"PNG metadata keys found: {list(metadata.keys())}")
                        if not has_metadata:
                            debug_print(f"Image lacks required metadata fields: {file_path}")
                            if os.path.exists(temp_input):
                                os.remove(temp_input)
                            return (
                                gr.update(visible=False),  # edit_group
                                "", # prompt
                                "", # n_prompt
                                gr.update(value=Config.DEFAULT_LORA_MODEL, choices=lora_choices, interactive=True), # lora_model
                                Config.DEFAULT_LORA_WEIGHT, # lora_weight
                                Config.DEFAULT_VIDEO_LENGTH, # video_length
                                "", # job_name
                                Config.OUTPUTS_FOLDER, # outputs_folder
                                Config.JOB_HISTORY_FOLDER, # job_history_folder
                                Config.DEFAULT_USE_TEACACHE, # use_teacache
                                Config.DEFAULT_SEED, # seed
                                Config.DEFAULT_STEPS, # steps
                                Config.DEFAULT_CFG, # cfg
                                Config.DEFAULT_GS, # gs
                                Config.DEFAULT_RS, # rs
                                Config.DEFAULT_IMAGE_STRENGTH, # image_strength
                                Config.DEFAULT_MP4_CRF, # mp4_crf
                                Config.DEFAULT_GPU_MEMORY, # gpu_memory
                                Config.KEEP_COMPLETED_JOB, # keep_completed_job
                                Config.DEFAULT_KEEP_TEMP_PNG, # keep_temp_png
                            )

                        # Convert to RGB if needed and make a copy
                        if orig_img.mode == 'RGBA':
                            img = orig_img.convert('RGB')
                        else:
                            img = orig_img.copy()
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)

            # Save image with metadata preserved
            temp_path = os.path.join(temp_queue_images, "meta_loaded_image.png")
            pnginfo = PngInfo()
            for key, value in metadata.items():
                try:
                    pnginfo.add_text(key, str(value))
                except Exception as e:
                    alert_print(f"Error adding metadata key {key}: {str(e)}")
                    continue

            try:
                img.save(temp_path, pnginfo=pnginfo)
            finally:
                img.close()
                # Clean up temporary input file
                if os.path.exists(temp_input):
                    os.remove(temp_input)

            # Extract metadata values
            prompt = metadata.get('prompt', '')
            n_prompt = metadata.get('n_prompt', '')
            lora_model = metadata.get('lora_model', Config.DEFAULT_LORA_MODEL)
            lora_weight = float(metadata.get('lora_weight', Config.DEFAULT_LORA_WEIGHT))
            video_length = float(metadata.get('video_length', Config.DEFAULT_VIDEO_LENGTH))
            job_name = metadata.get('job_name', '')
            use_teacache = metadata.get('use_teacache', str(Config.DEFAULT_USE_TEACACHE)).lower() == 'true'
            seed = int(float(metadata.get('seed', Config.DEFAULT_SEED)))
            steps = int(float(metadata.get('steps', Config.DEFAULT_STEPS)))
            cfg = float(metadata.get('cfg', Config.DEFAULT_CFG))
            gs = float(metadata.get('gs', Config.DEFAULT_GS))
            rs = float(metadata.get('rs', Config.DEFAULT_RS))
            image_strength = float(metadata.get('image_strength', Config.DEFAULT_IMAGE_STRENGTH))
            mp4_crf = float(metadata.get('mp4_crf', Config.DEFAULT_MP4_CRF))
            gpu_memory = float(metadata.get('gpu_memory', Config.DEFAULT_GPU_MEMORY))

            return (
                gr.update(visible=True),  # edit_group
                prompt,
                n_prompt,
                gr.update(value=lora_model, choices=lora_choices, interactive=True),
                lora_weight,
                video_length,
                job_name,
                Config.OUTPUTS_FOLDER,
                Config.JOB_HISTORY_FOLDER,
                use_teacache,
                seed,
                steps,
                cfg,
                gs,
                rs,
                image_strength,
                mp4_crf,
                gpu_memory,
                Config.KEEP_COMPLETED_JOB,
                Config.DEFAULT_KEEP_TEMP_PNG,
            )


        elif ext == '.mp4':
            # Use ffprobe to get metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            metadata = json.loads(result.stdout)['format']['tags']

            debug_print(f"[DEBUG] MP4 metadata found: {bool(metadata)}")
            debug_print(f"[DEBUG] MP4 metadata fields: {list(metadata.keys())}")
            debug_print(f"[DEBUG] MP4 Comments: {metadata.get('Comments', '')}")

            # Extract metadata from individual fields first
            prompt = metadata.get('prompt', '')
            n_prompt = metadata.get('n_prompt', '')
            lora_model = metadata.get('lora_model', Config.DEFAULT_LORA_MODEL)
            lora_weight = float(metadata.get('lora_weight', Config.DEFAULT_LORA_WEIGHT))
            video_length = float(metadata.get('video_length', Config.DEFAULT_VIDEO_LENGTH))
            job_name = metadata.get('job_name', '')
            use_teacache = metadata.get('use_teacache', str(Config.DEFAULT_USE_TEACACHE)).lower() == 'true'
            seed = int(float(metadata.get('seed', Config.DEFAULT_SEED)))
            steps = int(float(metadata.get('steps', Config.DEFAULT_STEPS)))
            cfg = float(metadata.get('cfg', Config.DEFAULT_CFG))
            gs = float(metadata.get('gs', Config.DEFAULT_GS))
            rs = float(metadata.get('rs', Config.DEFAULT_RS))
            image_strength = float(metadata.get('image_strength', Config.DEFAULT_IMAGE_STRENGTH))
            mp4_crf = float(metadata.get('mp4_crf', Config.DEFAULT_MP4_CRF))
            gpu_memory = float(metadata.get('gpu_memory', Config.DEFAULT_GPU_MEMORY))
            source_name = metadata.get('source_name', '')

            # If any field is missing, try to get it from Comments
            if not all([prompt, n_prompt, lora_model, lora_weight, video_length, job_name, use_teacache, seed, steps, cfg, gs, rs, image_strength, mp4_crf, gpu_memory]):
                comments = metadata.get('Comments', '')
                if comments:
                    for line in comments.split(';'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'prompt' and not prompt:
                                prompt = value
                            elif key == 'n_prompt' and not n_prompt:
                                n_prompt = value
                            elif key == 'lora_model' and not lora_model:
                                lora_model = value
                            elif key == 'lora_weight' and not lora_weight:
                                lora_weight = float(value)
                            elif key == 'video_length' and not video_length:
                                video_length = float(value)
                            elif key == 'job_name' and not job_name:
                                job_name = value
                            elif key == 'use_teacache' and not use_teacache:
                                use_teacache = value.lower() == 'true'
                            elif key == 'seed' and not seed:
                                seed = int(float(value))
                            elif key == 'steps' and not steps:
                                steps = int(float(value))
                            elif key == 'cfg' and not cfg:
                                cfg = float(value)
                            elif key == 'gs' and not gs:
                                gs = float(value)
                            elif key == 'rs' and not rs:
                                rs = float(value)
                            elif key == 'image_strength' and not image_strength:
                                image_strength = float(value)
                            elif key == 'mp4_crf' and not mp4_crf:
                                mp4_crf = float(value)
                            elif key == 'gpu_memory' and not gpu_memory:
                                gpu_memory = float(value)
                            elif key == 'source_name' and not source_name:
                                source_name = value

            # Extract first frame and save as PNG
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create PIL Image
                image = Image.fromarray(frame_rgb)
                
                # Create PNG info with metadata
                pnginfo = PngInfo()
                pnginfo.add_text('prompt', prompt)
                pnginfo.add_text('n_prompt', n_prompt)
                pnginfo.add_text('lora_model', lora_model)
                pnginfo.add_text('lora_weight', str(lora_weight))
                pnginfo.add_text('video_length', str(video_length))
                pnginfo.add_text('job_name', job_name)
                pnginfo.add_text('use_teacache', str(use_teacache))
                pnginfo.add_text('seed', str(seed))
                pnginfo.add_text('steps', str(steps))
                pnginfo.add_text('cfg', str(cfg))
                pnginfo.add_text('gs', str(gs))
                pnginfo.add_text('rs', str(rs))
                pnginfo.add_text('image_strength', str(image_strength))
                pnginfo.add_text('mp4_crf', str(mp4_crf))
                pnginfo.add_text('gpu_memory', str(gpu_memory))
                pnginfo.add_text('source_name', source_name)
                
                # Save as PNG
                temp_file = os.path.join(temp_queue_images, "meta_loaded_image.png")
                image.save(temp_file, "PNG", pnginfo=pnginfo)
                
                cap.release()
                
                return (
                    gr.update(visible=True),  # edit_group
                    prompt,
                    n_prompt,
                    gr.update(value=lora_model, choices=lora_choices, interactive=True),
                    lora_weight,
                    video_length,
                    job_name,
                    Config.OUTPUTS_FOLDER,
                    Config.JOB_HISTORY_FOLDER,
                    use_teacache,
                    seed,
                    steps,
                    cfg,
                    gs,
                    rs,
                    image_strength,
                    mp4_crf,
                    gpu_memory,
                    Config.KEEP_COMPLETED_JOB,
                    Config.DEFAULT_KEEP_TEMP_PNG,
                )
            else:
                alert_print("Failed to read video frame")
                return (
                    gr.update(visible=False),  # edit_group
                    "", # prompt
                    "", # n_prompt
                    gr.update(value=Config.DEFAULT_LORA_MODEL, choices=lora_choices, interactive=True), # lora_model
                    Config.DEFAULT_LORA_WEIGHT, # lora_weight
                    Config.DEFAULT_VIDEO_LENGTH, # video_length
                    "", # job_name
                    Config.OUTPUTS_FOLDER, # outputs_folder
                    Config.JOB_HISTORY_FOLDER, # job_history_folder
                    Config.DEFAULT_USE_TEACACHE, # use_teacache
                    Config.DEFAULT_SEED, # seed
                    Config.DEFAULT_STEPS, # steps
                    Config.DEFAULT_CFG, # cfg
                    Config.DEFAULT_GS, # gs
                    Config.DEFAULT_RS, # rs
                    Config.DEFAULT_IMAGE_STRENGTH, # image_strength
                    Config.DEFAULT_MP4_CRF, # mp4_crf
                    Config.DEFAULT_GPU_MEMORY, # gpu_memory
                    Config.KEEP_COMPLETED_JOB, # keep_completed_job
                    Config.DEFAULT_KEEP_TEMP_PNG, # keep_temp_png
                )
        else:
            alert_print(f"Unsupported file type: {ext}")
            return (
                gr.update(visible=False),  # edit_group
                "", # prompt
                "", # n_prompt
                gr.update(value=Config.DEFAULT_LORA_MODEL, choices=lora_choices, interactive=True), # lora_model
                Config.DEFAULT_LORA_WEIGHT, # lora_weight
                Config.DEFAULT_VIDEO_LENGTH, # video_length
                "", # job_name
                Config.OUTPUTS_FOLDER, # outputs_folder
                Config.JOB_HISTORY_FOLDER, # job_history_folder
                Config.DEFAULT_USE_TEACACHE, # use_teacache
                Config.DEFAULT_SEED, # seed
                Config.DEFAULT_STEPS, # steps
                Config.DEFAULT_CFG, # cfg
                Config.DEFAULT_GS, # gs
                Config.DEFAULT_RS, # rs
                Config.DEFAULT_IMAGE_STRENGTH, # image_strength
                Config.DEFAULT_MP4_CRF, # mp4_crf
                Config.DEFAULT_GPU_MEMORY, # gpu_memory
                Config.KEEP_COMPLETED_JOB, # keep_completed_job
                Config.DEFAULT_KEEP_TEMP_PNG, # keep_temp_png
            )
    except Exception as e:
        alert_print(f"Error processing file: {str(e)}")
        return (
            gr.update(visible=False),  # edit_group
            "", # prompt
            "", # n_prompt
            gr.update(value=Config.DEFAULT_LORA_MODEL, choices=lora_choices, interactive=True), # lora_model
            Config.DEFAULT_LORA_WEIGHT, # lora_weight
            Config.DEFAULT_VIDEO_LENGTH, # video_length
            "", # job_name
            Config.OUTPUTS_FOLDER, # outputs_folder
            Config.JOB_HISTORY_FOLDER, # job_history_folder
            Config.DEFAULT_USE_TEACACHE, # use_teacache
            Config.DEFAULT_SEED, # seed
            Config.DEFAULT_STEPS, # steps
            Config.DEFAULT_CFG, # cfg
            Config.DEFAULT_GS, # gs
            Config.DEFAULT_RS, # rs
            Config.DEFAULT_IMAGE_STRENGTH, # image_strength
            Config.DEFAULT_MP4_CRF, # mp4_crf
            Config.DEFAULT_GPU_MEMORY, # gpu_memory
            Config.KEEP_COMPLETED_JOB, # keep_completed_job
            Config.DEFAULT_KEEP_TEMP_PNG, # keep_temp_png
        )


def save_loaded_job(
    prompt,
    n_prompt,
    lora_model,
    lora_weight,
    video_length,
    job_name,
    outputs_folder,
    job_history_folder,
    use_teacache,
    seed,
    steps,
    cfg,
    gs,
    rs,
    image_strength,
    mp4_crf,
    gpu_memory,
    keep_completed_job,
    keep_temp_png
):
    temp_path = os.path.join(temp_queue_images, "meta_loaded_image.png")
    if os.path.exists(temp_path):
        img = np.array(Image.open(temp_path))
        job_name = add_to_queue(
            prompt=prompt,
            n_prompt=n_prompt,
            lora_model=lora_model,
            lora_weight=lora_weight,
            input_image=img,
            video_length=video_length,
            job_name=job_name,
            create_job_outputs_folder=outputs_folder,
            create_job_history_folder=job_history_folder,
            use_teacache=use_teacache,
            seed=seed,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            image_strength=image_strength,
            mp4_crf=mp4_crf,
            gpu_memory=gpu_memory,
            create_job_keep_completed_job=keep_completed_job,
            keep_temp_png=keep_temp_png,
            status="pending"
        )
        save_queue()
        try:
            os.remove(temp_path)
        except:
            pass
    return (
        gr.update(visible=False),  # edit_group
        None,  # clear file upload
        gr.update(visible=False),  # load_metadata_job_group
        update_queue_table(),
        update_queue_display()
    )

def cancel_loaded_job():
    temp_path = os.path.join(temp_queue_images, "meta_loaded_image.png")
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except:
        pass
    return (
        gr.update(visible=False),  # edit_group
        None,  # clear file upload
        gr.update(visible=False)  # load_metadata_job_group
    )

block = gr.Blocks(css=css).queue()
with block:
# GRADIO UI
    gr.Markdown('# FramePack (QueueItUp version)')
    with gr.Tabs() as tabs:
        with gr.Tab("Framepack_QueueItUp"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Gallery(
                        label="Image (adding multiple images will create a jobs for each, leaving image blank will be prompt Text 2 Video",
                        height=500,
                        columns=4,
                        object_fit="contain",
                        elem_classes=["input-gallery"],
                        show_label=True,
                        allow_preview=False,  # Disable built-in preview
                        show_download_button=True,
                        container=True
                    )
                    with gr.Group():    
                        queue_button = gr.Button(value="Add to Queue", interactive=True, elem_id="queue_button")
                        job_name = gr.Textbox(label="Job Name (optional prefix)", interactive=True, value=Config.DEFAULT_JOB_NAME, info=f"Optional prefix name for this job (comming soon you can enter source_image_filename or date_time) or blank will defaut to Job-")


                    with gr.Accordion("Quick List", open=False):
                        save_prompt_button = gr.Button("Save Prompt and settings to the Quick List as job name")
                        quick_list = gr.Dropdown(
                            label="Quick List",
                            choices=[item['job_name'] for item in quick_prompts],  # <-- Changed
                            value=quick_prompts[0]['job_name'] if quick_prompts else None,  # <-- Changed
                            allow_custom_value=True
                        )
                        delete_prompt_button = gr.Button("Delete Selected Prompt from the Quick List")

                    with gr.Group():   

                        prompt = gr.Textbox(label="Prompt", value=Config.DEFAULT_PROMPT)
                        n_prompt = gr.Textbox(label="Negative Prompt", value=Config.DEFAULT_N_PROMPT, visible=True)
                        video_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=Config.DEFAULT_VIDEO_LENGTH, step=1.0)

                    with gr.Accordion("LoRA Settings", open=False):
                        lora_choices = ["None"]
                        try:
                            lora_path = os.path.join(os.getcwd(), "loras")
                            if os.path.exists(lora_path):
                                files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
                                files.sort()
                                lora_choices += files
                        except Exception as e:
                            pass

                        lora_model = gr.Dropdown(
                            label="LoRA Model",
                            choices=lora_choices,
                            value=Config.DEFAULT_LORA_MODEL,
                            info="Select a LoRA .safetensors file to use (or None for no LoRA)"
                        )
                        lora_weight = gr.Slider(
                            label="LoRA Strength",
                            minimum=0,
                            maximum=10,
                            value=Config.DEFAULT_LORA_WEIGHT,
                            step=0.1,
                            info="Strength of the LoRA. 0 disables it."
                        )


                    
         

                    with gr.Accordion("Job Settings", open=False):
                        use_teacache = gr.Checkbox(label='Use TeaCache', value=Config.DEFAULT_USE_TEACACHE, info='Faster speed, but often makes hands and fingers slightly worse.')
                        seed = gr.Number(label="Seed use -1 to create random seed for job", value=Config.DEFAULT_SEED, precision=0)
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=Config.DEFAULT_STEPS, step=1, info='Changing this value is not recommended.')
                        cfg = gr.Slider(label="CFG Scale (aka CFG)", minimum=1.0, maximum=32.0, value=Config.DEFAULT_CFG, step=0.01, visible=False)  # Should not change
                        gs = gr.Slider(label="Distilled CFG Scale (aka GS)", minimum=1.0, maximum=32.0, value=Config.DEFAULT_GS, step=0.01, info='Changing this value is not recommended.')
                        #oringinallynot visible
                        rs = gr.Slider(label="CFG Re-Scale (aka RS)", minimum=0.0, maximum=1.0, value=Config.DEFAULT_RS, step=0.01, visible=True)  # Should not change
                        image_strength = gr.Slider(label="Image Strength", minimum=0.1, maximum=1.0, value=Config.DEFAULT_IMAGE_STRENGTH, step=0.01, info="default is 1 uses 100% use of the image, 0 is basically text to video")
                        mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=Config.DEFAULT_MP4_CRF, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                        gpu_memory = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=Config.DEFAULT_GPU_MEMORY, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")


                    with gr.Accordion("Files and Folders Names", open=False):
                        create_job_outputs_folder = gr.Textbox(label="Job Outputs Folder", value=Config.OUTPUTS_FOLDER, info="The path Where the output video for this job will be saved, the default directory is displayed below, optional to change, but a bad path will cause an error.")
                        create_job_history_folder = gr.Textbox(label="Job History Folder", value=Config.JOB_HISTORY_FOLDER, info="The path Where the job history will be saved, the default directory is displayed below, optional to change, but a bad path will cause an error.")
           
                    
                    with gr.Accordion("Post Job Cleanup", open=False):
                        create_job_keep_completed_job = gr.Checkbox(label="Keep this Job when completed", value=Config.KEEP_COMPLETED_JOB, info="If checked, when this job is completed it will stay in the queue as status 'completed' so you can edit them and run them again")
                        keep_temp_png = gr.Checkbox(label="Keep temp PNG file", value=Config.DEFAULT_KEEP_TEMP_PNG, info="If checked, temporary job history PNG file will not be deleted after job is processed")
                    with gr.Accordion("Save and Restore Job Defaults", open=False):
                        save_job_defaults_button = gr.Button(value="Save current job settings as Defaults", interactive=True, elem_id="save_job_defaults_button")
                        restore_job_defaults_button = gr.Button(value="Restore Original job settings", interactive=True, elem_id="restore_job_defaults_button")

                with gr.Column():
                    with gr.Row():
                        start_button = gr.Button(value="Start Queued Jobs", interactive=True, elem_id="start_button")
                    with gr.Row():                
                        abort_button = gr.Button(value="Abort and keep output video", interactive=False, elem_id="abort_button")
                        abort_delete_button = gr.Button(value="Abort and delete output video", interactive=False, elem_id="abort_delete_button")
                    with gr.Accordion("Output preview window:", open=True):
                        preview_image = gr.Image(label="Latents", visible=False, show_download_button=False, show_share_button=False, interactive=False)
                        progress_desc1 = gr.Markdown('', elem_classes=['no-generating-animation', 'progress-desc'], elem_id="progress_desc1")  # Job Step progress text
                        progress_bar1 = gr.HTML('', elem_classes=['no-generating-animation', 'progress-bar'], elem_id="progress_bar1")  # Job Step progress bar
                        gr.Markdown("Note: Video Previews and progress will appear when Image to Video jobs are started.")

                        result_video = gr.Video(label="Video Output Display", autoplay=True, show_share_button=False, height=400, loop=True, visible=False)
                        progress_desc2 = gr.Markdown('', elem_classes=['no-generating-animation', 'progress-desc'], elem_id="progress_desc2")  # Job progress text
                        progress_bar2 = gr.HTML('', elem_classes=['no-generating-animation', 'progress-bar'], elem_id="progress_bar2")  # Job progress bar


                    with gr.Accordion("Job Queue Gallery", open=True):

                        queue_display = gr.Gallery(label=f"{current_counts['pending']} Jobs pending in the Queue", show_label=True, show_download_button=False, show_share_button=False, columns=5, object_fit="contain", elem_classes=["queue-gallery"], allow_preview=True, container=True)


        with gr.Tab("Edit jobs in the Queue"):
            with gr.Row():
                metadata_job_button = gr.Button(value="Upload a metadata Image from job History", interactive=True)
            # Add file upload section
            with gr.Group(visible=False) as load_metadata_job_group:
                load_metadata_job = gr.File(
                    file_types=[".png", ".mp4"],
                    label="Upload a file with job metadata (PNG, JSON, or MP4)",
                    height=320
                )
                edit_metadata_job_button = gr.Button("Load metadata from file", visible=False, interactive=False)
            # --- Refactored metadata job edit group ---
            with gr.Group(visible=False) as edit_metadata_job_group:
                with gr.Row():
                    save_edit_metadata_job_button = gr.Button(value="Save Loaded Job")
                    cancel_edit_metadata_job_button = gr.Button(value="Cancel")
                gr.Markdown("<br>")
                edit_metadata_job_name = gr.Textbox(label="Job Name (optional prefix)", value=Config.DEFAULT_JOB_NAME, info=f"Optional prefix name for this job (comming soon you can enter source_image_filename or date_time) or blank will defaut to Job-")
                edit_metadata_job_prompt = gr.Textbox(label="Change Prompt")
                edit_metadata_job_n_prompt = gr.Textbox(label="Change Negative Prompt")
                edit_metadata_job_video_length = gr.Slider(label="Change Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                gr.Markdown("<br>")
                
                

                with gr.Accordion("LoRA Settings", open=False):
                    lora_choices = ["None"]
                    try:
                        lora_path = os.path.join(os.getcwd(), "loras")
                        if os.path.exists(lora_path):
                            files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
                            files.sort()
                            lora_choices += files
                    except Exception as e:
                        pass

                    edit_metadata_job_lora_model = gr.Dropdown(
                        label="LoRA Model",
                        choices=lora_choices,
                        value=Config.DEFAULT_LORA_MODEL,
                        info="Select a LoRA .safetensors file to use (or None for no LoRA)"
                    )
                    edit_metadata_job_lora_weight = gr.Slider(
                        label="LoRA Strength",
                        minimum=0,
                        maximum=10,
                        value=Config.DEFAULT_LORA_WEIGHT,
                        step=0.1,
                        info="Strength of the LoRA. 0 disables it."
                    )


                with gr.Accordion("Job Settings", open=False):
                    edit_metadata_job_use_teacache = gr.Checkbox(label='Change Use TeaCache', value=True)
                    edit_metadata_job_seed = gr.Number(label="Change Seed", value=-1, precision=0)
                    edit_metadata_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                    edit_metadata_job_steps = gr.Slider(label="Change Steps", minimum=1, maximum=100, value=25, step=1)
                    edit_metadata_job_cfg = gr.Slider(label="Change CFG Scale  (aka CFG)", visible=False, minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                    edit_metadata_job_gs = gr.Slider(label="Change Distilled CFG Scale (aka GS)", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                    edit_metadata_job_rs = gr.Slider(label="Change CFG Re-Scale (aka RS)", visible=True, minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                    edit_metadata_job_image_strength = gr.Slider(label="Change Image Strength", minimum=0.1, maximum=1.0, value=Config.DEFAULT_IMAGE_STRENGTH, step=0.01)
                    edit_metadata_job_mp4_crf = gr.Slider(label="Change MP4 Compression", minimum=0, maximum=100, value=16, step=1)
                    edit_metadata_job_gpu_memory = gr.Slider(label="Change GPU Memory Preservation (GB)", minimum=6, maximum=128, value=6, step=0.1)
                gr.Markdown("<br>")
                with gr.Accordion("File and Folder Names", open=False):
                    edit_metadata_job_outputs_folder = gr.Textbox(label="Change Outputs Folder", value=Config.OUTPUTS_FOLDER)
                    edit_metadata_job_history_folder = gr.Textbox(label="Change Job History Folder", value=Config.JOB_HISTORY_FOLDER)
                


                with gr.Accordion("Post Job Cleanup", open=False):
                    edit_metadata_job_keep_completed_job = gr.Checkbox(label="Keep Completed Jobs", value=Config.KEEP_COMPLETED_JOB)
                    edit_metadata_job_keep_temp_png = gr.Checkbox(label="Keep temp PNG", value=False)



# adding edit_job_group
            with gr.Group(visible=False) as edit_job_group:
                with gr.Row():
                    with gr.Column():
                        save_edit_job_button = gr.Button(value="Save Changes")
                    with gr.Column():
                        cancel_edit_job_button = gr.Button(value="Cancel")
                edit_job_name = gr.Textbox(label="Job Name", interactive=False, visible=False)
                change_job_name = gr.Textbox(label="Job Name", interactive=True, visible=True)
                edit_job_prompt = gr.Textbox(label="Change Prompt")
                edit_job_n_prompt = gr.Textbox(label="Change Negative Prompt")
                edit_job_video_length = gr.Slider(label="Change Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                gr.Markdown("<br>") 
                

                with gr.Accordion("LoRA Settings", open=False):
                    lora_choices = ["None"]
                    try:
                        lora_path = os.path.join(os.getcwd(), "loras")
                        if os.path.exists(lora_path):
                            files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
                            files.sort()
                            lora_choices += files
                    except Exception as e:
                        pass

                    edit_job_lora_model = gr.Dropdown(
                        label="LoRA Model",
                        choices=lora_choices,
                        value=Config.DEFAULT_LORA_MODEL,
                        info="Select a LoRA .safetensors file to use (or None for no LoRA)"
                    )
                    edit_job_lora_weight = gr.Slider(
                        label="LoRA Strength",
                        minimum=0,
                        maximum=10,
                        value=Config.DEFAULT_LORA_WEIGHT,
                        step=0.1,
                        info="Strength of the LoRA. 0 disables it."
                    )
                
                
                # with gr.Accordion("LoRA Settings", open=False):
                    # edit_job_lora_model = gr.Dropdown(label="LoRA Model", choices=lora_choices, value=Config.DEFAULT_LORA_MODEL, info="Select a LoRA .safetensors file to use (or None for no LoRA)")
                    # edit_job_lora_weight = gr.Slider(label="LoRA Strength", minimum=0, maximum=2.0, value=Config.DEFAULT_LORA_WEIGHT, step=0.1)
                    
                    
                    
                    
                gr.Markdown("<br>")
                with gr.Accordion("Job Settings", open=False):
                    edit_job_use_teacache = gr.Checkbox(label='Change Use TeaCache', value=True)
                    edit_job_seed = gr.Number(label="Change Seed", value=-1, precision=0)
                    edit_job_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                    edit_job_steps = gr.Slider(label="Change Steps", minimum=1, maximum=100, value=25, step=1)
                    edit_job_cfg = gr.Slider(label="Change CFG Scale (aka CFG)", visible=False, minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                    edit_job_gs = gr.Slider(label="Change Distilled CFG Scale (aka GS)", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                    edit_job_rs = gr.Slider(label="Change CFG Re-Scale (aka RS)", visible=True, minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                    edit_job_image_strength = gr.Slider(label="Change Image Strength", minimum=0.1, maximum=1.0, value=Config.DEFAULT_IMAGE_STRENGTH, step=0.01)  # Add Image Strength slider
                    edit_job_mp4_crf = gr.Slider(label="Change MP4 Compression", minimum=0, maximum=100, value=16, step=1)
                    edit_job_gpu_memory = gr.Slider(label="Change GPU Memory Preservation (GB)", minimum=6, maximum=128, value=6, step=0.1)
                gr.Markdown("<br>")
                with gr.Accordion("File and Folder Names", open=False): 
                    edit_job_outputs_folder = gr.Textbox(label="Change Outputs Folder", value=Config.OUTPUTS_FOLDER)
                    edit_job_history_folder = gr.Textbox(label="Change Job History Folder", value=Config.JOB_HISTORY_FOLDER)
                gr.Markdown("<br>")
                with gr.Accordion("Post Job Cleanup", open=False):
                    edit_job_keep_completed_job = gr.Checkbox(label="Keep Completed Jobs", value=Config.KEEP_COMPLETED_JOB)
                    edit_job_keep_temp_png = gr.Checkbox(label="Keep temp PNG", value=False)


            with gr.Group(visible=True, elem_id="edit_job_table_group") as edit_job_table_group:
                with gr.Row():
                    with gr.Column():
                        with gr.Row():         
                            queue_table = gr.DataFrame(
                                headers=None,
                                datatype=["markdown","str","str","str","str","str","str","str","markdown"],
                                col_count=(9, "fixed"),
                                value=[],
                                interactive=False,
                                visible=True,
                                elem_classes=["gradio-dataframe"], max_height=800
                            )
                        with gr.Row():           
                            delete_completed_button = gr.Button(value="Delete Completed Jobs", interactive=True)
                            delete_pending_button = gr.Button(value="Delete Pending Jobs", interactive=True)
                            delete_failed_button = gr.Button(value="Delete Failed Jobs", interactive=True)
                            delete_all_button = gr.Button(value="Delete All Jobs", interactive=True, variant="stop")

        with gr.Tab("Settings"):
            with gr.Tabs():
                with gr.Tab("System settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Settings")
                            settings_status = gr.Markdown()  # For showing status messages
                            
                            settings_outputs_folder = gr.Textbox(
                                label="Outputs Folder - Default folder where videos are initially saved", 
                                value=Config.OUTPUTS_FOLDER,
                                interactive=False,
                                info="This setting cannot be changed. Videos are initially saved here and then moved to your custom folder if specified in the job settings."
                            )
                            settings_job_history_folder = gr.Textbox(
                                label="Job History Folder (is where the individual job json files and input image are stored with the jobs metadata)", 
                                value=Config.JOB_HISTORY_FOLDER,
                                interactive=False,
                                info="This setting cannot be changed here,  it can be changed on a per job or batch job basis if specified in the job settings or edit settings."
                            )
                            settings_debug_mode = gr.Checkbox(
                                label="Debug Mode", 
                                value=Config.DEBUG_MODE
                            )
                            settings_keep_completed_job = gr.Checkbox(
                                label="Keep Completed Jobs", 
                                value=Config.KEEP_COMPLETED_JOB
                            )
                            settings_prefix_timestamp = gr.Checkbox(
                                label="Prefix Timestamp to Final Output Files",
                                value=Config.PREFIX_TIMESTAMP
                            )
                            settings_prefix_source_image_name = gr.Checkbox(
                                label="Prefix Source Image Name to Final Output Files",
                                value=Config.PREFIX_SOURCE_IMAGE_NAME
                            )
                            settings_remove_hexid_suffix = gr.Checkbox(
                                label="Remove Hex ID from Final Output files Suffix (only if timestamp is enabled)",
                                value=Config.REMOVE_HEXID_SUFFIX
                            )
                            with gr.Row():
                                save_system_button = gr.Button("Save System Settings", variant="primary")
                                restore_system_button = gr.Button("Restore System Defaults", variant="secondary")
                            
                            # Connect the buttons
                            save_system_button.click(
                                fn=save_system_settings,
                                inputs=[
                                    settings_outputs_folder,
                                    settings_job_history_folder,
                                    settings_debug_mode,
                                    settings_keep_completed_job,
                                    settings_prefix_timestamp,
                                    settings_prefix_source_image_name,
                                    settings_remove_hexid_suffix
                                ],
                                outputs=[settings_status]
                            )
                            
                            restore_system_button.click(
                                fn=restore_system_settings,
                                inputs=[],
                                outputs=[
                                    settings_outputs_folder,
                                    settings_job_history_folder,
                                    settings_debug_mode,
                                    settings_keep_completed_job,
                                    settings_prefix_timestamp,
                                    settings_prefix_source_image_name,
                                    settings_remove_hexid_suffix,
                                    settings_status
                                ]
                            )

                with gr.Tab("Model Defaults (BETA not working yet)"):
                    with gr.Row():
                        with gr.Column():      
                            gr.Markdown("### Hugging Face Settings")
                            settings_hf_token = gr.Textbox(
                                label="Hugging Face Token", 
                                value=Config.HF_TOKEN,
                                type="text",
                                info="Enter your Hugging Face token to enable downloading models. Get it from https://huggingface.co/settings/tokens"
                            )
                            save_token_button = gr.Button("Save Token")
                            
                            def save_hf_token(token):
                                # First save to settings.ini
                                config = configparser.ConfigParser()
                                config.read(INI_FILE)
                                
                                if 'System Defaults' not in config:
                                    config['System Defaults'] = {}
                                    
                                config['System Defaults']['hf_token'] = token
                                
                                with open(INI_FILE, 'w') as f:
                                    config.write(f)
                                    
                                # Update the global Config instance
                                global Config
                                if Config._instance is None:
                                    Config._instance = Config.from_settings(config)
                                Config._instance.HF_TOKEN = token
                                
                                # Also update the global Config variable
                                Config.HF_TOKEN = token
                                
                                # Reload settings to ensure everything is in sync
                                Config = Config.from_settings(load_settings())
                                
                                return "Token saved successfully!"
                            
                            save_token_button.click(
                                fn=save_hf_token,
                                inputs=[settings_hf_token],
                                outputs=[gr.Markdown()]
                            )
                            
                            gr.Markdown("choose a default Image 2 Image model for denoising and modifying your starup input image")
                            image2image_model = gr.Dropdown(
                                choices=[d for d in os.listdir(os.path.join(os.environ['HF_HOME'], 'hub')) if os.path.isdir(os.path.join(os.environ['HF_HOME'], 'hub', d))],
                                value=Config.DEFAULT_IMAGE2IMAGE_MODEL,
                                label="Image2Image Model",
                                allow_custom_value=True
                            )

                            def on_model_select(model_name):
                                if model_name:
                                    hub_path = os.path.join(os.environ['HF_HOME'], 'hub')
                                    model_path = os.path.join(hub_path, model_name)
                                    
                                    if not os.path.exists(model_path):
                                        try:
                                            snapshot_download(repo_id=model_name, local_dir=model_path)
                                            debug_print(f"Downloaded model {model_name} to {model_path}")
                                            # Update dropdown choices
                                            image2image_model.choices = [d for d in os.listdir(hub_path) if os.path.isdir(os.path.join(hub_path, d))]
                                        except Exception as e:
                                            alert_print(f"Error downloading model {model_name}: {str(e)}")
                                    return model_name
                                return Config.IMAGE2IMAGE_MODEL

                            image2image_model.change(fn=on_model_select, inputs=[image2image_model], outputs=[image2image_model])



    restore_job_defaults_button.click(
        fn=lambda: restore_job_defaults(),
        inputs=[],
        outputs=[
            prompt, n_prompt, lora_model, lora_weight, use_teacache, seed, job_name, video_length, steps,
            cfg, gs, rs, image_strength, mp4_crf, gpu_memory,
            keep_temp_png
        ]
    )



    
    save_job_defaults_button.click(
        fn=save_job_defaults_from_ui,
        inputs=[
            prompt,
            n_prompt,
            lora_model,
            lora_weight,
            video_length,
            job_name,
            use_teacache,
            seed,
            steps,
            cfg,
            gs,
            rs,
            image_strength,
            mp4_crf,
            gpu_memory,
            keep_temp_png
        ],
        outputs=[
            prompt,
            n_prompt,
            lora_model,
            lora_weight,
            video_length,
            job_name,
            use_teacache,
            seed,
            steps,
            cfg,
            gs,
            rs,
            image_strength,
            mp4_crf,
            gpu_memory,
            keep_temp_png
        ]
    )



    # Connect UI elements
    save_prompt_button.click(
        fn=save_quick_prompt,
        inputs=[
            prompt,
            n_prompt,
            lora_model,
            lora_weight,
            video_length,
            job_name,
            use_teacache,
            seed,
            steps,
            cfg,
            gs,
            rs,
            image_strength,
            mp4_crf,
            gpu_memory,
            create_job_outputs_folder,
            create_job_history_folder,
            create_job_keep_completed_job,
            keep_temp_png
        ],
        outputs=[
            prompt,
            n_prompt,
            quick_list,
            lora_model,
            lora_weight,
            video_length,
            job_name,
            create_job_outputs_folder,
            create_job_history_folder,
            use_teacache,
            seed,
            steps,
            cfg,
            gs,
            rs,
            image_strength,
            mp4_crf,
            gpu_memory,
            create_job_keep_completed_job,
            keep_temp_png
        ],
        queue=False
    )
    delete_prompt_button.click(
        delete_quick_prompt,
        inputs=[quick_list],
        outputs=[prompt, n_prompt, quick_list, lora_model, lora_weight, video_length, job_name, gs, steps, use_teacache, seed, cfg, rs, image_strength, mp4_crf , gpu_memory, create_job_outputs_folder, create_job_history_folder, create_job_keep_completed_job, keep_temp_png],
        queue=False
    )

    quick_list.change(
        lambda x, current_n_prompt, current_lora_model, current_lora_weight, current_video_length, current_job_name, current_use_teacache, current_seed, current_steps, current_cfg, current_gs, current_rs, current_image_strength, current_mp4_crf, current_gpu_memory, current_outputs_folder, current_job_history_folder, current_keep_completed_job, current_keep_temp_png: (
            next((item['prompt'] for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), x),  # prompt
            next((item.get('n_prompt', current_n_prompt) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_n_prompt),  # n_prompt
            next((item.get('lora_model', current_lora_model) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_lora_model),  # lora_model
            next((item.get('lora_weight', current_lora_weight) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_lora_weight),  # lora_weight
            next((item.get('video_length', current_video_length) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_video_length),  # video_length
            next((item.get('job_name', current_job_name) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_job_name),  # job_name
            next((item.get('use_teacache', current_use_teacache) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_use_teacache),  # use_teacache
            next((item.get('seed', current_seed) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_seed),  # seed
            next((item.get('steps', current_steps) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_steps),  # steps
            next((item.get('cfg', current_cfg) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_cfg),  # cfg
            next((item.get('gs', current_gs) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_gs),  # gs
            next((item.get('rs', current_rs) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_rs),  # rs
            next((item.get('image_strength', current_image_strength) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_image_strength),  # image_strength
            next((item.get('mp4_crf', current_mp4_crf) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_mp4_crf),  # mp4_crf
            next((item.get('gpu_memory', current_gpu_memory) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_gpu_memory),  # gpu_memory
            next((item.get('outputs_folder', current_outputs_folder) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_outputs_folder),  # outputs_folder
            next((item.get('job_history_folder', current_job_history_folder) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_job_history_folder),  # job_history_folder
            next((item.get('keep_completed_job', current_keep_completed_job) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_keep_completed_job),  # keep_completed_job
            next((item.get('keep_temp_png', current_keep_temp_png) for item in quick_prompts if item['job_name'] == x.split(" - ")[0]), current_keep_temp_png),  # keep_temp_png
        ) if x else (
            Config.DEFAULT_PROMPT,  # prompt
            Config.DEFAULT_N_PROMPT,  # n_prompt
            Config.DEFAULT_LORA_MODEL,  # lora_model
            Config.DEFAULT_LORA_WEIGHT,  # lora_weight
            Config.DEFAULT_VIDEO_LENGTH,  # video_length
            Config.DEFAULT_JOB_NAME,  # job_name
            Config.DEFAULT_USE_TEACACHE,  # use_teacache
            Config.DEFAULT_SEED,  # seed
            Config.DEFAULT_STEPS,  # steps
            Config.DEFAULT_CFG,  # cfg
            Config.DEFAULT_GS,  # gs
            Config.DEFAULT_RS,  # rs
            Config.DEFAULT_IMAGE_STRENGTH,  # image_strength
            Config.DEFAULT_MP4_CRF,  # mp4_crf
            Config.DEFAULT_GPU_MEMORY,  # gpu_memory
            Config.OUTPUTS_FOLDER,  # outputs_folder
            Config.JOB_HISTORY_FOLDER,  # job_history_folder
            Config.KEEP_COMPLETED_JOB,  # keep_completed_job
            Config.DEFAULT_KEEP_TEMP_PNG  # keep_temp_png
        ),
        inputs=[quick_list, n_prompt, lora_model, lora_weight, video_length, job_name, use_teacache, seed,
         steps, cfg, gs, rs, image_strength, mp4_crf, gpu_memory, create_job_outputs_folder, create_job_history_folder, create_job_keep_completed_job, keep_temp_png],
        outputs=[prompt, n_prompt, lora_model, lora_weight, video_length, job_name, use_teacache, seed, steps, cfg, gs, rs,
         image_strength, mp4_crf, gpu_memory, create_job_outputs_folder, create_job_history_folder, create_job_keep_completed_job, keep_temp_png],
        queue=False
    )

    # Load queue on startup
    block.load(
        fn=lambda: (update_queue_table(), update_queue_display()),
        outputs=[queue_table, queue_display]
    )
    # Connect queue actions
    queue_table.select(
        fn=handle_queue_action,
        inputs=[],
        outputs=[
            edit_job_prompt,
            edit_job_n_prompt,  
            edit_job_lora_model,
            edit_job_lora_weight,
            edit_job_video_length,
            edit_job_seed,
            edit_job_use_teacache,
            edit_job_steps,
            edit_job_cfg,
            edit_job_gs,
            edit_job_rs,
            edit_job_image_strength,
            edit_job_mp4_crf,
            edit_job_gpu_memory,
            edit_job_keep_temp_png,
            edit_job_outputs_folder,
            edit_job_history_folder,
            edit_job_keep_completed_job,
            edit_job_name,
            change_job_name,
            edit_job_group,
            edit_job_table_group,
            queue_table,
            queue_display
        ]
    )

    # Add Load Jobs button connections
    metadata_job_button.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
        outputs=[load_metadata_job_group, load_metadata_job, edit_metadata_job_button]
    )

    # Only enable/disable edit button when file changes
    load_metadata_job.change(
        fn=enable_edit_metadata_button,
        inputs=[load_metadata_job],
        outputs=[edit_metadata_job_button]
    )

    # Show edit group and populate with metadata when edit button clicked
    edit_metadata_job_button.click(
        fn=prepare_metadata_job_edit,
        inputs=[load_metadata_job],
        outputs=[
            edit_metadata_job_group,
            edit_metadata_job_prompt,
            edit_metadata_job_n_prompt,
            edit_metadata_job_lora_model,
            edit_metadata_job_lora_weight,
            edit_metadata_job_video_length,
            edit_metadata_job_name,
            edit_metadata_job_outputs_folder,
            edit_metadata_job_history_folder,
            edit_metadata_job_use_teacache,
            edit_metadata_job_seed,
            edit_metadata_job_steps,
            edit_metadata_job_cfg,
            edit_metadata_job_gs,
            edit_metadata_job_rs,
            edit_metadata_job_image_strength,
            edit_metadata_job_mp4_crf,
            edit_metadata_job_gpu_memory,
            edit_metadata_job_keep_completed_job,
            edit_metadata_job_keep_temp_png
        ]
    )

    save_edit_metadata_job_button.click(
        fn=save_loaded_job,
        inputs=[
            edit_metadata_job_prompt,
            edit_metadata_job_n_prompt,
            edit_metadata_job_lora_model,
            edit_metadata_job_lora_weight,
            edit_metadata_job_video_length,
            edit_metadata_job_name,
            edit_metadata_job_outputs_folder,
            edit_metadata_job_history_folder,
            edit_metadata_job_use_teacache,
            edit_metadata_job_seed,
            edit_metadata_job_steps,
            edit_metadata_job_cfg,
            edit_metadata_job_gs,
            edit_metadata_job_rs,
            edit_metadata_job_image_strength,
            edit_metadata_job_mp4_crf,
            edit_metadata_job_gpu_memory,
            edit_metadata_job_keep_completed_job,
            edit_metadata_job_keep_temp_png
        ],
        outputs=[
            edit_metadata_job_group,
            load_metadata_job,
            load_metadata_job_group,
            queue_table,
            queue_display
        ]
    )

    cancel_edit_metadata_job_button.click(
        fn=cancel_loaded_job,
        outputs=[edit_metadata_job_group, load_metadata_job, load_metadata_job_group]
    )

    delete_completed_button.click(
        fn=delete_completed_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )

    delete_pending_button.click(
        fn=delete_pending_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )
    delete_failed_button.click(
        fn=delete_failed_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )
    delete_all_button.click(
        fn=delete_all_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )
    save_edit_job_button.click(
        fn=edit_job,
        inputs=[
            edit_job_name,
            change_job_name,
            edit_job_prompt,
            edit_job_n_prompt,
            edit_job_lora_model,
            edit_job_lora_weight,
            edit_job_video_length,
            edit_job_outputs_folder,
            edit_job_history_folder,
            edit_job_use_teacache,
            edit_job_seed,
            edit_job_steps,
            edit_job_cfg,
            edit_job_gs,
            edit_job_rs,
            edit_job_image_strength,  # Add Image Strength
            edit_job_mp4_crf,
            edit_job_gpu_memory,
            edit_job_keep_completed_job,
            edit_job_keep_temp_png
        ],
        outputs=[
            queue_table,
            queue_display,
            edit_job_group,
            edit_job_table_group
        ]
    )

    cancel_edit_job_button.click(
        fn=hide_edit_window,
        outputs=[edit_job_group, edit_job_table_group, queue_table]
    )

    job_data = [input_image, prompt, n_prompt, seed, video_length, latent_window_size, steps, cfg, gs, rs, image_strength, use_teacache, mp4_crf, gpu_memory, keep_temp_png]
        
    
    
    start_button.click(
        fn=process,
        inputs=[],
        outputs=[
            queue_button, start_button, abort_button, abort_delete_button,
            preview_image, result_video,
            progress_desc1, progress_bar1,
            progress_desc2,
            queue_display, queue_table
        ]
    )
    abort_button.click(
        fn=abort_process,
        outputs=[queue_table, queue_display, start_button, abort_button, abort_delete_button, queue_button]
    )
    abort_delete_button.click(
        fn=abort_and_delete_process,
        outputs=[queue_table, queue_display, start_button, abort_button, abort_delete_button, queue_button]
    )




    queue_button.click(
        fn=add_to_queue_handler,
        inputs=[
            input_image,
            prompt,
            n_prompt,
            lora_model,
            lora_weight,
            video_length,
            job_name,
            create_job_outputs_folder,
            create_job_history_folder,
            use_teacache,
            seed,
            steps,
            cfg,
            gs,
            rs,
            image_strength,
            mp4_crf,
            gpu_memory,
            create_job_keep_completed_job,
            keep_temp_png
        ],
        outputs=[
            queue_table,
            queue_display,
            queue_button
        ]
    )

    block.load(
        fn=lambda: (
            gr.update(interactive=True),      # queue_button
            gr.update(interactive=True),      # start_button
            gr.update(interactive=False),     # abort_button
            gr.update(interactive=False),     # abort_delete_button
            gr.update(visible=False),         # preview_image
            gr.update(visible=False),         # result_video
            "",                               # progress_desc1
            "",                               # progress_bar1
            "",                               # progress_desc2
            update_queue_display(),           # queue_display
            update_queue_table()              # queue_table
        ),
        outputs=[
            queue_button,
            start_button,
            abort_button,
            abort_delete_button,
            preview_image,
            result_video,
            progress_desc1,
            progress_bar1,
            progress_desc2,
            queue_display,
            queue_table
        ]
    )

# Add these calls at startup
reset_processing_jobs()
cleanup_orphaned_files()


# Launch the interface

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser
)





# End of file







        


    