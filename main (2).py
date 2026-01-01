# -----------------
# Imports
# -----------------

from ast import main
from urllib import response
from numpy import clip
import models.test_0.load as loads
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import glob

# -----------------
# Model and Tokenizer Loading
# -----------------

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def get_available_checkpoints():
    """Get list of available checkpoints from the checkpoints directory"""
    checkpoints = []
    checkpoint_files = glob.glob(os.path.join("models", "test_0", "checkpoints", "*.pth"))
    for checkpoint in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint)
        checkpoints.append(checkpoint_name)
    return checkpoints

def load_model(checkpoint_name=None):
    """Load model with specified checkpoint"""
    global adaptor, smol_model, tokenizer, clip_model, processor
    
    # Configure quantization based on device
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        smol_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # CPU configuration - avoid quantization completely
        model_checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        smol_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )

    adaptor = loads.SimpleMLP(64, text_dim=576)

    # Load model weights with map_location to handle device correctly
    if checkpoint_name is None:
        # Default to the first available checkpoint
        available_checkpoints = get_available_checkpoints()
        if available_checkpoints:
            checkpoint_name = available_checkpoints[0]
        else:
            checkpoint_name = "adaptor_mlp_tested.pth"  # Fallback

    model_path = os.path.join("models", "test_0", "checkpoints", checkpoint_name)
    adaptor.load_state_dict(torch.load(model_path, map_location=device))
    adaptor.to(device)
    adaptor.eval()

    # Load CLIP model with correct device
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    smol_model.eval()
    
    return {"adaptor": adaptor, "model": smol_model, "tokenizer": tokenizer, 
            "clip_model": clip_model, "processor": processor}

# Initialize with default checkpoint
current_model = load_model()
adaptor = current_model["adaptor"]
smol_model = current_model["model"]
tokenizer = current_model["tokenizer"]
clip_model = current_model["clip_model"]
processor = current_model["processor"]

def generate_answer(prompt, image, parameters, checkpoint_name=None):
    global adaptor, smol_model, tokenizer, clip_model, processor
    
    # If checkpoint is specified and different from current, reload the model
    if checkpoint_name is not None:
        current_model = load_model(checkpoint_name)
        adaptor = current_model["adaptor"]
        smol_model = current_model["model"]
        tokenizer = current_model["tokenizer"]
        clip_model = current_model["clip_model"]
        processor = current_model["processor"]
    
    image.thumbnail((512,512), Image.LANCZOS)
    image_patches = loads.split_image_into_patches_pil(image)
    response_tokens = adaptor.generate(
            main_model=smol_model,
            main_tokenizer=tokenizer,
            Qmodel=smol_model.model.embed_tokens,
            image_model=clip_model,
            image_processor=processor,
            input_text=prompt,
            image=image_patches,
            wrapping_function=None,
            max_new_tokens=parameters['max_new_tokens'],
            Repetation_penalty=parameters['Repetation_penalty'],
            sentence_repeat_penalty=parameters['sentence_repeat_penalty'],
            temperature=parameters['temperature'],
            top_k=parameters['top_k'],
            top_p=parameters['top_p']
    )
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    return response
