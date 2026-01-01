# -----------------
# Imports
# -----------------

import os
import glob
import torch
from PIL import Image

# -----------------
# Model and Checkpoint Loading
# -----------------

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def get_available_checkpoints():
    """Get list of available checkpoints from the checkpoints directory"""
    checkpoints = []
    checkpoint_files = glob.glob(os.path.join("models", "test_2", "checkpoints", "*.pth"))
    for checkpoint in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint)
        checkpoints.append(checkpoint_name)
    return checkpoints

def generate_answer(prompt, image, parameters, checkpoint_name=None):
    """Generate answer from the model"""
    # This is a placeholder implementation - you would replace with actual model loading and generation
    # In a real implementation, you would load the model and checkpoint here
    
    if checkpoint_name:
        checkpoint_path = os.path.join("models", "test_2", "checkpoints", checkpoint_name)
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        available_checkpoints = get_available_checkpoints()
        if available_checkpoints:
            checkpoint_name = available_checkpoints[0]
            checkpoint_path = os.path.join("models", "test_2", "checkpoints", checkpoint_name)
            print(f"Using default checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints available")
    
    # Placeholder response
    return f"Test_2 model response for prompt: {prompt}\nUsing checkpoint: {checkpoint_name}\nParameters: {parameters}" 