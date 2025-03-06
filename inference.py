import torch
from PIL import Image
import numpy as np
import argparse

categories = ["music", "noise", "speech"]

def load_model(model_path="audio_classifier.pt"):
    """Loads the TorchScript model."""
    model = torch.jit.load(model_path)
    model.eval()
    return model

def preprocess_image(image, start_x, width, height):
    """Extracts a window from the image and converts it into a tensor."""
    image_window = image.crop((start_x, 0, start_x + width, height))
    image_array = np.array(image_window).astype(np.float32) / 255.0  # Normalize
    image_array = image_array.transpose(2, 0, 1)  # Convert to [C, H, W]
    return torch.tensor(image_array).unsqueeze(0)  # Add batch dimension

def predict(model, file_path):
    image = Image.open(file_path).convert('RGB')
    width, height = image.size
    step_size = 14
    window_size = 28
    
    for start_x in range(0, width - window_size + 1, step_size):
        input_tensor = preprocess_image(image, start_x, window_size, height)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        print(f"{start_x}: {categories[prediction]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image file with sliding windows.")
    parser.add_argument("file_path", type=str, help="Path to the image file.")
    parser.add_argument("--model", type=str, default="audio_classifier.pt", help="Path to the model file.")
    
    args = parser.parse_args()
    model = load_model(args.model)
    predict(model, args.file_path)
