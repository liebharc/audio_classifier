from fastai.vision.all import *
import tensorflow as tf
import torch

def train_classifier(data_path):
    """Loads preprocessed images and trains a classifier."""
    dls = ImageDataLoaders.from_folder(data_path, valid_pct=0.2, seed=42)
    learn = vision_learner(dls, squeezenet1_1, metrics=accuracy)
    learn.fine_tune(5)
    return learn

def convert_to_torchscript(learner, output_file="audio_classifier.pt", onnx_filename='audio_classifier.onnx'):
    """Converts the trained FastAI model to a TorchScript format for web deployment."""
    model = learner.model.cpu().eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_file)
    print(f"TorchScript model saved as {output_file}")
    dummy_input = torch.randn(1, 3, 40, 28)  # Image size in RGB
    
    # Export to ONNX
    torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'])
    print(f"Model exported to {onnx_filename}")

dataset_path = "processed"

# Training
learner = train_classifier(dataset_path)
learner.export("audio_classifier.pkl")
convert_to_torchscript(learner, "audio_classifier.pt")
