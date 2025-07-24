import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os

def load_resnet_model():
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove FC layer
    model.eval()
    return model

# Preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Extract a feature vector for a single crop
def extract_feature(image_path, model, device='cpu'):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image).squeeze().cpu().numpy()  # Shape: (2048,)
    return feature / (feature.sum() + 1e-10)  # Normalize

# Batch extract features for all frames' detections
def extract_all_features(frame_data, model, device='cpu'):
    for frame in frame_data:
        for det in frame['detections']:
            crop_path = det['crop_path']
            embedding = extract_feature(crop_path, model, device)
            det['embedding'] = embedding
    return frame_data
