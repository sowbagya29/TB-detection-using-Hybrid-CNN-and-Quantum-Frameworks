"""
Inference: load X-Ray image -> CLAHE -> model -> TB Positive / Negative.
"""
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from config import IMG_SIZE, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE, MODEL_SAVE_DIR, CLASSES
from dataset import apply_clahe
from model import build_model

try:
    from torchvision import transforms
except ImportError:
    transforms = None


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])


def load_model(ckpt_path: Path = None, device: str = None):
    if ckpt_path is None:
        ckpt_path = Path(MODEL_SAVE_DIR) / "best_tb_model.pt"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = build_model(pretrained=False)
    # weights_only=False because we saved a full dict with metadata
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: np.ndarray, use_clahe: bool = True):
    """Convert to PIL after CLAHE if needed; then apply transform."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if use_clahe:
        image = apply_clahe(image, clip_limit=CLAHE_CLIP_LIMIT, grid_size=CLAHE_GRID_SIZE)
        image = np.stack([image] * 3, axis=-1)
    pil = Image.fromarray(image)
    transform = get_inference_transform()
    tensor = transform(pil).unsqueeze(0)
    return tensor


def predict_from_array(model, image: np.ndarray, device, use_clahe: bool = True):
    """Predict from numpy image (H, W) or (H, W, C). Returns class index and probability."""
    x = preprocess_image(image, use_clahe=use_clahe)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        conf = probs[0, pred_idx].item()
    return pred_idx, conf, probs[0].cpu().numpy()


def predict_from_path(model, image_path: str, device, use_clahe: bool = True):
    """Load image from path and predict."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.array(Image.open(path).convert("RGB"))
    return predict_from_array(model, img, device, use_clahe=use_clahe)


def run_inference(image_path: str, model_path: Path = None):
    """
    Single entry point: path -> (label_str, confidence, probs).
    label_str is "Tuberculosis" (Positive) or "Normal" (Negative).
    """
    model, device = load_model(ckpt_path=model_path)
    pred_idx, conf, probs = predict_from_path(model, image_path, device, use_clahe=True)
    label_str = CLASSES[pred_idx]
    return label_str, conf, probs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_chest_xray_image>")
        sys.exit(1)
    path = sys.argv[1]
    label, conf, probs = run_inference(path)
    print(f"Prediction: {label if label != 'Tuberculosis' else 'TB'}")
    print(f"Confidence: {conf:.4f}")
    # Simple suspicion level based on confidence
    if conf >= 0.90:
        suspicion = "HIGH"
    elif conf >= 0.70:
        suspicion = "MODERATE"
    else:
        suspicion = "LOW"
    print(f"Suspicion Level: {suspicion}")
    print("\nClinical recommendations:")
    if label == "Tuberculosis":
        print("- Seek clinical consultation with a pulmonologist")
        print("- Order sputum smear, Xpert MTB/RIF, or culture as indicated")
        print("- Isolate patient until infectious TB is excluded")
        print("- Start appropriate anti-TB therapy per national / WHO guidelines")
        print("- Ensure nutritional support and contact tracing")
    else:
        print("- Current X-ray does not show TB pattern according to this model")
        print("- If symptoms (cough, fever, weight loss) persist, consult a clinician")
        print("- Maintain healthy lifestyle and routine clinical check-ups as needed")
    print(f"\nProbabilities: Normal={probs[0]:.4f}, Tuberculosis={probs[1]:.4f}")
