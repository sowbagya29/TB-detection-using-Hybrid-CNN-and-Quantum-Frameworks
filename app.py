"""
Gradio app: Upload chest X-Ray image -> TB Positive / Negative.
Hybrid Quantum-Driven Learning Framework for Tuberculosis Detection.
"""
import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image

from inference import load_model, predict_from_array, preprocess_image
from gradcam import generate_gradcam_overlay

MODEL_SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
CKPT_PATH = MODEL_SAVE_DIR / "best_tb_model.pt"


def load_model_once():
    if not CKPT_PATH.exists():
        return None, "Model not found. Train first: python train.py --data_root <path_to_dataset>"
    model, device = load_model(ckpt_path=CKPT_PATH, device=None)
    return model, device


# Lazy load model on first prediction
_model_cache = None


def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model_once()
    return _model_cache


def predict(image):
    if image is None:
        return "Please upload a chest X-Ray image.", "", None, None
    model, device = get_model()
    if model is None:
        return device, "", None, None  # device holds error message
    try:
        img = np.array(image) if not isinstance(image, np.ndarray) else image
        # Prediction
        pred_idx, conf, probs = predict_from_array(model, img, device, use_clahe=True)
        label = "Tuberculosis" if pred_idx == 1 else "Normal"
        result = "TB Positive" if pred_idx == 1 else "TB Negative"
        detail = f"Prediction: **{result}** ({label}) with confidence **{conf:.1%}**\n\n"
        # Recommendations (impressive output)
        if pred_idx == 1:
            detail += (
                "**Recommended next steps (screening output):**\n"
                "- Consult a pulmonologist / physician\n"
                "- Confirm with lab tests: sputum smear, Xpert MTB/RIF, culture as advised\n"
                "- Follow isolation and treatment guidelines per clinician\n"
                "- Ensure nutrition and contact tracing support\n"
            )
        else:
            detail += (
                "**Recommended next steps:**\n"
                "- No TB pattern detected by the model\n"
                "- If symptoms persist, consult a clinician\n"
                "- Maintain routine health monitoring\n"
            )
        probs_str = f"Normal: {probs[0]:.1%}  |  Tuberculosis: {probs[1]:.1%}"
        # Grad-CAM (highlight suspected region)
        input_tensor = preprocess_image(img, use_clahe=True)
        _, overlay = generate_gradcam_overlay(model, input_tensor, original_rgb=img[..., :3], class_index=pred_idx)
        return result, detail + "\n\n" + probs_str, {"Normal": float(probs[0]), "Tuberculosis": float(probs[1])}, overlay
    except Exception as e:
        return "Error", str(e), None, None


def main():
    title = "Hybrid Quantum-Driven Learning Framework for Tuberculosis Detection"
    description = (
        "Upload a chest X-Ray image. The model will predict **TB Positive** or **TB Negative**. "
        "Uses CLAHE preprocessing and a hybrid quantum-inspired classifier trained on the TB Chest X-Ray dataset."
    )
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Chest X-Ray Image"),
        outputs=[
            gr.Textbox(label="Result", value=""),
            gr.Markdown(label="Details"),
            gr.Label(label="Probabilities"),
            gr.Image(type="numpy", label="Grad-CAM (highlighted TB area)"),
        ],
        title=title,
        description=description,
        examples=None,
        allow_flagging="never",
    )
    demo.launch(share=False)


if __name__ == "__main__":
    main()
