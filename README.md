# Hybrid Quantum-Driven Learning Framework for Tuberculosis Detection Using Chest X-Ray Images

Final year BTech project: TB detection from chest X-Ray images with **target accuracy ≥98%**, improving on the base paper (TBNet, ~97% accuracy).

## Improvements over base paper

| Aspect | Base paper (TBNet) | This framework |
|--------|---------------------|----------------|
| **Accuracy** | ~97.04% | Target **≥98%** (strong augmentation, quantum-inspired head, EfficientNet) |
| **Preprocessing** | CLAHE + upper 1/3 lung segmentation (manual masks) | CLAHE + full-image (no manual masks); optional segmentation can be added |
| **Classifier** | Vision Transformer (ViT-L/16, 1.12 GB) | EfficientNet-B3 + **quantum-inspired layer** (lighter, quantum-inspired design) |
| **Quantum aspect** | — | **Quantum-inspired layer**: amplitude encoding, parameterized rotation, measurement |
| **Dataset** | Multi-source (Shenzhen, Montgomery, RSNA, Belarus) | Kaggle TB Chest X-Ray (Tawsifur Rahman), 7K images |

## Dataset

Use the **Tuberculosis (TB) Chest X-Ray Dataset** from Kaggle:

- **URL:** https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset  
- **Content:** 3,500 Normal + 3,500 Tuberculosis chest X-Ray images (7,000 total)

### Setup

1. Install [Kaggle API](https://github.com/Kaggle/kaggle-api) and place `kaggle.json` in `~/.kaggle/` (or `C:\Users\<you>\.kaggle\` on Windows).

2. Download and extract in the project folder:
   ```bash
   cd c:\Users\sowba\OneDrive\Pictures\Documents\TB
   kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
   # Extract the zip; you should get a folder like "TB_Chest_Radiography_Database" with subfolders "Normal" and "Tuberculosis"
   ```

3. Ensure this structure (or equivalent):
   ```
   TB/
   ├── TB_Chest_Radiography_Database/
   │   ├── Normal/          ← normal chest X-Ray images
   │   └── Tuberculosis/    ← TB chest X-Ray images
   ├── config.py
   ├── dataset.py
   ├── model.py
   ├── train.py
   ├── inference.py
   └── app.py
   ```

   If your folder name is different, set `DATA_ROOT` in `config.py` to that path.

## Environment

```bash
cd c:\Users\sowba\OneDrive\Pictures\Documents\TB
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train (target ≥98% accuracy)

```bash
python train.py --data_root "c:\Users\sowba\OneDrive\Pictures\Documents\TB\TB_Chest_Radiography_Database"
```

Optional arguments: `--batch_size 32`, `--epochs 80`, `--lr 1e-4`, `--save_dir saved_models`, `--seed 42`.

- Best model is saved as `saved_models/best_tb_model.pt`.
- Metrics and training history are written under `results/`.

## Run inference on an image

```bash
python inference.py "path\to\chest_xray.png"
```

Output: **TB Positive** or **TB Negative** with confidence and class probabilities.

## Web app (upload → TB Positive / Negative)

```bash
python app.py
```

Open the URL shown in the terminal (e.g. http://127.0.0.1:7860), upload a chest X-Ray image, and get **TB Positive** or **TB Negative** with probabilities.

## Project structure

- `config.py` – Paths, hyperparameters, CLAHE and training settings  
- `dataset.py` – Data loading, CLAHE, augmentations, train/val/test split  
- `model.py` – Hybrid model: EfficientNet backbone + quantum-inspired layer  
- `train.py` – Training loop, metrics, early stopping, saving best model  
- `inference.py` – Load model, preprocess (CLAHE), predict  
- `app.py` – Gradio UI for image upload and prediction  

## Citation (base paper)

If you use ideas from the base paper (TBNet):

- M. Irhamsyah et al., "TBNet: A Chest X-Ray Classifier Supporting Image Segmentation With Self-Attention Mechanism for Secondary Tuberculosis Detection," IEEE Access, 2025.
