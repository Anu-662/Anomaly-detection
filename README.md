# Anomaly Detection for Luxury Product Images

Unsupervised anomaly detection system for identifying non-standard images in a Chanel luxury bag dataset. Built and compared 4 deep learning models with a final ensemble approach.

## Overview

E-commerce platforms and luxury resellers need to automatically flag low-quality, promotional, or misleading product images. This project tackles that as an **unsupervised anomaly detection** problem — no labeled data required.

**Dataset:** 10,597 Chanel luxury bag images

## Models

| Model | Architecture | ROC-AUC | F1 Score |
|---|---|---|---|
| Isolation Forest | ResNet50 features + IsolationForest | 0.637 | — |
| Autoencoder | ResNet50 encoder + transposed conv decoder | 0.701 | — |
| VAE | ResNet50 encoder + probabilistic latent space (256-dim) | 0.678 | — |
| Attention Autoencoder | Autoencoder + self-attention between encoder/decoder | **0.733** | — |
| **Ensemble (4 models)** | Min-max normalized score averaging | **0.746** | **0.485** |

## Anomaly Types Detected

- Product images with text overlays or promotional graphics
- Marketplace listings featuring multiple products
- Low-quality or heavily edited images
- Images with unusual color distributions or blur

## Approach

1. **Feature extraction** — Pretrained ResNet50 (ImageNet) used as backbone across all models
2. **Model training** — All models trained unsupervised on the full dataset
3. **Evaluation** — Synthetic anomalies (blur, color shift, noise, cutout) used to generate ground truth for quantitative evaluation
4. **Explainability** — Grad-CAM visualization highlights which image regions triggered anomaly detection
5. **Ensemble** — Scores from all 4 models normalized and averaged for best performance

## Tech Stack

- **PyTorch** — Model training and inference
- **torchvision / timm** — ResNet50 pretrained backbone
- **scikit-learn** — Isolation Forest, metrics, t-SNE
- **Grad-CAM** — Visual explainability
- **Streamlit** — Web application for deployment
- **Google Colab** — Training environment (GPU)

## Repository Structure

```
├── FINAL_code.ipynb     # Full pipeline: data loading → training → evaluation
└── README.md
```

## Results

The **Attention Autoencoder** performed best among individual models (AUC: 0.733), benefiting from self-attention capturing long-range spatial dependencies in bag images. The **4-model ensemble** (AUC: 0.746) outperformed all individual models by combining complementary detection strategies — reconstruction-based models catch pixel-level anomalies while Isolation Forest captures semantic outliers in feature space. Spearman correlation between model scores was near-zero, confirming they detect different types of anomalies.

## Saved Models

- `autoencoder_chanel.pth` — Autoencoder weights
- `vae_chanel.pth` — VAE weights
- `attention_ae_chanel.pth` — Attention Autoencoder weights
- `isolation_forest_model.pkl` — Trained Isolation Forest
- `scaler.pkl` — Feature scaler for Isolation Forest
