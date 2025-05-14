# Temporal to Spatial Knowledge Distillation for Real-time Fire Segmentation: An Approach involving Kolmogorov-Arnold Networks

This repository contains the source code for fire segmentation using the customized models on the **Chino_v2** dataset:
- U-Net
- U-Net-KAN
- U-Net-KAN-LSTM
- U-Net-KAN-LSTM-MobileNetV2

It also includes a **teacher-student distillation** process to train a lightweight student model.

---

## Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Test Dataset: `test/Chino` (place in `./data/firevideo`). You can find the dataset attached with the [paper](https://arxiv.org/pdf/1506.03495).

---

## Installation

**1. Clone the repository:**
```
git clone https://github.com/lebadac/1301_Fire-Semantic-Segmentation.git
   ```
**2. Install dependencies:**
```
pip install -r requirements.txt
```

## 3. Prepare Dataset

### Test Dataset

Place the **Chino et al. dataset** in the following directory:
```
./data/firevideo
```

## Usage
Run the main script to load pre-trained weights and evaluate models on the test dataset:
```
python main.py
```
**Directory Structure**
```
1301_Fire-Semantic-Segmentation/
├── src/                                # Source code
│   ├── data_loader.py                  # Test dataset loading and preprocessing
│   ├── unet_model.py                   # U-Net model definition
│   ├── unet_kan_model.py               # U-Net-KAN model definition
│   ├── unet_kan_lstm.py                # U-Net-KAN-LSTM model definition
│   ├── unet_kan_lstm_mobilenetv2.py   # U-Net-KAN-LSTM-MobileNetV2 model definition
│   ├── evaluate.py                     # Model evaluation
│   ├── distillation.py                 # Student model definition
│   ├── utils.py                        # Shared utility functions
│   └── main.py                         # Main script for evaluation
│
├── saved_weights/                      # Pre-trained model weights
│   ├── unetweights.h5
│   ├── u_kan.weights.h5
│   ├── u_kan_lstm.weights.h5
│   ├── u_kan_lstm_mobilenetv2.weights.h5
│   └── distilled_student_model_weights.weights.h5
│
├── requirements.txt                    # Dependencies
└── README.md                           # Project description

```

**Models**

- U-Net: Standard U-Net model for image segmentation.
- U_KAN: U-Net with a tokenized KAN block in the bottleneck.
- U_KAN_LSTM: U-Net with a tokenized KAN block incorporating ConvLSTM2D for temporal processing.
- U_KAN_LSTM_MobileNetV2: U-Net with MobileNetV2 encoder and KAN-LSTM bottleneck (used as teacher model).
- Student Model: Lightweight model trained via knowledge distillation from the teacher model.

**Notes**
- Update the dataset path in main.py to match your local setup.

- Model weights are saved in saved_weights/ during training.

- Visualization of predictions is saved as PNG files during evaluation.



