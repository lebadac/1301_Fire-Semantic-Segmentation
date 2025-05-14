# Temporal to Spatial Knowledge Distillation for Real-time Fire Segmentation: An Approach involving Kolmogorov-Arnold Networks

This repository contains the source code for fire segmentation using the customized models on the **firevideo** dataset:
- U-Net
- U-Net-KAN
- U-Net-KAN-LSTM
- U-Net-KAN-LSTM-MobileNetV2

It also includes a **teacher-student distillation** process to train a lightweight student model.

---

## 1. Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Test Dataset: `test/Chino` (place in `./data/firevideo`). You can find the dataset attached with the [paper](https://arxiv.org/pdf/1506.03495).

---

## 2. Installation

**Step 1: Clone the repository:**
```
git clone https://github.com/lebadac/1301_Fire-Semantic-Segmentation.git
   ```
**Step 2: Install dependencies:**
```
pip install -r requirements.txt
```

## 3. Prepare Dataset

### 3.1. Train Dataset

#### Fire

This dataset includes approximately 1,479 fire images extracted from **57 video clips**. Among them, **40 videos** were manually labeled using the [Labelme](https://github.com/wkentaro/labelme) tool, and these labeled folders are named according to their original video filenames.

| **Source**            | **# Videos** | **Indoor / Outdoor** | **Short / Long Dist.** | **Low / High Act.** | 🔗 **Link** |
|-----------------------|--------------|-----------------------|-------------------------|----------------------|-------------|
| Firesense             | 11           | 3 / 8                 | 10 / 1                  | 11 / 0               | [Link](https://zenodo.org/records/836749) |
| KMU                   | 20           | 4 / 16                | 11 / 9                  | 19 / 1               | [Link](https://cvpr.kmu.ac.kr/) |
| FireNet               | 5            | 0 / 5                 | 5 / 0                   | 3 / 2                | [Link](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection?tab=readme-ov-file) |
| NIST                  | 4            | 3 / 1                 | 2 / 2                   | 4 / 0                | [Link](https://www.nist.gov/programs-projects/national-fire-research-laboratory-advanced-metrology/360-degree-video-fire) |
| Muhammad et al.       | 17           | 0 / 17                | 11 / 6                  | 3 / 14               | [Link](https://github.com/hayatkhan8660-maker/Fire_Seg_Dataset) |
| **Total**             | **57**       | **10 / 47**           | **39 / 18**             | **40 / 17**          |             |

#### Non-fire

In addition to the fire images, a total of **605 non-fire images** were collected from various publicly available datasets. The details are provided in our paper.

This subset represents approximately **30%** of the full dataset. You can explore it [here](https://drive.google.com/drive/folders/11RjIO5WFxCDWxjwz_dXqSOUJjYov2yHi?usp=drive_link).

### 3.2. Test Dataset

Place the **Chino et al. dataset** in the following directory:
```
./data/firevideo
```

## 4. Usage
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
│   ├── unet_kan_lstm_mobilenetv2.py    # U-Net-KAN-LSTM-MobileNetV2 model definition
│   ├── distillation.py                 # Student model definition
│   ├── utils.py                        # Shared utility functions
│   ├── training_process.py             # Training process definition
│   ├── evaluate.py                     # Model evaluation
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

## 5. Result
### 5.1. Individual Modules
| **Model**          | **Pixel Acc.** | **Mean Acc.** | **Mean IoU** | **FWIoU** | **FPS**  |
|--------------------|----------------|---------------|--------------|-----------|----------|
| U-Net              | 92.65          | 77.55         | 64.34        | 88.40     | 72.05    |
| U-KAN              | 93.33          | 89.15         | 69.37        | 89.78     | 97.63    |
| U-KAN-LSTM         | 95.32          | 86.42         | 73.61        | 92.20     | 99.46    |
| Teacher Model      | 97.88          | 89.80         | 80.72        | 94.68     | 66.05    |
| **Student Model**  | **97.25**      | **89.41**     | **81.63**    | **95.03** | **147.02**|

*Comparison of semantic segmentation performance across models.*
### 5.2. Comparison with State-of-the-Art Segmentation Methods

| **Method**              | **Pixel Acc.** | **Mean Acc.** | **Mean IoU**  | **FWIoU**   |
|-------------------------|----------------|---------------|---------------|-------------|
| SegNet                  | 84.63          | 75.92         | 80.41         | 87.66       |
| FCN                     | 85.76          | 75.47         | 72.65         | 89.20       |
| PSPNet                  | 88.17          | 78.62         | 74.19         | 89.58       |
| Muhammad et al.         | 94.54          | 85.27         | **83.35**     | 93.96       |
| **Student Model**       | **97.25**      | **89.41**     | 81.63         | **95.03**   |

*Comparison with state-of-the-art segmentation methods on the Chino et al. test set.*


## 6. Notes
- Update the dataset path in main.py to match your local setup.

- Model weights are saved in saved_weights/ during training.

- Visualization of predictions is saved as PNG files during evaluation.

- The FPS can be different because of running on different hardware. However, the relative speed (fast or slow) between models is still accurate.

- If having problems with the file `weights.h5`, please download it [here](https://drive.google.com/drive/folders/1uFZ_qdeCEUr0p-H1r1GJcNQhVtoq-Knm?usp=sharing).



