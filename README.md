
# Identifying Osteoarthritis Severity in Anterior-Posterior Knee Radiographs Using Transfer Learning with a Pre-trained Self-Supervised UNet Model

## Introduction  
Osteoarthritis (OA) is one of the most prevalent forms of arthritis, affecting millions globally. This project focuses on classifying the severity of OA using anterior-posterior (AP) knee radiographs. We utilize transfer learning with a pre-trained self-supervised UNet model to enhance classification accuracy. 

## Objectives  
- **Multi-class classification** of knee radiographs based on Kellgren and Lawrence’s classification system (severity levels 0–4).  
- **Transfer learning approach** using a self-supervised pre-trained UNet model for feature extraction and fine-tuning on labeled data.  

---

## Datasets  
### Pool 1  
- **Size:** 2,352 labeled radiographs.  
- **Labels:** Severity levels (0–4) assigned by radiologists based on the Kellgren and Lawrence classification.  

### Pool 2  
- **Size:** 7,434 unlabeled radiographs.  
- **Purpose:** Used for self-supervised training (inpainting task).  

Both datasets were obtained from the Department of Orthopedic Surgery, DCW Medical Center (MN, USA).

---

## Methods  

### Preprocessing  
- Images resized to **112×112 pixels** to avoid memory issues.  
- Black-masked patches generated for inpainting training on Pool 2.

### Model Architecture  
1. **Self-Supervised Learning**:  
   - UNet-like model trained for image inpainting on Pool 2.
   - Encoder weights extracted and used for transfer learning.  
2. **Transfer Learning**:  
   - Encoder fine-tuned on Pool 1 for classification.  
   - **Metrics:** Accuracy, precision, recall, and F1-score.  

### Evaluation  
- **5-Fold Cross-Validation** for robust evaluation.  
- **Principal Component Analysis (PCA)** for dimensionality reduction.  

- **Comparison:** Fine-tuned model outperformed the naïve classifier in all metrics.  

---

### Prerequisites  
1. **Programming Languages:** Python 3.x  
2. **Libraries:**  
   - PyTorch  
   - NumPy  
   - scikit-learn  
   - Matplotlib  
   - OpenCV  
   - Jupyter Notebook  


### Running the Code  
1. **Inpainting Task**  
   Open `Inpainting.ipynb` and run all cells to train the UNet model on Pool 2 for inpainting.  

2. **Classification Task**  
   Open `classification.ipynb` and run all cells to fine-tune the classifier and evaluate performance on Pool 1.  

### Outputs  
- Trained UNet model for inpainting.  
- Fine-tuned classifier for OA severity classification.  
- Metrics (accuracy, precision, recall, F1-score) visualized as box plots.  

---

## Project Structure  
```
OA-severity-classification/
│
├── Inpainting.ipynb      # Self-supervised UNet inpainting model
├── classification.ipynb  # Transfer learning for OA classification
├── data/                 # Folder for datasets
├── results/              # Folder for saving outputs
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

---

## Acknowledgments  
This work utilizes datasets provided by the Department of Orthopedic Surgery, DCW Medical Center (MN, USA).  
