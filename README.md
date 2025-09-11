# 🛠️ Surface Defect Detection using Deep Learning  

## 📌 Project Overview  
This project focuses on **automated surface defect detection** in steel quality assurance. Surface defects are common in industrial production and can significantly impact the safety and durability of the final product. The goal is to build a **deep learning model** that classifies six common types of steel surface defects, providing a scalable alternative to manual inspection.  

**Defect classes:**  
- Crazing (Cr)  
- Inclusion (In)  
- Patches (Pa)  
- Pitted Surface (PS)  
- Rolled-in Scale (RS)  
- Scratches (Sc)  

## 📊 Dataset  
- **Source:** [NEU Surface Defect Database (Kaggle)](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)  
- **Size:** 1,800 grayscale images (300 per defect class)  
- **Split:** 240 training, 60 testing samples per class  
- **Preprocessing:**  
  - Resized to 224×224 RGB  
  - Normalized pixel values (0–1)  
  - Data augmentation (random flips, rotations, zoom, contrast, translations)  

## ⚙️ Methodology  
1. **Data Pipeline**  
   - Preprocessed images into TensorFlow datasets  
   - Created shuffled and batched datasets for training, validation, and testing  

2. **Model Architecture**  
   - **Backbone:** MobileNetV2 (transfer learning)  
   - **Regularization:** Dropout (0.5), L2 penalty  
   - **Loss function:** Categorical Crossentropy with label smoothing  
   - **Optimizer:** Adam (learning rate = 1e-5)  

3. **Training Strategy**  
   - Early stopping & model checkpointing  
   - TensorBoard monitoring  
   - 5-fold cross-validation  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score (macro)  
   - Confusion matrix  
   - Class-wise precision/recall  

## 📈 Results  
- **Validation Accuracy:** 95.1%  
- **Test Performance:**  
  - Accuracy: **96.4%**  
  - Precision: **96.5%**  
  - Recall: **96.4%**  

- **Per-Class Performance:**  
  - Crazing – Precision: 0.95, Recall: 0.95  
  - Inclusion – Precision: 0.98, Recall: 0.95  
  - Patches – Precision: 0.95, Recall: 1.00  
  - Pitted Surface – Precision: 1.00, Recall: 0.88  
  - Rolled-in Scale – Precision: 0.95, Recall: 1.00  
  - Scratches – Precision: 0.95, Recall: 1.00  

Grad-CAM visualizations confirmed that the model learned **discriminative regions** for each defect class.  

## 🚀 Next Steps  
- **Deployment:** FastAPI backend, containerization with Docker, and deployment on Render/AWS.  
- **Extensions:**  
  - Scale to larger industrial datasets  
  - Integrate into real-time quality assurance pipelines  

## 🚀 Quick Start  
1. Clone the repo and install dependencies:  
   ```bash
   pip install -r requirements.txt

2. Download the dataset from Kaggle.

3. Open surface-defect-detection.ipynb in Jupyter/Colab and run all cells to train and evaluate the model.

## 🛠️ Tech Stack  
- Python  
- TensorFlow/Keras  
- NumPy, Pandas, Matplotlib  
- scikit-learn  
- Docker (deployment)  

## 📂 Repository Structure  
├── notebooks/           # surface-defect-detection.ipynb
├── requirements.txt     
├── README.md     
