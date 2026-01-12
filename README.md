🧠 Breast Cancer MRI Classification

EfficientNet • Stratified K-Fold • ROC Analysis • Clinical Metrics

This project implements a deep learning pipeline for breast cancer classification from MRI images using EfficientNet-B0 with Stratified K-Fold Cross-Validation to ensure robust and clinically reliable evaluation.

The model focuses on binary classification:

Healthy

Sick (Cancerous)

🚀 Key Features

EfficientNet-B0 with ImageNet pretraining

Stratified K-Fold Cross-Validation (5 folds)

Class imbalance handling using weighted loss

Data augmentation for MRI robustness

ROC curve & AUC computed per fold

Clinical metrics:

Sensitivity (Recall – Sick)

Specificity (Recall – Healthy)

Final model export for deployment

🧪 Classification Task
Class Index	Label
0	Healthy
1	Sick (Breast Cancer)
📁 Dataset Structure (REQUIRED)

Your MRI dataset must follow this exact directory structure:

train/
│
├── Healthy/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
│
├── Sick/
│   ├── img_101.jpg
│   ├── img_102.jpg
│   └── ...

📌 Important Notes

Images must be inside class folders (Healthy, Sick)

Supported formats: .jpg, .png

MRI images can be grayscale or RGB

Folder names define class labels automatically

⚙️ Configuration Summary
Parameter	Value
Image Size	224 × 224
Batch Size	30
Epochs	20
Folds	5
Optimizer	AdamW
Learning Rate	3e-4
Loss Function	Weighted CrossEntropy
🧠 Data Augmentation

Applied only during training:

Random horizontal flip

Random rotation (±15°)

Brightness & contrast jitter

Validation data is not augmented.

▶️ How to Run
1️⃣ Install dependencies
pip install torch torchvision scikit-learn matplotlib numpy

2️⃣ Prepare dataset

Place your dataset inside the train/ directory following the structure above.

3️⃣ Run training
python main.py

📊 Evaluation Outputs
🔹 Per-Fold Metrics

ROC curve plotted for each fold

AUC score computed per fold

Mean AUC reported at the end

🔹 Final Metrics

Confusion Matrix

Classification Report

Sensitivity (Cancer Recall)

Specificity (Healthy Recall)

📈 ROC Curve Interpretation

High AUC (≥ 0.90) → Excellent diagnostic capability

Sensitivity → Ability to detect cancer cases

Specificity → Ability to correctly identify healthy cases

These metrics are clinically critical in medical imaging systems.

💾 Saved Model

After training completes, the final model is saved as:

breast_cancer_efficientnet_final.pth


This model can be reused for:

Inference

Fine-tuning

Deployment in clinical research pipelines

⚠️ Medical Disclaimer

This project is intended for research and educational purposes only.
It is NOT a medical device and must not be used for real-world diagnosis.
All clinical decisions must be made by certified healthcare professionals.


<img width="1500" height="1200" alt="clinical_metrics" src="https://github.com/user-attachments/assets/90148d54-bcb3-458a-96c4-59a57041d3c0" />
<img width="600" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/f63dccc2-2f20-4d6d-8612-fe2f44b6aa12" />
<img width="1500" height="1200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c0d7074c-30f2-4fd0-bb6b-1acba999f54f" />
