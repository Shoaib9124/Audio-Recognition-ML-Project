Audio Recognition for Gender and Age Group Classification

This repository contains the code and results for an audio recognition project focused on classifying gender and age groups from audio samples using Machine Learning and Deep Learning techniques.

🔎 Project Overview

The project leverages audio feature extraction methods such as Mel-Frequency Cepstral Coefficients (MFCCs) and spectrograms to build robust classification models.
The models are based on Long Short-Term Memory (LSTM) networks, trained on real audio data to perform:

Gender Classification → Male / Female

Age Group Classification →

0 → Age 15 and under

1 → Age 16–40

2 → Age 41 and above

✨ Key Features

Audio Features: MFCCs, spectrograms, and augmentation strategies for robust training.

Gender Classification: Classifies audio samples as Male or Female.

Age Group Classification: Predicts one of the three defined age groups.

Fallback Strategy: Handles ambiguous inputs gracefully to improve reliability.

Data Augmentation: Pitch shifting and time stretching used to expand the dataset and reduce overfitting.

📂 Repository Structure
notebooks/
├── final_model_mfcc40_augmented.ipynb          # 40 MFCCs + data augmentation (best results)
├── baseline_mfcc13_with_fallback.ipynb # 13 MFCCs + fallback strategy + evaluation
├── hyperparam_tuned_mfcc13.ipynb     # 13 MFCCs + optimized parameters
└── experiments/                        # Scratch notebooks & earlier versions

results/
├── predictions.csv
├── predictions_baseline.csv
├── predictions_optimized.csv   

And others for trial purposes.

README.md


Note: The LSTM models are trained directly in the notebooks. No pre-trained model files are included.

🚀 How It Works

Feature Extraction → Extract MFCCs / spectrograms from raw audio using librosa.

Preprocessing → Augment dataset with pitch shifting, time stretching; split into train/validation sets.

Model Training → Train LSTM models with different input sizes (13 vs 40 MFCCs).

Evaluation → Use accuracy, confusion matrix, and CSV results for performance comparison.

🛠️ Tech Stack

Python

TensorFlow / Keras

Librosa

NumPy, Pandas, Matplotlib

📊 Results

Best performance achieved using 40 MFCCs + augmentation (final_model_mfcc40_augmented.ipynb).

Smaller models (13 MFCCs) used for quick experiments and evaluation with fallback strategy.

Robust accuracy across both gender and age group classification tasks.