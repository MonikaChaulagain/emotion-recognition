
# Emotion Recognition Using CNN (FER2013 Dataset)

This project implements a Convolutional Neural Network (CNN) in PyTorch to recognize human emotions from grayscale facial images, trained on the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset.

## 📌 Overview

- **Goal:** Classify facial expressions into 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **Dataset:** FER2013 (48x48 grayscale facial images, 35,887 samples).
- **Model:** CNN built using `torch.nn`, trained from scratch.
- **Evaluation Metrics:** Accuracy,loss
- **Tools:** PyTorch, Torchvision, Matplotlib

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* Matplotlib
* torchvision


## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/MonikaChaulagain/emotion-recognition.git
cd emotion-recognition
```

2. Run the notebook:

Open `emotion_recognition.ipynb` in Jupyter or VSCode and run all cells to:

* Preprocess the data
* Define and train the CNN
* Evaluate the model performance


## 🧠 Model Architecture (example)

* Conv2D → ReLU 
* Conv2D → ReLU → MaxPool
* Flatten
*  Linear → ReLU
* Linear (Output Layer)


## 🗂️ Dataset Info

* 35,887 grayscale images (48x48 pixels)
* 7 Emotion labels: `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`


## ✅ Future Improvements

* Use pre-trained CNNs (e.g., ResNet)
* Data augmentation for better generalization
* Hyperparameter tuning (learning rate, dropout, etc.)
* Use real-time webcam inference


Let me know if you also want a version that includes sample output images, training plots, or model predictions.
```
