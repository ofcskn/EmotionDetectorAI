# EmotionDetectorAI

EmotionDetectorAI is an emotion detection application built using deep learning. It utilizes a Convolutional Neural Network (CNN) to classify facial expressions from images or video streams, specifically from webcam input. The model is trained using the FER-2013 dataset from Kaggle to detect emotions such as happy, sad, surprise, fear, anger, disgust, and neutral.

## Features
- Train a model using the FER-2013 dataset from Kaggle.
- Start a webcam application that uses the trained model to predict emotions in real-time.

## Requirements

Before running the application, make sure to install the required dependencies listed in `requirements.txt`. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Train a Model and Start App

In the project folder, run:

```bash
python main.py
```