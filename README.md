# Face Emotion Detection with Emoji Overlay

A real-time facial emotion recognition system that detects human emotions from webcam input and overlays corresponding emojis for visual feedback.

The project uses OpenCV for face detection and a CNN-based deep learning model trained on the FER2013 dataset for emotion classification.

---

## Features

- Real-time webcam emotion detection
- CNN-based facial emotion classification
- Emoji overlay visualization
- Live emotion label display
- FER2013-trained emotion recognition model

---

## Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy

---

## Project Structure

```text
face-emotion-detection-with-emoji-overlay/
│
├── src/
│   └── main.py
│
├── model/
│   ├── emotiondetector.json
│   ├── emotiondetector.h5
│
├── assets/
│   └── emoji/
│
├── notebooks/
│   └── train_model.ipynb
│
├── requirements.txt
└── README.md
```

---

## Workflow

1. Capture webcam feed
2. Detect face using Haar Cascade classifier
3. Preprocess grayscale face image
4. Run CNN inference for emotion classification
5. Predict emotion label
6. Overlay matching emoji onto webcam feed

---


## Dataset

The emotion classification model was trained using the FER2013 facial emotion recognition dataset.

---

## Challenges Faced

- maintaining real-time inference performance
- handling varying lighting conditions
- balancing model accuracy with responsiveness
- preprocessing consistency across webcam frames

---

## Learning Outcomes

This project strengthened my understanding of:

- CNN inference workflows
- real-time computer vision systems
- image preprocessing
- facial emotion recognition pipelines
- integrating ML models into interactive applications
