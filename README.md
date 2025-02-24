# sign-language-AI-detection-using-hand-keypoints

A simple sign language hand detection process from extracting hand keypoints to real-time testing.

## Overview
This repository provides a complete pipeline for sign language recognition using hand keypoints. It includes scripts for extracting hand keypoints from videos or in real-time, training machine learning models on the extracted keypoints, and testing models for real-time sign language recognition.

## Project Structure

- `extract_hand_kp.py`: Extracts hand keypoints from pre-recorded videos.
- `extract_hand_realtime.py`: Extracts hand keypoints in real-time using a webcam.
- `model_training_ensemble.py`: Trains an ensemble model using the extracted keypoints.
- `realtime_test_ensemble.py`: Tests real-time sign language prediction using an ensemble of models.
- `realtime_test_single_fold.py`: Tests real-time sign language prediction using a single trained model fold.

## Requirements
- tensorflow: 2.15.1
- keras: 2.15.0
- opencv-python: 4.11.0.86
- mediapipe: 0.10.20

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install mediapipe numpy pandas tensorflow keras opencv-python matplotlib
```

## Usage

### Extract Hand Keypoints from Videos
```bash
python extract_hand_kp.py --input path/to/video.mp4 --output path/to/output.json
```

### Extract Hand Keypoints in Real-Time
```bash
python extract_hand_realtime.py
```

### Train the Model
```bash
python model_training_ensemble.py --data path/to/training_data.json --epochs 50
```

### Test Real-Time Prediction using Ensemble Model
```bash
python realtime_test_ensemble.py
```

### Test Real-Time Prediction using Single Fold Model
```bash
python realtime_test_single_fold.py
```

## How It Works
1. **Hand Keypoint Extraction**: Using MediaPipe, the script detects hand landmarks and extracts keypoint coordinates.
2. **Model Training**: The extracted keypoints are used to train a model capable of recognizing different sign language gestures.
3. **Real-Time Testing**: The trained model is used for real-time sign recognition, either using an ensemble approach or a single-fold model.

## Contributing
If you find any issues or have suggestions for improvement, feel free to create an issue or submit a pull request.

## License
This project is licensed under the MIT License. Feel free to use and modify it for your own projects.

---

Happy coding! 🚀

