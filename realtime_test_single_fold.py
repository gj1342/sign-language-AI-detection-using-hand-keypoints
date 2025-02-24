import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp

# Paths
ensemble_model_dir = r"directory of the saved ensemble"
labels_paths = r"directory of the saved ensemble"
labels_path = os.path.join(labels_paths, 'model_labels.txt')
SEQUENCE_LENGTH = 60
NUM_KEYPOINTS = 126  # 42 points (21 from each hand) * 3 (x, y, z)

# Load labels from the text file
with open(labels_path, 'r') as file:
    LABELS = [line.strip() for line in file.readlines()]

# Load the single model
model_path = os.path.join(ensemble_model_dir, 'model_fold_3.h5')
model = load_model(model_path)

# Mediapipe for extracting hand keypoints from webcam
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand keypoints from the mediapipe hands model result
def extract_hand_keypoints(results):
    # Initialize empty keypoints for left and right hands
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if the hand is left or right
            label = handedness.classification[0].label
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                lh = keypoints
            elif label == 'Right':
                rh = keypoints

    return np.concatenate([lh, rh])

# Real-time Prediction Loop
cap = cv2.VideoCapture(0)
sequence = []
prediction = ""
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = hands.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints only if hands are detected
            keypoints = extract_hand_keypoints(results)
            sequence.append(keypoints)

            # Maintain the sequence length of SEQUENCE_LENGTH
            if len(sequence) == SEQUENCE_LENGTH:
                # Prepare input data for the model
                input_data = np.expand_dims(np.array(sequence), axis=0)

                # Get prediction from the model
                result = model.predict(input_data)
                prediction = LABELS[np.argmax(result[0])]

                # Reset the sequence to start again
                sequence = []
        else:
            # If no hands are detected, clear the sequence
            sequence = []

        # Display prediction on the frame if available
        if prediction:
            cv2.putText(image, f'Prediction: {prediction}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Sign Language Recognition', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
