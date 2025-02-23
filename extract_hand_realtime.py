import cv2
import mediapipe as mp
import os
import numpy as np
import time

# Pathse
video_saving_path = r"D:\SCHOOL WORKS\3RD YEAR\CAPSTONE\KOMUNIKA AI TRAINING\FSLAR - SMALL PROTOTYPE\NUMBERS\NUMBERS\100"
keypoints_saving_path = r"D:\SCHOOL WORKS\3RD YEAR\CAPSTONE\KOMUNIKA AI TRAINING\FSLAR - SMALL PROTOTYPE\NUMBERS\NUMBERS\100\100 keypoints"

# Ensure the output directories exist
if not os.path.exists(video_saving_path):
    os.makedirs(video_saving_path)

if not os.path.exists(keypoints_saving_path):
    os.makedirs(keypoints_saving_path)

# Get the latest video index
existing_videos = [f for f in os.listdir(video_saving_path) if f.startswith("realtime_video_")]
latest_index = 25
if existing_videos:
    latest_index = max([int(f.split('_')[2].split('.')[0]) for f in existing_videos]) + 1

# MediaPipe Hands Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video capture parameters
fps = 30
capture_duration = 2  # seconds
frame_count_target = fps * capture_duration

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start iteration index (useful to resume from a specific point)
start_iteration = latest_index  # Start where we left off
total_iterations = 25

for iteration in range(start_iteration, start_iteration + total_iterations):
    # Preparation time countdown
    for i in range(2, 0, -1):
        # Display countdown on the screen
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame for countdown. Exiting...")
            break
        cv2.putText(frame, f"Get ready... {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('Sign Language Capture', frame)
        cv2.waitKey(1000)  # 1-second delay
    print("Start signing now!")

    # Initialize variables
    frames = []
    keypoints_sequence = []

    # Start capturing frames
    start_time = time.time()
    frame_count = 0

    while frame_count < frame_count_target:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Store the frame
        frames.append(frame)

        # Extract keypoints
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:2]:  # Limit to two hands
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
            # Pad with zeros if only one hand is detected
            if len(results.multi_hand_landmarks) == 1:
                keypoints.extend([np.nan] * 63)  # Pad with NaN for the second hand
        else:
            keypoints = [np.nan] * 126  # No hands detected, pad with NaN

        keypoints_sequence.append(keypoints)

        # Display the frame
        cv2.imshow('Sign Language Capture', frame)

        # Increment frame count
        frame_count += 1

        # Stop capturing if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the captured video
    video_name = f"realtime_video_{iteration}.avi"
    video_path = os.path.join(video_saving_path, video_name)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 480))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved at: {video_path}")

    # Create a subfolder for the keypoints named after the video
    keypoints_subfolder = os.path.join(keypoints_saving_path, os.path.splitext(video_name)[0])
    if not os.path.exists(keypoints_subfolder):
        os.makedirs(keypoints_subfolder)

    # Save keypoints as 'keypoints.npy'
    keypoints_path = os.path.join(keypoints_subfolder, 'keypoints.npy')
    np.save(keypoints_path, np.array(keypoints_sequence, dtype=np.float32))
    print(f"Keypoints saved at: {keypoints_path}")

# Release VideoCapture and destroy windows
cap.release()
hands.close()
cv2.destroyAllWindows()
