import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

# Paths
videos_dir = r"D:\SCHOOL WORKS\3RD YEAR\CAPSTONE\KOMUNIKA\FSLAR - SMALL PROTOTYPE\GREETINGS\DOTW_CROPPED_PADDED"
output_keypoints_dir = r"D:\SCHOOL WORKS\3RD YEAR\CAPSTONE\KOMUNIKA\FSLAR - SMALL PROTOTYPE\GREETINGS\KEYPOINT_SEQUENCES_GREETINGS"
labels_csv_path = r"D:\SCHOOL WORKS\3RD YEAR\CAPSTONE\KOMUNIKA\FSLAR - SMALL PROTOTYPE\GREETINGS\LABELS\labels.csv"

# Load labels from CSV
labels_df = pd.read_csv(labels_csv_path)
labels_dict = dict(zip(labels_df['id'], labels_df['label']))

# Initialize MediaPipe Hands (reuse this instance)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract keypoints from videos and save as sequences
def extract_keypoints_from_videos(videos_dir, output_keypoints_dir):
    # Walk through all the label directories
    for label_id in os.listdir(videos_dir):
        label_path = os.path.join(videos_dir, label_id)
        if os.path.isdir(label_path):
            label = labels_dict.get(int(label_id), "Unknown")
            output_label_dir = os.path.join(output_keypoints_dir, label)
            
            # Ensure output label directory exists
            os.makedirs(output_label_dir, exist_ok=True)
            
            # Walk through each video file inside the label folder
            for video_file in os.listdir(label_path):
                video_path = os.path.join(label_path, video_file)
                if os.path.isfile(video_path) and video_file.lower().endswith((".mov", ".avi")):
                    output_video_dir = os.path.join(output_label_dir, os.path.splitext(video_file)[0])
                    
                    # Ensure output video directory exists
                    os.makedirs(output_video_dir, exist_ok=True)
                    
                    keypoints_sequence = []
                    previous_keypoints = [np.nan] * 126  # Default to NaN if no keypoints are detected
                    consecutive_no_hand_frames = 0  # Counter for consecutive frames with no hands
                    
                    # Capture video using OpenCV
                    cap = cv2.VideoCapture(video_path)

                    for frame_count in range(60):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Process each frame to extract hand keypoints
                        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        # Extract keypoints for up to two hands
                        keypoints = []
                        if results.multi_hand_landmarks:
                            # Extract keypoints for both detected hands (up to 2)
                            for hand_landmarks in results.multi_hand_landmarks[:2]:
                                for landmark in hand_landmarks.landmark:
                                    keypoints.extend([landmark.x, landmark.y, landmark.z])  # x, y, z coordinates

                                # Draw hand landmarks on the frame for visualization
                                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            # If only one hand is detected, add NaNs for the second hand
                            if len(results.multi_hand_landmarks) == 1:
                                keypoints += [np.nan] * 63  # Add 63 NaN values for the missing hand

                            # Update the previous keypoints to use in case of lost detection
                            previous_keypoints = keypoints
                            consecutive_no_hand_frames = 0  # Reset no-hand frame counter
                        else:
                            # No hands detected, use the previous keypoints
                            consecutive_no_hand_frames += 1
                            if consecutive_no_hand_frames >= 10:
                                # After 10 frames without hands, assume the gesture ended and use NaNs
                                keypoints = [np.nan] * 126
                            else:
                                # Otherwise, use previous keypoints to maintain continuity
                                keypoints = previous_keypoints

                        # Ensure keypoints length is always 126 before appending
                        if len(keypoints) == 126:
                            keypoints_sequence.append(keypoints)
                        else:
                            print(f"Warning: Keypoints length mismatch for frame {frame_count}. Expected 126, but got {len(keypoints)}")

                        # Display the frame with keypoints
                        cv2.imshow('Hand Keypoints Extraction', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Release the video capture
                    cap.release()
                    cv2.destroyAllWindows()

                    # If the sequence length is less than 60, pad with NaNs
                    while len(keypoints_sequence) < 60:
                        keypoints_sequence.append([np.nan] * 126)

                    # Convert keypoints_sequence to a NumPy array with a consistent data type
                    keypoints_sequence_array = np.array(keypoints_sequence, dtype=np.float32)
                    
                    # Save the keypoints sequence as an NPY file in the appropriate folder
                    output_keypoints_path = os.path.join(output_video_dir, "keypoints.npy")
                    np.save(output_keypoints_path, keypoints_sequence_array)
                    
                    print(f"Extracted keypoints sequence for '{label}/{video_file}' and saved to '{output_keypoints_path}'.")

# Start extracting keypoints from videos
extract_keypoints_from_videos(videos_dir, output_keypoints_dir)

# Close the MediaPipe hands instance
hands.close()
