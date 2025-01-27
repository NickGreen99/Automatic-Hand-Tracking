import mediapipe as mp
import cv2
import numpy as np

def hand_locations(video_path):
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils  # For visualization
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)

    # List to store the frame data
    frames_data = []

    frame_count = 0  # Counter to track frames processed

    while cap.isOpened() and frame_count < 1:  # Process only the first frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Convert BGR to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Get frame dimensions
        height, width, _ = frame.shape

        # List to store data for hands in this frame
        hands_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                hand_data = []

                # Extract all 21 landmarks
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = int(landmark.x * width)
                    landmark_y = int(landmark.y * height)
                    hand_data.append((landmark_x, landmark_y))

                # Append this hand's data
                hands_data.append(hand_data)

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green dots
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Red connections
                )

        # Append this frame's data
        frames_data.append(hands_data)

        frame_count += 1

    cap.release()
    hands.close()

    # Convert the collected data into a 4D NumPy array
    # The array shape will be (frames, hands, 21, 2) to accommodate the (x,y) values of the 21 mediapipe landmarks 
    max_hands = 2 
    array_shape = (len(frames_data), max_hands, 21, 2)
    data_array = np.zeros(array_shape, dtype=int)

    # Populate the NumPy array
    for frame_idx, hands_data in enumerate(frames_data):
        for hand_idx, hand_data in enumerate(hands_data):
            for landmark_idx, (x, y) in enumerate(hand_data):
                data_array[frame_idx, hand_idx, landmark_idx] = [x, y]

    return data_array
    