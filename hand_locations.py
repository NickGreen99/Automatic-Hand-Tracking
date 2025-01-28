import mediapipe as mp
import cv2
import numpy as np

def hand_locations(video_path):
    
    # Use mediapipe to find hand locations
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)

    frames_data = []

    frame_count = 0

    while cap.isOpened() and frame_count < 1:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        height, width, _ = frame.shape

        hands_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                hand_data = []

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = int(landmark.x * width)
                    landmark_y = int(landmark.y * height)
                    hand_data.append((landmark_x, landmark_y))


                hands_data.append(hand_data)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green dots
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Red connections
                )

        frames_data.append(hands_data)

        frame_count += 1

    cap.release()
    hands.close()

    # 4D NumPy array
    max_hands = 2 
    array_shape = (len(frames_data), max_hands, 21, 2)
    data_array = np.zeros(array_shape, dtype=int)

    for frame_idx, hands_data in enumerate(frames_data):
        for hand_idx, hand_data in enumerate(hands_data):
            for landmark_idx, (x, y) in enumerate(hand_data):
                data_array[frame_idx, hand_idx, landmark_idx] = [x, y]

    return data_array
    