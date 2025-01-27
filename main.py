import os
from cuda_availability import check_cuda
from hand_locations import hand_locations
import cv2
from mask_generation import mask_generation

video_path = "test.mp4"
frames_dir = "frames" 

if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video loaded successfully.")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_idx += 1

cap.release()
print(f"Extracted {frame_idx} frames to '{frames_dir}'.")

frame_names = [
    p for p in os.listdir(frames_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Part 1 (Creates numpy array of hand locations and landmarks)
data_array = hand_locations(video_path)

# Part 2 (Generates the masks to detect the hands in the video and writes a video of the masks in action)
mask_generation(frames_dir, frame_names, data_array)
