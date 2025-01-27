from show_sam2 import show_box, show_mask, show_points
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from cuda_availability import check_cuda
import matplotlib.pyplot as plt
import cv2
import os

def mask_generation(frames, frame_names, data_array):
    device = check_cuda()
    if device.type == "cuda":
        print("We can use SAM2")


    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    inference_state = predictor.init_state(video_path=frames)

    prompts={}
    ann_frame_idx = 0 
    ann_obj_id = 4  

    # Extract all 21 Mediapipe landmarks for the first hand (data_array[ann_frame_idx][0])
    points = np.array(
        [data_array[ann_frame_idx][0][i].tolist() for i in range(21)],
        dtype=np.float32
    )

    labels = np.array(
        [1] * 21 ,
        np.int32
    )

    # Define the bounding box around the hand (slightly padded)
    box = np.array([
        data_array[ann_frame_idx][0][0][0] - 50,  # x_min (wrist landmark x minus padding)
        data_array[ann_frame_idx][0][0][1] + 50,  # y_min (wrist landmark y plus padding)
        data_array[ann_frame_idx][0][12][0] + 50,  # x_max (middle finger tip x plus padding)
        data_array[ann_frame_idx][0][12][1] - 50   # y_max (middle finger tip y minus padding)
    ], dtype=np.float32)

    # Add points, labels, and box to SAM2 predictor
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
    )

    # Visualize the results
    plt.figure(figsize=(9, 6))
    plt.title(f"Frame {ann_frame_idx} (Right Hand)")
    plt.imshow(Image.open(os.path.join(frames, frame_names[ann_frame_idx])))
    show_box(box, plt.gca())  # Show the bounding box
    show_points(points, labels, plt.gca())  # Show the points
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)  # Show the mask
    plt.show()
    
    ann_frame_idx = 0  
    ann_obj_id = 8  

    # Extract all 21 Mediapipe landmarks for the second hand (data_array[ann_frame_idx][1])
    points = np.array(
        [data_array[ann_frame_idx][1][i].tolist() for i in range(21)],
        dtype=np.float32
    )

    labels = np.array(
        [1] * 21,  
        np.int32
    )

    # Define the bounding box around the second hand (slightly padded)
    box = np.array([
        data_array[ann_frame_idx][1][0][0] + 60,  # x_min (wrist landmark x minus padding)
        data_array[ann_frame_idx][1][0][1] + 30,  # y_min (wrist landmark y minus padding)
        data_array[ann_frame_idx][1][8][0] - 50,  # x_max (middle finger tip x plus padding)
        data_array[ann_frame_idx][1][8][1] - 30   # y_max (middle finger tip y plus padding)
    ], dtype=np.float32)

    # Add points, labels, and box to SAM2 predictor
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
    )
    
    # Visualize the results
    plt.figure(figsize=(9, 6))
    plt.title(f"Frame {ann_frame_idx} (Left Hand)")
    plt.imshow(Image.open(os.path.join(frames, frame_names[ann_frame_idx])))
    show_box(box, plt.gca())  # Show the bounding box
    show_points(points, labels, plt.gca())  # Show the points
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)  # Show the mask
    plt.show()
    

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(frames, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    # Define output video parameters
    output_video_path = "output_video.mp4"
    fps = 30  # Set frames per second for the output video
    frame_size = Image.open(os.path.join(frames, frame_names[0])).size  # (width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Overlay masks and write frames to video
    for out_frame_idx in range(len(frame_names)):
        frame_path = os.path.join(frames, frame_names[out_frame_idx])
        frame = np.array(Image.open(frame_path))  # Open frame as a NumPy array

        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                # Overlay mask
                mask = (out_mask * 255).astype(np.uint8)  # Convert mask to 8-bit
                colored_mask = np.zeros_like(frame, dtype=np.uint8)  # Create a colored mask
                colored_mask[:, :, 1] = mask  # Green channel for mask (adjust as needed)

                # Blend the mask with the frame
                frame = cv2.addWeighted(frame, 0.8, colored_mask, 0.2, 0)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Segmented video saved to {output_video_path}")
