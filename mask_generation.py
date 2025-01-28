from show_sam2 import show_mask, visualize_frame
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from cuda_availability import check_cuda
from frame_creation import frame_creation
import matplotlib.pyplot as plt
import cv2
import os

def mask_generation(data_array, output_video_path = "output_video.mp4"):
    
    # Create frames along with frame names
    frames, frame_names = frame_creation()
    
    # Check device uses cdua
    device = check_cuda()

    if device.type == "cuda":
        print("We can use SAM2")

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Build predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=frames)

    # Refine right hand in the first frame and visualize
    ann_frame_idx = 0 
    ann_obj_id = 0

    points = np.array(
        [data_array[ann_frame_idx][0][i].tolist() for i in range(21)],
        dtype=np.float32
    )

    labels = np.array(
        [1] * 21 ,
        np.int32
    )

    box = np.array([
        data_array[ann_frame_idx][0][0][0] - 50,  
        data_array[ann_frame_idx][0][0][1] + 50,  
        data_array[ann_frame_idx][0][12][0] + 50,  
        data_array[ann_frame_idx][0][12][1] - 50   
    ], dtype=np.float32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
    )

    visualize_frame(frames=frames,
                    frame_names=frame_names,
                    points=points,
                    labels=labels,
                    box=box,
                    out_mask_logits=out_mask_logits,
                    out_obj_ids=out_obj_ids,
                    ann_obj_id=ann_obj_id)
    
    # Refine left hand in the first frame and visualize
    ann_frame_idx = 0  
    ann_obj_id = 1

    points = np.array(
        [data_array[ann_frame_idx][1][i].tolist() for i in range(21)],
        dtype=np.float32
    )

    labels = np.array(
        [1] * 21,  
        np.int32
    )

    box = np.array([
        data_array[ann_frame_idx][1][0][0] + 60, 
        data_array[ann_frame_idx][1][0][1] + 30, 
        data_array[ann_frame_idx][1][8][0] - 50, 
        data_array[ann_frame_idx][1][8][1] - 30  
    ], dtype=np.float32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
    )
    
    visualize_frame(frames=frames,
                    frame_names=frame_names,
                    points=points,
                    labels=labels,
                    box=box,
                    out_mask_logits=out_mask_logits,
                    out_obj_ids=out_obj_ids,
                    ann_obj_id=ann_obj_id)
    
    # Propagate to all frames and visualize
    video_segments = {} 
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"Frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(frames, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()
    
    # Write video to output_video_path
    fps = 30
    frame_size = Image.open(os.path.join(frames, frame_names[0])).size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for out_frame_idx in range(len(frame_names)):
        frame_path = os.path.join(frames, frame_names[out_frame_idx])
        frame = np.array(Image.open(frame_path))  

        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                mask = (out_mask * 255).astype(np.uint8)  
                colored_mask = np.zeros_like(frame, dtype=np.uint8)  
                colored_mask[:, :, 1] = mask  

                frame = cv2.addWeighted(frame, 0.8, colored_mask, 0.2, 0)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Segmented video saved to {output_video_path}")
