# Automatic Hand Tracking

This project utilizes SAM2 and mediapipe, in order to build an automatic pipeline that tracks hand movements in a video.

## Table of Contents

- [Installation](#installation)
- [Part 1 (Find hand locations)](#part-1-find-hand-locations)
- [Part 2 (Create video with hand masks)](#part-2-create-video-with-hand-masks)

## Installation
SAM 2 needs to be installed first before use. Create a conda environment with python `python>=3.10`. Here we create a `conda` environment with `python=3.12` and activate it.

```shell
conda create -n sam2_test python=3.12 -y
conda activate sam2_test
```

Make sure you have `pip` installed to install the requirements.

```shell
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
```

If you are installing on Windows, it's strongly recommended to use Windows Subsystem for Linux (`WSL`) with Ubuntu to run the `.sh` file. Keep in mind for the `.sh` file to work, the line endings need to be `LF` not `CRLF` which is the default for Windows systems. Therefore you need a text editor to change it to `LF`, save and then run the following commands. Another thing you can do is run the shell script in WSL and drag the `.pt` files in the `checkpoints` folder.

```shell
cd checkpoints
bash download_ckpts.sh
cd ..
```

## Part 1 (Find hand locations)

A simple Python function that uses MediaPipe Hands to detect hand landmarks in the first frame of a given video. It returns a NumPy array containing the (x, y) coordinates for up to two hands, each with 21 landmarks.

### Features

One-frame detection: This function only processes the first frame from the specified video file.

Hand landmark detection: Uses MediaPipe to detect landmarks for up to 2 hands.

Coordinates in pixel space: Converts normalized landmark coordinates to (x, y) pixel positions.

Visualization: Internally draws landmarks and connections on the frame (though it returns only the coordinates array).

### How It Works

Initialize MediaPipe Hands: The Hands object is configured to detect up to 2 hands with a minimum detection confidence of 0.5.

Open the video file: cv2.VideoCapture(video_path) loads the video.

Read the first frame:

The function converts the frame to RGB (required by MediaPipe).

MediaPipe processes the frame to find hand landmarks.

Each landmark is stored as (x, y) in pixel coordinates rather than normalized floats.

Draw landmarks: Landmarks are drawn onto the frame (for visualization), but the function ultimately returns only coordinate data.

Return: A NumPy array shaped (1, 2, 21, 2):

1 = The single frame processed.

2 = Up to two hands.

21 = Each hand’s landmarks.

2 = The (x, y) coordinates.

If no hands are detected, the returned array will contain zeros for missing landmarks.

## Part 2 (Create video with hand masks)
dashdjasbdjhdbjsad


