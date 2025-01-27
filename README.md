# Automatic Hand Tracking

This project utilizes SAM2 and mediapipe, in order to build an automatic pipeline that tracks hand movements in a video.

## Table of Contents

- [Installation](#installation)
- [Part 1 (Find hand locations)](#part-1-find-hand-locations)
- [Part 2 (Create video with hand masks)](#part-2-create-video-with-hand-masks)

## Installation
SAM 2 needs to be installed first before use. We need `torch>=2.5.1`, as well as `python>=3.10`. Create a conda environment with python `python>=3.10`. Here we create a `conda` environment with `python=3.12` and activate it.

```shell
conda create -n sam2_test python=3.12 -y
conda activate sam2_test
```

Make sure you have `pip` installed to install the requirements.

```shell
git clone https://github.com/NickGreen99/Automatic-Hand-Tracking.git

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

This function processes the first frame of a video to detect hand landmarks using MediaPipe. It detects up to 2 hands with a minimum detection and tracking confidence of 0.5. The frame is first converted to RGB (required by MediaPipe), and the landmarks for each hand are detected. These landmarks are then converted to pixel coordinates (x, y). Even though the function draws landmarks on the image for visualization, it returns only the coordinate data. The output is a NumPy array with shape `(1, 2, 21, 2)` where the dimensions correspond to one frame, up to two hands, 21 landmarks per hand, and two coordinates (x, y). If no hands are detected, the returned array contains zeros for missing landmarks. for 10 seconds.

This function processes the first frame of a given video to detect up to two hands using MediaPipe, converting normalized hand landmarks to pixel coordinates. If no hands are detected, the output contains zeros for missing landmarks. It internally draws landmarks on the frame, but returns a NumPy array of shape (1, 2, 21, 2), meaning one frame, up to two hands, 21 landmarks per hand, and (x, y) coordinates for each. The frame is converted to RGB before processing, with detection set to a minimum confidence of 0.5. This makes it a quick way to retrieve basic hand position data from a single frame.
| Landmark Index | Name       | Typical Location                           |
|---------------:|:-----------|:-------------------------------------------|
| 0             | WRIST      | Base of the palm                           |
| 1             | THUMB_CMC  | Thumb carpometacarpal joint                |
| 2             | THUMB_MCP  | Thumb metacarpophalangeal joint            |
| 3             | THUMB_IP   | Thumb interphalangeal joint                |
| 4             | THUMB_TIP  | Tip of the thumb                           |
| 5             | INDEX_MCP  | Index finger metacarpophalangeal joint     |
| 6             | INDEX_PIP  | Index finger proximal interphalangeal joint|
| 7             | INDEX_DIP  | Index finger distal interphalangeal joint  |
| 8             | INDEX_TIP  | Tip of the index finger                    |
| 9             | MIDDLE_MCP | Middle finger MCP joint                    |
| 10            | MIDDLE_PIP | Middle finger PIP joint                    |
| 11            | MIDDLE_DIP | Middle finger DIP joint                    |
| 12            | MIDDLE_TIP | Tip of the middle finger                   |
| 13            | RING_MCP   | Ring finger MCP joint                      |
| 14            | RING_PIP   | Ring finger PIP joint                      |
| 15            | RING_DIP   | Ring finger DIP joint                      |
| 16            | RING_TIP   | Tip of the ring finger                     |
| 17            | PINKY_MCP  | Little finger MCP joint                    |
| 18            | PINKY_PIP  | Little finger PIP joint                    |
| 19            | PINKY_DIP  | Little finger DIP joint                    |
| 20            | PINKY_TIP  | Tip of the little finger                   |


## Part 2 (Create video with hand masks)
dashdjasbdjhdbjsad


