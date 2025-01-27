# Automatic Hand Tracking

This project utilizes SAM2 and mediapipe, in order to build an automatic pipeline that tracks hand movements in a video.

## Table of Contents

- [Installation](#installation)
- [Part 1 (Find hand locations)](#part1)
- [Part 2 (Create video with hand masks)](#part2)

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

If you are installing on Windows, it's strongly recommended to use Windows Subsystem for Linux (`WSL`) with Ubuntu to run the `.sh` file. Keep in mind for the `.sh` file to work the line endings need to be `LF` not `CRLF` which is the default for Windows system. Therefore you need a text editor to change it to `LF`, save and then run the following commands. Another thing you can do is run the shell script in WSL and drag the `.pt` files in the `checkpoints` folder.

```shell
cd checkpoints
bash download_ckpts.sh
cd ..
```

## Part 1 (Find hand locations)

## Part 2 (Create video with hand masks)


