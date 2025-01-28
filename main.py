from hand_locations import hand_locations
from mask_generation import mask_generation

def main():
    video_path = "test.mp4"

    # Part 1 (Creates numpy array of hand locations and landmarks)
    data_array = hand_locations(video_path)

    # Part 2 (Generates the masks to detect the hands in the video and writes a video of the masks in action)
    mask_generation(data_array)

if __name__ == '__main__':
    main()