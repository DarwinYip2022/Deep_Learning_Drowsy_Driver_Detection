import cv2
import sys
import os

# Check if a file path was provided as a command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_video>")
    sys.exit()

video_path = sys.argv[1]
video = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not video.isOpened():
    print(f"Failed to open video file: {video_path}")
    sys.exit()

save_dir = "Outputs"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1
print(f"Total {frame_count} frames were extracted and saved")
video.release()


