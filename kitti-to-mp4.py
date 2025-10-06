import cv2
import os

image_folder = '/media/mrt/Whale/data/kitti/07/image_2'
video_name = 'drive.mp4'


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # Important: sort images to maintain correct order

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, channels = frame.shape

# Use MP4V codec for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Alternatively, you can use other codecs like:
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# fourcc = cv2.VideoWriter_fourcc(*'X264')

video = cv2.VideoWriter(video_name, fourcc, 10, (width,height))  # 20 FPS example

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)
    else:
        print(f"Warning: Could not read image {image}")

video.release()
print(f"Video saved as {video_name}")
