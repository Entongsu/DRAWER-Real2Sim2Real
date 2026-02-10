import imageio
import os

import cv2


def images_to_video(folder_path, output_video_path, fps=30):
    # Get a sorted list of all image files in the folder
    images = [
        os.path.join(folder_path, img) for img in os.listdir(folder_path)
        if img.endswith(('.png', '.jpg', '.jpeg'))
    ]
    images.sort()  # Ensure correct order (e.g., numerically sorted)

    # Check if there are images
    if not images:
        print("No images found in the folder.")
        return
    writer = imageio.get_writer(output_video_path, fps=fps)

    for index in range(len(images)):
        frame = cv2.imread(folder_path +
                           f"/frame_{index+1:05d}.jpg")  # Read each image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        writer.append_data(frame)  # Add the frame to the video

    print(f"Video saved to {output_video_path}")


# Example usage
folder_path = "/home/ensu/Downloads/bridge_kitchen01/images"
output_video_path = "/home/ensu/Downloads/bridge_kitchen01/polycam_video.gif"
fps = 20  # Adjust frames per second as needed

images_to_video(folder_path, output_video_path, fps)
