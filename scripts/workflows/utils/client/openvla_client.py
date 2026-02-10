from gradio_client import Client, handle_file
from PIL import Image
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(
        img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False,
                             dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


class OpenVLAClient:

    def __init__(self, url=None):
        self.url = url
        # Initialize the client
        self.client = Client(self.url)

    def step(self, prompt, image):

        # Crop and then resize back to original size
        image = resize_image(image, IMAGE_SIZE)

        data = {
            "instruction": prompt,
            "image": image.tolist(),
            "unnorm_key": "bridge_orig"
        }
        response = self.client.predict(
            data,
            api_name="/predict"  # Optional: Specify the API name if needed
        )
        response["action"][-1] = response["action"][-1] * 2 - 1

        print("Response:", response)

        response["image"] = image
        return response


# openvla_client = OpenVLAClient("https://883b46f492667a2f4c.gradio.live")
# for i in range(10):
#     openvla_client.step(
#         "In: What action should the robot take to pick up the eggplant?\nOut:",
#         np.array(
#             Image.open(
#                 "/home/ensu/Documents/weird/IsaacLab/logs/gs_image_0.png")))
# url = "https://2a59ea9dd953324a2f.gradio.live"
# files = {
#     "image":
#     Image.open("/home/ensu/Documents/weird/IsaacLab/logs/gs_image_0.png")
# }
# data = {
#     "prompt":
#     "In: What action should the robot take to pick up the eggplant?\nOut:"
# }

# response = requests.post(url, data=data, files=None)
# if response.status_code == 200:
#     print(response.json())
# else:
#     print(f"Error: {response.status_code}, {response.text}")
