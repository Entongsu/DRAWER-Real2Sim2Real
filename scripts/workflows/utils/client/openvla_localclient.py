import json_numpy

json_numpy.patch()

import requests
import urllib
import numpy as np
import cv2
import tensorflow as tf


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
    RESIZE_SIZE = 224

    def __init__(self, url="http://localhost:8000/"):
        self.url = url

    def reset(self, task_description):
        pass

    def step(self, prompt, image, unnorm_key="bridge_orig"):
        # crop
        # image = cv2.resize(image, (256, 256))
        # crop 480 x 640 -> 480 x 480
        # image = image[:, 80:560]
        # image = cv2.resize(image, (256, 256))

        # image = resize_image(image, (self.RESIZE_SIZE, self.RESIZE_SIZE))

        data = {
            "image": image,
            "instruction": prompt,
            "unnorm_key": unnorm_key
        }

        action = np.array(
            requests.post(
                "http://localhost:8000/act",
                json=data,
            ).json())
        response = {}

        response["action"] = action
        print("Response:", response)
        response["image"] = image

        return response


if __name__ == "__main__":
    client = OpenVLAClient()

    print("testing")
    print(client.reset("pick up the red block"))

    print(
        client.step(
            "pick up the red block",
            np.random.rand(256, 256, 3).astype(np.uint8),
        ))
    print(
        client.step(
            "pick up the red block",
            np.random.rand(256, 256, 3).astype(np.uint8),
        ))

    print("done")
