from openpi_client import websocket_client_policy

import numpy as np


class OpenPiClient:

    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=self.host, port=self.port)

    def step(self, example):

        action_chunk = self.policy_client.infer(example)["actions"]

        return action_chunk


if __name__ == "__main__":
    client = OpenPiClient()

    print("testing")
    example = {
        "observation/exterior_image_1_left":
        np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left":
        np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position":
        np.random.rand(7),
        "observation/gripper_position":
        np.random.rand(1),
        "prompt":
        "do something",
    }

    print(client.step(example))

    print("done")
