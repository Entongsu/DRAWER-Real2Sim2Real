import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt


def make_dir(file_path):
    file_path = Path(file_path)
    file_path.mkdir(exist_ok=True, parents=True)


class Error:

    def __init__(self,
                 error_type="compounding_error",
                 error_path=None) -> None:
        # if error_type == "compounding_error":
        #     self.error_function = self.get_compounding_error

        self.traj_buffer = []
        self.error = []
        self.error_path = error_path
        self.count_index = 0

        make_dir(self.error_path)

    def get_compounding_error(self, predict_ee, truth_ee):
        self.error.append(np.linalg.norm(predict_ee - truth_ee))
        self.traj_buffer.append([predict_ee, truth_ee])
        return self.error

    def save_error_info(self):
        np.save(f"{self.error_path}/error_{self.count_index}.npy", self.error)
        np.save(f"{self.error_path}/traj_{self.count_index}.npy",
                self.traj_buffer)

    def reset_buffer(self):
        self.traj_buffer = []
        self.error = []
        self.count_index += 1

    def visualize(self, ):
        self.error = np.array(self.error)
        step = np.arange(0, len(self.error), step=1)

        plt.plot(step, self.error)

        # Adding title and labels
        plt.title("compounding_error")
        plt.xlabel("x axis")
        plt.ylabel("y axis")

        # Show the plot
        plt.show()
        self.reset_buffer()
