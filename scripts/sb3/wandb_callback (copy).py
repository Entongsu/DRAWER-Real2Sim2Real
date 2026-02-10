import logging
import os
from typing import Optional, Callable, List

import numpy as np
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

from collections import deque

logger = logging.getLogger(__name__)
#=====================tactile===========================
import torch

import cv2

import matplotlib.pyplot as plt
import copy
from stable_baselines3.common.callbacks import BaseCallback

from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union


class KVWriter:
    """
    Key Value writer
    """

    def write(self,
              key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step:
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


class WandbOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    :param folder: the folder to write the log to
    """

    def __init__(self):
        pass

    def write(self,
              key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:

        write_dict = {}
        for (key, value), (_, excluded) in zip(sorted(key_values.items()),
                                               sorted(key_excluded.items())):

            if excluded is not None and "wandb" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                write_dict[key] = value

            if isinstance(value, torch.Tensor):
                write_dict[key] = value.item()

        # Flush the output to the file
        # wandb.log(write_dict, step=step)
        # import pdb
        # pdb.set_trace()

    def close(self) -> None:
        """
        closes the file
        """
        pass


import logging

logging.getLogger("wandb").setLevel(logging.ERROR)


class WandbCallback(BaseCallback):
    """Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 100,
        eval_freq: Optional[int] = None,
        eval_env_fn: Optional[Callable] = None,
        eval_cam_names: Optional[List[str]] = None,
        viz_point_cloud=False,
        gradient_save_freq: int = 0,
        success_threshold=None,
    ):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error(
                "You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        self.eval_freq = eval_freq
        self.eval_env_fn = eval_env_fn
        self.eval_cam_names = eval_cam_names
        self.viz_point_cloud = viz_point_cloud
        self.success_threshold = success_threshold
        self.success_counter = {}

        self.gradient_save_freq = gradient_save_freq

        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

        self.roll_out = 0

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(self.model.policy,
                        log_freq=self.gradient_save_freq,
                        log="all")
        wandb.config.setdefaults(d)

    def _on_rollout_end(self, save_path=False) -> None:
        if save_path:
            self.save_model()
            return

        need_restore = self.model.__dict__.get("need_restore", False)
        current_restore_step = self.model.__dict__.get("current_restore_step",
                                                       0)
        # wandb.log({"rollout/restore": current_restore_step},
        #           step=self.roll_out + 1)
        if need_restore and current_restore_step <= 5:
            return

        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.roll_out % self.model_save_freq == 0:
                    self.save_model()
        if self.eval_env_fn is not None:
            if self.eval_env_fn.env.demo_begin_frame not in self.success_counter.keys(
            ):
                self.success_counter[self.eval_env_fn.env.demo_begin_frame] = 0

            if self.eval_env_fn.env.ready_eval:
                print("Using deterministic evaluation success_rate: ",
                      self.eval_env_fn.env.eval_success_rate,
                      "demo_begin_frame: ",
                      self.eval_env_fn.env.demo_begin_frame)
                self.eval_env_fn.env.ready_eval = False
                if self.eval_env_fn.env.eval_success_rate > self.success_threshold[
                        1]:
                    self.eval_env_fn.env.reset_demo_begin_frame = True
                    demo_begin_frame = self.eval_env_fn.env.demo_begin_frame
                    path = os.path.join(
                        self.model_save_path,
                        f"demo_begin_frame_{demo_begin_frame}_model_{self.roll_out}_success_rate_{self.eval_env_fn.env.eval_success_rate}.pt"
                    )
                    self.model.save(path)

            # infer stochastic evaluation only when the success rate is above the threshold
            if self.eval_env_fn is not None and self.eval_env_fn.env.success_rate > 0:
                if self.eval_env_fn.env.demo_begin_frame not in self.success_counter.keys(
                ):
                    self.success_counter[
                        self.eval_env_fn.env.demo_begin_frame] = 0

                if self.eval_env_fn.env.success_rate > self.success_threshold[
                        1] or self.success_counter[
                            self.eval_env_fn.env.
                            demo_begin_frame] >= self.success_threshold[2]:
                    self.eval_env_fn.env.reset_demo_begin_frame = True

                    success_rate = self.eval_env_fn.env.success_rate
                    demo_begin_frame = self.eval_env_fn.env.demo_begin_frame
                    path = os.path.join(
                        self.model_save_path,
                        f"demo_begin_frame_{demo_begin_frame}_model_{self.roll_out}_success_rate_{success_rate}.pt"
                    )
                    print("Using stochastic evaluation success_rate: ",
                          success_rate, "demo_begin_frame: ",
                          self.eval_env_fn.env.demo_begin_frame)
                    self.model.save(path)
                    self.eval_env_fn.env.success_rate = 0.0
                elif self.eval_env_fn.env.success_rate > self.success_threshold[
                        0]:
                    self.success_counter[
                        self.eval_env_fn.env.demo_begin_frame] += 1

        self.current_restore_step = 0
        self.roll_out += 1

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        path = os.path.join(self.model_save_path, f"model_{self.roll_out}")

        self.model.save(path)
        wandb.save(path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info("Saving model checkpoint to " + path)

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True


def setup_wandb(parser_config, exp_name, tags=None, project="isaaclab"):
    run = wandb.init(
        project=project,
        name=exp_name,
        config=parser_config,
        monitor_gym=True,
        save_code=False,  # optional
        tags=tags,
        entity="entongsu",
        resume="allow")
    return run
