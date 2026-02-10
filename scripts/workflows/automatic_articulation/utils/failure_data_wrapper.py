from scripts.workflows.automatic_articulation.utils.data_wrapper import Datawrapper
from scripts.workflows.automatic_articulation.utils.map_env import step_buffer_map, reset_buffer_map, reset_data_buffer, load_config

from scripts.workflows.automatic_articulation.task.failure_env_grasp import FailureEnvGrasp
from scripts.workflows.automatic_articulation.task.failure_env_cabinet import FailureEnvCabinet
import torch
import numpy as np
import copy


class FailureDatawrapper(Datawrapper):

    def __init__(self,
                 env,
                 collision_checker,
                 use_relative_pose,
                 init_grasp,
                 init_open,
                 init_placement,
                 init_close,
                 collect_data=None,
                 args_cli=None,
                 env_config=None,
                 filter_keys=None,
                 load_path=None,
                 save_path=None,
                 use_fps=False,
                 use_joint_pos=False,
                 use_demo_data=True,
                 faiulure_type=None,
                 failure_attempt=1):
        # failure type is the type of failure to be simulated: [x_offset,y_offset,z_offset,roll_offset,pitch_offset,yaw_offset,slip,miss_grasp,wrong_sequeeze,]
        super().__init__(env,
                         collision_checker=collision_checker,
                         use_relative_pose=use_relative_pose,
                         init_grasp=init_grasp,
                         init_open=init_open,
                         init_placement=init_placement,
                         init_close=init_close,
                         collect_data=collect_data,
                         args_cli=args_cli,
                         env_config=env_config,
                         filter_keys=filter_keys,
                         use_joint_pos=use_joint_pos,
                         load_path=load_path,
                         use_fps=use_fps,
                         use_demo_data=use_demo_data,
                         save_path=save_path)
        self.collect_data = collect_data
        self.failure_attempt = failure_attempt

        if self.init_grasp and self.init_placement:
            self.failure_env = FailureEnvGrasp(env=self.multi_env,
                                               failure_type=faiulure_type)
        if self.init_open:
            self.failure_env = FailureEnvCabinet(env=self.multi_env,
                                                 failure_type=faiulure_type)
        self.failure_type = faiulure_type
        self.start_frame = self.failure_env.start_frame

        self.assign_failure()

    def assign_failure(self):

        if self.failure_type == "slip":
            self.extract_failure_action = self.failure_env.slip_failure_env
        if "offset" in self.failure_type:
            self.extract_failure_action = self.failure_env.xyz_offset_failure_env
        if self.failure_type == "miss_placement":
            self.extract_failure_action = self.failure_env.miss_placement_failure_env
        if self.failure_type == "miss_grasp":
            self.extract_failure_action = self.failure_env.miss_grasp_failure_env
        if self.failure_type == "miss_grasp":
            self.extract_failure_action = self.failure_env.miss_grasp_failure_env

    def step_failure_env_espisode(self, skip_frame=2):
        last_obs, _ = self.forward_manipulate_env()
        failure_action = self.extract_failure_action(demo=self.demo)
        if failure_action is None:
            return False
        failure_action, failure_reasoning = failure_action[::skip_frame]
        collect_cabinet, collect_close, collect_grasp, collect_placement = self.init_open, self.init_close, self.init_grasp, False

        for i in range(0, len(failure_action)):

            observation, reward, terminate, time_out, info, actions, success = self.multi_env.step_manipulate(
                failure_action[i],
                collect_grasp=self.init_grasp and not self.init_placement,
                collect_placement=self.init_placement,
                collect_cabinet=self.init_open,
                collect_close=self.init_close)

            for flag, buffers in step_buffer_map.items():
                if locals(
                )[flag]:  # Dynamically check if the corresponding flag is True
                    getattr(self, buffers[0]).append(
                        last_obs["policy"])  # Observation

                    getattr(self, buffers[1]).append(
                        failure_action[i].unsqueeze(0))  # Actions
                    getattr(self, buffers[2]).append(reward)  # Rewards
                    getattr(self,
                            buffers[3]).append(terminate)  # Termination status
            self.failure_env.episode_check_success(last_obs)
            last_obs = observation
        return not self.failure_env.check_success(self, last_obs)

    def step_failure_env(self, skip_frame=2):
        failure_count = 0
        success_buffer = None
        while failure_count < self.failure_attempt:

            # run the success data collection
            if success_buffer is None:
                print(
                    f"Running success data collection for demo {self.demo_index}"
                )
                success, success_buffer = self.step_unnormalized_env(
                    skip_frame=skip_frame, reset_every_step=False)

                if not success:
                    reset_data_buffer(self,
                                      reset_grasp=self.init_grasp,
                                      reset_cabinet=self.init_open,
                                      reset_close=self.init_close,
                                      reset_placement=False)
                    self.demo_index += 1
                    return None

            failure = self.step_failure_env_espisode(skip_frame=skip_frame)
            if failure:
                self.cache_data()
            failure_count += 1

            reset_data_buffer(
                self,
                reset_grasp=self.init_grasp,
                reset_cabinet=self.init_open,
                reset_close=self.init_close,
                reset_placement=False,
                presave_buffer=None if failure_count == self.failure_attempt
                else copy.deepcopy(success_buffer))
        self.demo_index += 1
