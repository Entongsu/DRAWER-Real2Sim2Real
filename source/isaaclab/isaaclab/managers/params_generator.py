import torch
import numpy as np


class ParamsGenerator:

    def __init__(self,
                 defomrable_parames,
                 defomrable_parames_range,
                 device="cpu"):

        self.defomrable_parames = defomrable_parames
        self.defomrable_parames_range = defomrable_parames_range
        self.device = device
        self.set_up()

        self.random_method = "uniform"
        self.sim_distri = None
        self.lows()
        self.highs()

    def create_init_video_params(self, num_envs):
        # Define the possible values for each parameter
        values = np.linspace(0.1, 1, 3)  # Generates [0.0, 0.33, 0.66, 0.99]

        # Use nested loops to generate combinations
        self.params_range = []
        for i, i_value in enumerate(values):
            for j, j_value in enumerate(values):
                if i * 3 + j >= num_envs:
                    return np.array(self.params_range)
                self.params_range.append([i_value, j_value])
        return np.array(self.params_range)

    def set_up(self):

        self.all_para_num = []
        self.all_para_range = []
        self.parames_name = []

        for key in self.defomrable_parames.keys():
            # Set the attribute on the instance with the value and length of the corresponding array
            value_array = np.array(self.defomrable_parames[key])
            value_len = len(value_array)
            setattr(self, f'{key}', value_array)
            setattr(self, f'{key}_num', value_len)

            # Store the values in all_para_num and all_para_range lists
            self.all_para_num.append(value_len)
            self.all_para_range.append(value_array)
            self.parames_name.append(key)

        # Convert all_para_num to a numpy array for consistency
        self.all_para_num = np.array(self.all_para_num)

    def dynamic_friction(self):
        return np.array(self.defomrable_parames["dynamic_friction"]), len(
            np.array(self.defomrable_parames["dynamic_friction"]))

    def youngs_modulus(self):
        return np.array(self.defomrable_parames["youngs_modulus"]), len(
            np.array(self.defomrable_parames["youngs_modulus"]))

    def poissons_ratio(self):
        return np.array(self.defomrable_parames["poissons_ratio"]), len(
            np.array(self.defomrable_parames["poissons_ratio"]))

    def elasticity_damping(self):
        return np.array(self.defomrable_parames["elasticity_damping"]), len(
            np.array(self.defomrable_parames["elasticity_damping"]))

    def damping_scale(self):
        return np.array(self.defomrable_parames["damping_scale"]), len(
            np.array(self.defomrable_parames["damping_scale"]))

    def density(self):
        return np.array(self.defomrable_parames["density"]), len(
            np.array(self.defomrable_parames["density"]))

    def map_value_to_paras(self, params, num_envs, reset=True):

        # Calculate indices and corresponding parameter values
        self.params_range = np.clip(self.params_range, 1e-6, 0.99)
        self.param = np.clip((np.floor(
            self.params_range / (1 / self.all_para_num))).astype(np.int16), 0,
                             10000)
        param_value = (self.params_range %
                       (1 / self.all_para_num)) / (1 / self.all_para_num)

        self.classifer = np.zeros(self.param.shape[0])
        # Add the weighted sums of the previous elements
        for i in range(self.param.shape[1]):

            if i == self.param.shape[1] - 1:
                self.classifer += self.param[:, -1]
            else:
                self.classifer += self.param[:, i] * np.sum(
                    self.all_para_num[i + 1:])

        # Initialize list of dictionaries for each environment

        dict = [{} for _ in range(num_envs)]
        if reset:
            self.step_param_range_index = []
            self.step_param_value = []

        for index, name in enumerate(self.parames_name):

            # Retrieve the range for the current parameter
            range_index = np.array(
                self.all_para_range[index][self.param[:, index]])

            # Calculate the randomized parameter values using broadcasting
            randomized_value = param_value[:, index] * (
                range_index[:, 1] - range_index[:, 0]) + range_index[:, 0]
            if name == "youngs_modulus":
                randomized_value = np.exp(randomized_value) * 5000

            # if name == "dynamic_friction":
            #     randomized_value = np.exp(randomized_value)

            # Update all dictionaries at once using list comprehension
            [
                dict[env_idx].update({name: randomized_value[env_idx]})
                for env_idx in range(num_envs)
            ]
            if reset:
                # Store the range used
                self.step_param_range_index.append(self.param[:, index])
                self.step_param_value.append(randomized_value)
        if reset:
            self.step_param_range_index = np.array(self.step_param_range_index)
            self.step_param_value = np.array(self.step_param_value)

        return dict

    def lows(self):
        self.low_range = []
        for key in self.defomrable_parames.keys():
            self.low_range.append(self.defomrable_parames_range[key][0])
        self.low_range = np.array(self.low_range)
        return self.low_range

    def highs(self):
        self.high_range = []
        for key in self.defomrable_parames.keys():
            self.high_range.append(self.defomrable_parames_range[key][1])
        self.high_range = np.array(self.high_range)
        return self.high_range

    def uniform_random(self, num_envs):

        self.params_range = np.clip(
            np.random.uniform(0, 1, (num_envs, len(self.all_para_num))) *
            (np.array(self.high_range) - np.array(self.low_range)) +
            np.array(self.low_range), 1e-6, 100)
        return self.params_range

    def gen(self, num_envs):

        self.params_range = self.sim_distri.gen(num_envs)
        self.params_range = np.clip(self.params_range, 0.0, 1.0)

    def step_randomize(self, num_envs):
        # Randomize parameter ranges
        if self.random_method == "uniform":
            self.uniform_random(num_envs)
        elif self.random_method == "sim_distri":
            self.gen(num_envs)
        elif self.random_method == "init_video":
            self.create_init_video_params(num_envs)
        elif self.random_method == "customize":
            self.params_range = self.params_range

        reset_dict = self.map_value_to_paras(self.params_range, num_envs)
        self.step_param_value = np.transpose(self.step_param_value, (1, 0))
        parms = self.params_range.copy()

        return reset_dict, parms
