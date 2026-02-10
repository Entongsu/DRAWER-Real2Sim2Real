import os
import yaml
import argparse
import gym
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument("--disable_fabric",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",
                    type=int,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--config_file",
                    type=str,
                    default=None,
                    help="Name of the task.")
parser.add_argument("--log_dir", type=str, default=None, help="log dir.")
parser.add_argument("--filename", type=str, default=None, help="test")
parser.add_argument("--num_demos",
                    type=int,
                    default=200,
                    help="number of demos")
parser.add_argument("--normalize_action",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--eval_type",
                    default="train",
                    help="the eval type for the training")
parser.add_argument("--open_loop",
                    action="store_true",
                    help="Set to True to use open loop evaluation")
parser.add_argument("--init_grasp",
                    action="store_true",
                    help="Set to True to initialize grasp")
parser.add_argument("--init_open",
                    action="store_true",
                    help="Set to True to initialize cabinet")
parser.add_argument("--init_placement",
                    action="store_true",
                    help="Set to True to initialize placement")
parser.add_argument("--init_close",
                    action="store_true",
                    help="Set to True to initialize close")

parser.add_argument("--end_name",
                    default="",
                    help="the eval type for the training")

parser.add_argument("--noise_pc",
                    action="store_true",
                    help="Set to True to initialize close")

parser.add_argument("--load_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--save_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--failure_type",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--replay",
                    action="store_true",
                    help="the eval type for the training")
parser.add_argument("--failure_config",
                    type=str,
                    default=None,
                    help="Name of the task.")
parser.add_argument("--rl_config",
                    type=str,
                    default=None,
                    help="rl config file.")
parser.add_argument("--client_url",
                    type=str,
                    default=None,
                    help="url for the inference clinet link.")
parser.add_argument("--save_replay_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--load_replay_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--load_openvla_dir",
                    default=None,
                    help="the eval type for the training")
parser.add_argument(
    "--eval_freq",
    default=10,
)


def scale_joint_values(env_config):
    scale_factor = env_config["params"]["ArticulationObject"]["kitchen"][
        "scale"][0]
    for joint, value in env_config["params"]["ArticulationObject"]["kitchen"][
            "joints"].items():
        if "prismatic" in joint:
            env_config["params"]["ArticulationObject"]["kitchen"]["joints"][
                joint] = value * scale_factor

    return env_config


def deep_merge(dict1, dict2):
    """Recursively merge two dictionaries."""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(
                dict1[key], dict):
            dict1[key] = deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def save_params_to_yaml(args_cli, save_config_path="logs/"):
    config_file = args_cli.config_file

    with open(config_file, 'r') as file:
        env_config = yaml.safe_load(file)
    if args_cli.failure_config is not None:
        with open(args_cli.failure_config, 'r') as file:
            failure_config = yaml.safe_load(file)
        env_config = deep_merge(env_config, failure_config)

    os.makedirs(save_config_path, exist_ok=True)
    save_config = f'{save_config_path}/config.yaml'

    articulate_object = env_config['params'].get('ArticulationObject', {})
    if articulate_object is not None:
        env_config = scale_joint_values(env_config)

    with open(save_config, 'w') as file:
        yaml.dump(env_config, file, default_flow_style=False)

    return env_config, save_config
