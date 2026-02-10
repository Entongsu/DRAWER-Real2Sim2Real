import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math
import isaaclab.envs.mdp as mdp

from pathlib import Path
from dataclasses import dataclass, field
from isaaclab.utils import configclass
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.utils .math import subtract_frame_transforms

from isaaclab.sim.spawners import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class DroidnUWSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),
                                     intensity=1000.0),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.0]),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0],
                                                rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=
            f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


@configclass
class ActionCfg:
    arm_action: mdp.JointPositionActionCfg = None
    gripper_action: mdp.BinaryJointPositionActionCfg = None


def get_jp(
        env,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    gripper_state = robot.data.joint_pos
    return gripper_state


def get_camera_data(env: ManagerBasedEnv,
                    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
                    type: str = "rgb") -> torch.Tensor:

    camera = env.scene[camera_cfg.name]
    return camera.data.output[type][0][..., :3]


@configclass
class ObservationCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        # control_joint_action = ObsTerm(func=mdp.control_joint_action)
        # actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(func=mdp.joint_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot = EventTerm(func=reset_ee_pose, mode="reset")


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class DroidnUWCfg(ManagerBasedEnvCfg):
    env_config: str = field(default=None)

    scene = DroidnUWSceneCfg(num_envs=1, env_spacing=2.0)

    actions = ActionCfg()
    rewards = RewardsCfg()
    observations = ObservationCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 10

        self.viewer.eye = (4.5, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 3
        self.sim.dt = 1 / 60
        self.sim.render_interval = 12

        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30

        self.instruction = "put the eggplant in the sink"
