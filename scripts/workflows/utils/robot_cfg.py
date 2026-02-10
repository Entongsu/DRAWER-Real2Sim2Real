import numpy as np
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

DATA_PATH = Path(__file__).parent.parent.parent / "data"

WIDOWX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=
        f"source/assets/kitchen/usd/bridge_kitchen_uw/robot/wx250s_processed.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={
        "waist": -0.26,
        "shoulder": -0.12,
        "elbow": 0,
        "forearm_roll": -0.006,
        "wrist_angle": 1.70,
        "wrist_rotate": -0.269,
        "gripper": 0.0,
        "left_finger": 0.036,
        "right_finger": -0.036,
    },
                                               pos=[0.0, 0.0, 0.0]),
    actuators={
        "waist":
        ImplicitActuatorCfg(
            joint_names_expr=["waist"],
            velocity_limit=1.5,
            effort_limit=10.0,
            stiffness=1169,
            damping=330.0,
        ),
        "shoulder":
        ImplicitActuatorCfg(joint_names_expr=["shoulder"],
                            velocity_limit=1.5,
                            effort_limit=10.0,
                            stiffness=730,
                            damping=180),
        "elbow":
        ImplicitActuatorCfg(joint_names_expr=["elbow"],
                            velocity_limit=1.5,
                            effort_limit=10.0,
                            stiffness=808,
                            damping=152),
        "forearm_roll":
        ImplicitActuatorCfg(joint_names_expr=["forearm_roll"],
                            velocity_limit=1.5,
                            effort_limit=10.0,
                            stiffness=1229,
                            damping=309),
        "wrist_angle":
        ImplicitActuatorCfg(joint_names_expr=["wrist_angle"],
                            velocity_limit=1.5,
                            effort_limit=10.0,
                            stiffness=1272,
                            damping=201),
        "wrist_rotate":
        ImplicitActuatorCfg(joint_names_expr=["wrist_rotate"],
                            velocity_limit=1.5,
                            effort_limit=10.0,
                            stiffness=1056,
                            damping=269),
        "gripper":
        ImplicitActuatorCfg(joint_names_expr=["gripper"],
                            velocity_limit=1.5,
                            effort_limit=10.0,
                            stiffness=1056,
                            damping=269),
        "left_finger":
        ImplicitActuatorCfg(
            joint_names_expr=["left_finger"],
            velocity_limit=5.0,
            effort_limit=10.0,
            stiffness=2e3,
            damping=100,
        ),
        "right_finger":
        ImplicitActuatorCfg(
            joint_names_expr=["right_finger"],
            velocity_limit=5.0,
            effort_limit=10.0,
            stiffness=2e3,
            damping=100,
        ),
    })

DROID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/assets/robot/droid_uw_robot/robot.usd",
        # usd_path=DATA_PATH / "droid/mesh_joint_fric.usdz",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={
        "panda_joint1": 0.0,
        "panda_joint2": -6.2832e-01,
        "panda_joint3": 0.0,
        "panda_joint4": -2.5133e+00,
        "panda_joint5": 0.0,
        "panda_joint6": 1.8850e+00,
        "panda_joint7": 0,
        "finger_joint": 0,
    }, ),
    actuators={
        "panda_shoulder":
        ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm":
        ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            # effort_limit=12.0,
            effort_limit=300.0,
            velocity_limit=2.61,
            stiffness=800.0,
            damping=80.0,
        ),
        "gripper":
        ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint", "right_outer_finger_joint",
                "left_outer_finger_joint", "right_outer_knuckle_joint"
            ],
            # effort_limit=300,
            # velocity_limit=10.0,
            # stiffness=10.0,
            # damping=0.002,
            effort_limit=100,
            velocity_limit=1.0,
            stiffness=1.0,
            damping=0.002,
        ),
    },
)


def set_droid_config(object):
    # Set Actions for the specific robot type (franka)
    setattr(object, "actions", ActionCfg())
    self.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True)

    self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[
            'finger_joint',
            # 'right_outer_knuckle_joint',
            # 'left_inner_finger_joint', 'right_inner_finger_joint',
            # 'left_inner_finger_knuckle_joint',
            # 'right_inner_finger_knuckle_joint'
        ],
    )
