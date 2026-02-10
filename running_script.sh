./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [elasticity_damping] --parmas_range  [20] --num_explore_actions [0,3] --enable_cameras --headless --use_gripper
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.0\"}" --name dynamic_friction_elasticity_damping0.0
pkill -f "python"
sleep 1
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.01\"}" --name dynamic_friction_elasticity_damping0.01
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.02\"}" --name dynamic_friction_elasticity_damping0.02
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.03\"}" --name dynamic_friction_elasticity_damping0.03
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.04\"}" --name dynamic_friction_elasticity_damping0.04
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.05\"}" --name dynamic_friction_elasticity_damping0.05
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.06\"}" --name dynamic_friction_elasticity_damping0.06
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.07\"}" --name dynamic_friction_elasticity_damping0.07
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.08\"}" --name dynamic_friction_elasticity_damping0.08
pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/bimanual/random_agent.py --task Isaac-Lift-DeformCube-Franka-IK-Abs-v0  --env_config source/config/sysID/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --parmas_range  [10] --num_explore_actions [3,0] --enable_cameras --headless  --fix_params "{\"elasticity_damping\":\"0.09\"}" --name dynamic_friction_elasticity_damping0.09
pkill -f "python"
sleep 10