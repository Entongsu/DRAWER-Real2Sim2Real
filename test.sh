
# # # # # # # # # # #collect data
# #./isaaclab.sh -p source/standalone/workflows/vlm_failure/stack_block/random_agent.py --task Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0  --num_envs=1 --enable_cameras --config_file=source/config/task/vlm_failure/stackblock01.yaml --log_dir=logs/stack3blocks --num_demos=50 --save_path=raw_data
# # ./isaaclab.sh -p scripts/workflows/utils/convert_npz_to_h5py.py --task=Isaac-Open-Drawer-Franka-IK-Abs-v0  --num_envs=1 --config_file=source/config/task/automatic_articulation/kitchen02_yunchu.yaml --log_dir=logs/0114_cabinet --num_demos=2000 --load_path=raw_data --save_path=raw_data
# #replay for joint pos
# #./isaaclab.sh -p ource/standalone/workflows/automatic_articulation/replay_multi_step.py --task=Isaac-Open-Drawer-Franka-IK-Abs-v0 --num_envs=1 --config_file=source/config/task/automatic_articulation/kitchen02_yunchu.yaml --log_dir=logs/kitchen02_yunchu --enable_cameras --num_demos=2000  --init_open --load_path=raw_data --save_path=render_data
# # # # # #save into the h5py format for joint pos controller
# #./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py --task=Isaac-Open-Drawer-Franka-IK-Abs-v0  --num_envs=1 --config_file=source/config/task/automatic_articulation/kitchen02_prismatic.yaml --log_dir=logs/1213_cabinet --num_demos=2000 --load_path=render_data --save_path=render_data
# # # # # augment the data
# # python source/standalone/workflows/automatic_articulation/utils/aug_mesh.py  --log_dir=logs/failure/1125_placement --source_path cabinet_normalized_noise
# # #save into the zarr format
# # python source/standalone/workflows/automatic_articulation/utils/convert_zarr.py --source_path cabinet_normalized_noise_aug --log_dir=logs/failure/1125_placement
# # # # # # # # # # train the model
# # cd ..
# # # cd robomimic/
# # #python robomimic/scripts/train_mt.py --config=robomimic/exps/templates/co_bc.json --dataset=../IsaacLab/logs/failure/1125_placement/cabinet_normalized_noise_aug.hdf5
# # cd 3D-Diffusion-Policy
# # bash scripts/train_policy.sh dp3  kitchen_grasp 1125_placement 0 0 logs/failure/1125_placement/cabinet_normalized_noise_aug.zarr/


# #./isaaclab.sh -p source/standalone/workflows/vlm_failure/stack_block/failure_agent.py --task Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0  --num_envs=1 --enable_cameras --config_file=source/config/task/vlm_failure/stackblock01.yaml --log_dir=logs/stack3blocks --num_demos=30 --load_path=raw_data --save_path=/failure/y_offset/raw_failure_data_y_offset --failure_type=y_offset --failure_config=source/config/task/failure_env/stackblock01.yaml
# #./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py --task=Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0 --num_envs=1 --config_file=source/config/task/vlm_failure/stackblock01.yaml --log_dir=logs/stack3blocks --num_demos=2000 --load_path=/failure/y_offset/raw_failure_data_y_offset --save_path=/failure/y_offset/raw_failure_data_y_offset
# # ./isaaclab.sh -p source/standalone/workflows/vlm_failure/stack_block/failure_agent.py --task Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0  --num_envs=1 --enable_cameras --config_file=source/config/task/vlm_failure/stackblock01.yaml --log_dir=logs/stack3blocks --num_demos=30 --load_path=/failure/y_offset/raw_failure_data_y_offset --save_path=/failure/y_offset/failure_data_y_offset --failure_type=y_offset --failure_config=source/config/task/failure_env/stackblock01.yaml --replay
# # ./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py --task=Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0 --num_envs=1 --config_file=source/config/task/vlm_failure/stackblock01.yaml --log_dir=logs/stack3blocks --num_demos=2000 --load_path=/failure/y_offset/failure_data_y_offset --save_path=/failure/y_offset/failure_data_y_offset
# #./isaaclab.sh -p source/standalone/workflows/vlm_failure/stack_block/llm_agent.py --task Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Abs-v0  --num_envs=1 --enable_cameras --config_file=source/config/task/vlm_failure/stackblock01.yaml --log_dir=logs/stack3blocks --num_demos=30  --load_path=/failure/y_offset/raw_failure_data_y_offset --failure_type=y_offset --failure_config=source/config/task/failure_env/stackblock01.yaml


# # kithecn multi-step setup
# ./isaaclab.sh -p source/standalone/workflows/automatic_articulation/random_multi_step.py --task=Isaac-Open-Drawer-Franka-IK-Abs-v0  --num_envs=1 --config_file=source/config/task/automatic_articulation/kitchen02_yunchu.yaml --log_dir=logs/0122_cabinet --enable_cameras --num_demos=60 --init_open --save_path=raw_data
# #./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py   --task=Isaac-Open-Drawer-Franka-IK-Abs-v0 --log_dir=logs/1217_oven01 --num_demos=2000 --load_path=raw_data --save_path=raw_data
# #./isaaclab.sh -p source/standalone/workflows/automatic_articulation/replay_multi_step.py --task=Isaac-Open-Drawer-Franka-IK-Abs-v0 --num_envs=1 --config_file=source/config/task/automatic_articulation/oven01.yaml --log_dir=logs/1217_oven01 --enable_cameras --num_demos=2000  --init_open --load_path=raw_data --save_path=render_data
# #./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py   --task=Isaac-Open-Drawer-Franka-IK-Abs-v0 --log_dir=logs/1217_oven01 --num_demos=2000 --load_path=render_data --save_path=render_data

# #./isaaclab.sh -p source/standalone/workflows/automatic_articulation/failure_multi_step.py --task Isaac-Open-Drawer-Franka-IK-Abs-v0  --num_envs=1 --enable_cameras --config_file=source/config/task/automatic_articulation/oven01.yaml --log_dir=logs/1217_oven01 --num_demos=30 --load_path=raw_data --save_path=/failure/y_offset/raw_failure_data_y_offset --failure_type=y_offset  --failure_config=source/config/task/failure_env/oven01_failure.yaml --init_open
# # ./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py   --task=Isaac-Open-Drawer-Franka-IK-Abs-v0 --log_dir=logs/1217_oven01 --num_demos=2000 --load_path=failure/y_offset/raw_failure_data_y_offset --save_path=failure/y_offset/raw_failure_data_y_offset
# # ./isaaclab.sh -p source/standalone/workflows/automatic_articulation/failure_multi_step.py --task Isaac-Open-Drawer-Franka-IK-Abs-v0  --num_envs=1 --enable_cameras --config_file=source/config/task/automatic_articulation/oven01.yaml --log_dir=logs/1217_oven01 --num_demos=30 --load_path=failure/y_offset/raw_failure_data_y_offset --save_path=/failure/y_offset/failure_data_y_offset --failure_type=y_offset  --failure_config=source/config/task/failure_env/oven01_failure.yaml --init_open --replay
# # ./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py   --task=Isaac-Open-Drawer-Franka-IK-Abs-v0 --log_dir=logs/1217_oven01 --num_demos=2000 --load_path=failure/y_offset/failure_data_y_offset --save_path=failure/y_offset/failure_data_y_offset

# ./isaaclab.sh -p source/standalone/workflows/utils/convert_npz_to_h5py.py --task=Isaac-Reach-Franka-IK-Rel-v0  --config_file=source/config/task/vlm_failure/reach_robot.yaml --log_dir=logs/reach --num_demos=2000 --load_path=raw_data --save_path=raw_data
#./isaaclab.sh -p scripts/workflows/vlm_failure/reach_robot/task/converter.py --task Isaac-Reach-Franka-IK-Rel-v0 --config_file source/config/task/vlm_failure/reach_robot.yaml --log_dir /media/aurmr/data1/isaac_data/logs/reach --num_env=1 --save_path raw_clean_data --enable_camera --num_demos 10000 --load_path raw_data
./isaaclab.sh -p scripts/workflows/vlm_failure/reach_robot/replay_agent.py --task Isaac-Reach-Franka-IK-Rel-v0 --config_file source/config/task/vlm_failure/reach_robot.yaml --log_dir /media/aurmr/data1/isaac_data/logs/reach --num_env=1 --load_replay_path raw_clean_data --save_replay_path test_data  --enable_camera --num_demos 10