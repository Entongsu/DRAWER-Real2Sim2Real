# Drawer Simulation: Digital Reconstruction and Articulation With Environment Realism

This project provides a simulation and data generation pipeline for drawer articulation using Isaac Sim 4.2 and Isaac Lab, with support for large-scale demonstration collection and visual replay.


⚠️ Requirement
This code is developed and tested on Isaac Sim 4.2.
Please install Isaac Sim 4.2 before running the commands below.

---
### 🏠 Kitchen Asset Download

Download the reconstructed kitchen assets used in this project from the link below and place them in the appropriate asset directory specified in the task configuration:

🔗 Kitchen Assets (Google Drive)
https://drive.google.com/drive/folders/17bVju4wAgy6MGNono_AfAqQ7pl0-VQC_?usp=drive_link

---

### 📊 Data Collection
```
./isaaclab.sh -p scripts/workflows/automatic_articulation/random_multi_step.py --task Isaac-Open-Drawer-Franka-IK-Abs-v0 --num_envs=1 --enable_cameras --config_file source/config/task/automatic_articulation/kitchen01.yaml --log_dir logs/kitchen01 --num_demos=50 --save_path raw_data --init_open
```

### 🔄 Data Conversion (NPZ → HDF5)
```
./isaaclab.sh -p scripts/workflows/utils/convert_npz_to_h5py.py --task Isaac-Open-Drawer-Franka-IK-Abs-v0 --num_envs=1 --config_file source/config/task/automatic_articulation/kitchen01.yaml --log_dir logs/kitchen01 --num_demos=2000 --load_path raw_data --save_path raw_data
```
### 🎥 Replay & Rendering
```
./isaaclab.sh -p scripts/workflows/automatic_articulation/replay_multi_step.py --task Isaac-Open-Drawer-Franka-IK-Abs-v0 --num_envs=1 --config_file source/config/task/automatic_articulation/kitchen01.yaml --log_dir logs/kitchen02_yunchu --enable_cameras --num_demos=2000 --init_open --load_path raw_data --save_path render_data
```
---
### 📝 Notes
```
Use --num_envs=1 for stable articulation and debugging.

Camera settings and articulation parameters are defined in the YAML config.

The replay step is recommended to verify articulation quality and data correctness before training.