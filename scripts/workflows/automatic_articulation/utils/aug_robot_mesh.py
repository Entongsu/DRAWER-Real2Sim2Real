from tools.curobo_planner import IKPlanner
import torch
import h5py
normalized_grasp = h5py.File(f"logs/1105_grasp/grasp_normalized2.hdf5",'r+')

device = "cuda:0"


curobo_ik = IKPlanner( env=None,)
for i in range(len(normalized_grasp["data"])):
    jpos = normalized_grasp["data"][f"demo_{i}"]["obs"]["joint_pos"]
    for k in range(jpos.shape[0]):
        fk_result = curobo_ik.ik_solver.fk(torch.as_tensor(jpos[k]).unsqueeze(0)[...,:7].to(device))
    print('done')
   