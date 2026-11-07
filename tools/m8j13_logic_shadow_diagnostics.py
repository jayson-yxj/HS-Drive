"""Native path transport and one official PID diagnostic, no simulator API."""
import copy
import numpy as np
from datasets.m8j12_contract import native_target
from simulator.m8j13_safety import transport_native_path,sanitize_control


def transported_pid(prediction,row,frozen,controller):
    path=prediction['lat_reg_final']
    original=path.detach().cpu().numpy().copy() if hasattr(path,'detach') else np.asarray(path).copy()
    transported=transport_native_path(original,row['ego_world_matrix'],frozen['pose'])
    matrix=np.linalg.inv(np.asarray(frozen['pose']))@np.asarray(row['ego_world_matrix'])
    target=native_target(frozen['pose'],row['target_world_xy'])
    changed=dict(prediction)
    if hasattr(path,'new_tensor'):changed['lat_reg_final']=path.new_tensor(transported)
    else:changed['lat_reg_final']=transported.copy()
    def invoke():
        result=controller.command(changed,frozen['speed'],target)
        current=path.detach().cpu().numpy() if hasattr(path,'detach') else np.asarray(path)
        if not np.array_equal(original,current):raise ValueError('NativePathMutated')
        _,raw_error=sanitize_control(result['raw_pid']);_,agent_error=sanitize_control(result['native_agent_control'])
        return result,raw_error,agent_error
    return invoke,dict(original_path=original.tolist(),transport_matrix=matrix.tolist(),transported_path=transported.tolist(),current_speed_mps=frozen['speed'],current_navigation_target=target.tolist(),world_navigation_target=copy.deepcopy(row['target_world_xy']))
