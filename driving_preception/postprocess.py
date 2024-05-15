"""
Combine stage 1, ground removal and voxelization, and stage 2, detector profiling result.
"""
import json
import os
import copy 

stage_1_result = []
stage_1_profile_file_path = 'legacy/profile_result.json'
with open (stage_1_profile_file_path, 'r') as fp:
    stage_1_result = json.load(fp)         


profile_result = []

models = ['centerpoint_dcn', 'centerpoint_circlenms', 'pointpillars_secfpn', 'ssn_second']
for s1_result in stage_1_result:

    g, v = int(s1_result['ground_factor']*100), int(s1_result['voxel_factor']*100)
    for model in models:
        path = f'work_dirs/{model}/g{g}-v{v}' 
        dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        for dir_name in dir_names:
            stage_2_profile_file_path = os.path.join(path, dir_name, f'{dir_name}.json')

            result = copy.deepcopy(s1_result) 
            with open(stage_2_profile_file_path, 'r') as fp:
                stage_2_result = json.load(fp) 
                result['model'] = model
                result['result']['accuracy'] = stage_2_result['NuScenes metric/pred_instances_3d_NuScenes/mAP']
                result['result']['detector_compute_latency'] = stage_2_result['time']

                profile_result.append(result)

with open('profile_result_all.json', 'w') as fp:
    json.dump(profile_result, fp)
    