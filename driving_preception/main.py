

knobs = [
    ('ground_removal_factor', [0.1, 0.2, 0.3, 0.4, 0.5]),
    ('voxel_size', [0.1, 0.2, 0.3, 0.4, 0.50]),
    ('model', ['PolintPillars-SECFPN, PolintPillars-SECFPN-fp16', 'SSN-SECOND', 'SSN-RegNet', 'CenterPoint-DCN', 'CenterPoint-Circular-NMS'])
]

def get_pipeline_args():
    return {
       'ground_removal_factor': 0.1,
        'voxel_size': 0.1,
        'model': 'PolintPillars-SECFPN',
    }

def profile_pipeline():
    pipe_args = get_pipeline_args()


if __name__ == "__main__":
    result = profile_pipeline()

    print(result)