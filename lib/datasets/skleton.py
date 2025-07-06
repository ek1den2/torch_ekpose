CUSTOM_PERSON_SKELETON = [[1, 2], [1, 5], [2, 5], [2, 3], [2, 8], [3, 4], [5, 6], [5, 11], [6, 7], [8, 11], [8, 9], [9, 10], [11, 12], [12, 13]]
    
CUSTOM_KEYPOINTS = [
    'Head',           # 1
    'LShoulder',      # 2
    'LElbow',         # 3
    'LWrist',        # 4
    'RShoulder',      # 5
    'RElbow',         # 6
    'RWrist',        # 7
    'LHip',           # 8
    'LKnee',          # 9
    'LAnkle',        # 10
    'RHip',           # 11
    'RKnee',          # 12
    'RAnkle',        # 13
]


HFLIP = {
    'LShoulder': 'RShoulder',
    'LElbow': 'RElbow',
    'LWrist': 'RWrist',
    'RShoulder': 'LShoulder',
    'RElbow': 'LElbow',
    'RWrist': 'LWrist',
    'LHip': 'RHip',
    'LKnee': 'RKnee',
    'LAnkle': 'RAnkle',
    'RHip': 'LHip',
    'RKnee': 'LKnee',
    'RAnkle': 'LAnkle',
}

CUSTOM_PERSON_SIGMAS = [
    0.062, # head
    0.079, # left_shoulder
    0.072, # left_elbow
    0.062, # left_wrist
    0.079, # right_shoulder
    0.072, # right_elbow
    0.062, # right_wrist
    0.107, # left_hip
    0.087, # left_knee
    0.089, # left_ankle
    0.107, # right_hip
    0.087, # right_knee
    0.089, # right_ankle
]

def print_associations():
    for j1, j2 in CUSTOM_PERSON_SKELETON:
        print(CUSTOM_KEYPOINTS[j1 - 1], '-', CUSTOM_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()
