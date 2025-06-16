CUSTOM_PERSON_SKELETON = [[1, 2], [1, 5], [2, 5], [2, 3], [2, 8], [3, 4], [5, 6], [5, 11], [6, 7], [8, 11], [8, 9], [9, 10], [11, 12], [12, 13]]
    
CUSTOM_KEYPOINTS = [
    'head',           # 1
    'left_shoulder',  # 2
    'left_elbow',     # 3
    'left_wrist',     # 4
    'right_shoulder', # 5
    'right_elbow',    # 6
    'right_wrist',    # 7
    'left_hip',       # 8
    'left_knee',      # 9
    'left_ankle',     # 10
    'right_hip',      # 11
    'right_knee',     # 12
    'right_ankle',    # 13
]


HFLIP = {
    'left_shoulder': 'right_shoulder',
    'left_elbow': 'right_elbow',
    'left_wrist': 'right_wrist',
    'right_shoulder': 'left_shoulder',
    'right_elbow': 'left_elbow',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'left_knee': 'right_knee',
    'left_ankle': 'right_ankle',
    'right_hip': 'left_hip',
    'right_knee': 'left_knee',
    'right_ankle': 'left_ankle',
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
