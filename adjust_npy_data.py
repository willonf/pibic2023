import os

import helpers

autsl_rgb_datasets = os.listdir('./output/autsl_rgb')
autsl_depth_datasets = os.listdir('./output/autsl_depth')

for dataset_path in autsl_rgb_datasets:
    keypoints = helpers.load_npy_data(dataset_path)
