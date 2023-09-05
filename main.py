import os
from os.path import exists

from dotenv import load_dotenv

from extractors.pose_tracking_extractor import PoseLandmarkerExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
environment = os.path.join(BASE_DIR, '.env')

if exists(environment):
    load_dotenv(environment)

POSE_LITE_MODEL_PATH = './models/pose_landmarker_lite.task'
POSE_FULL_MODEL_PATH = './models/pose_landmarker_full.task'
POSE_HEAVY_MODEL_PATH = './models/pose_landmarker_heavy.task'
HAND_MODEL_PATH = './models/hand_landmarker.task'
FACE_MODEL_PATH = './models/face_landmarker.task'

autsl_rgb_files = os.listdir('./dataset/autsl_rgb')
autsl_depth_files = os.listdir('./dataset/autsl_depth')

# Pose extraction - RGB data
for file_path in autsl_rgb_files:
    pose_landmarker_extractor = PoseLandmarkerExtractor(
        dataset_name='autsl_rgb',
        model_path=POSE_HEAVY_MODEL_PATH,
        video_path=f'../dataset/autsl_depth/{file_path}'
    )
    pose_landmarker_extractor.extract()

# Pose extraction - Depth data
# for file_path in autsl_depth_files:
#     pose_landmarker_extractor = PoseLandmarkerExtractor(
#         dataset_name='autsl_depth',
#         model_path=POSE_HEAVY_MODEL_PATH,
#         video_path=file_path
#     )
#     pose_landmarker_extractor.extract()
