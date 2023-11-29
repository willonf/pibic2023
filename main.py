import os
from os.path import exists

import cv2
import mediapipe as mp
import numpy
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

POSE_LITE_MODEL_PATH = './models/pose_landmarker_lite.task'
POSE_FULL_MODEL_PATH = './models/pose_landmarker_full.task'
POSE_HEAVY_MODEL_PATH = './models/pose_landmarker_heavy.task'
HAND_MODEL_PATH = './models/hand_landmarker.task'
FACE_MODEL_PATH = './models/face_landmarker.task'

autsl_rgb_files = os.listdir('dataset/autsl_rgb')
autsl_depth_files = os.listdir('dataset/autsl_depth')


def show_npy_data(dataset_path):
    print(np.load(file=dataset_path, allow_pickle=True))


def load_npy_data(dataset_path):
    return np.load(file=dataset_path, allow_pickle=True)


class HandLandmarkerExtractor:

    def __init__(self, dataset_name, model_path, video_path):
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.video_path = video_path
        self.filename = self.video_path.split('/')[-1].replace('.mp4', '')
        self.pose_world_landmarks = None
        self.pose_normalized_landmarks = None
        self.normalized_npy_path = f"./output/{self.dataset_name}/{self.filename}-normalized"
        self.world_npy_path = f"./output/{self.dataset_name}/{self.filename}-world"
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2
        )

    def __extract_landmarks(self):
        current_timestamp = 0
        with HandLandmarker.create_from_options(self.options) as landmarker:
            # Capturing the video
            video_capture = cv2.VideoCapture(self.video_path)
            # video_capture = cv2.VideoCapture(0)
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            while video_capture.isOpened():
                has_frame, frame = video_capture.read()
                if not has_frame:
                    break

                cv2.imshow('Depth Frame', frame)

                # Convert the frame to a NumPy array
                numpy_frame = np.array(frame)
                print(numpy_frame)

                # Transforming into an Image MediaPipe object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame)

                hand_landmarker_result = landmarker.detect_for_video(mp_image, current_timestamp)
                self.pose_world_landmarks = hand_landmarker_result.hand_world_landmarks
                self.pose_normalized_landmarks = hand_landmarker_result.hand_landmarks
                current_timestamp += 1

    def __export_landmarks_arrays(self):

        if not exists(f"{os.environ.get('OUTPUT_PATH')}/{self.dataset_name}"):
            os.mkdir(f"{os.environ.get('OUTPUT_PATH')}/{self.dataset_name}")

        np.save(file=self.normalized_npy_path, arr=self.pose_world_landmarks)
        np.save(file=self.world_npy_path, arr=self.pose_normalized_landmarks)

    def extract(self):
        self.__extract_landmarks()
        self.__export_landmarks_arrays()


class PoseLandmarkerExtractor:

    def __init__(self, dataset_name, model_path, video_path):
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.video_path = video_path
        self.filename = self.video_path.split('/')[-1].replace('.mp4', '')
        self.pose_world_landmarks = []
        self.pose_normalized_landmarks = []
        self.normalized_npy_path = f"./output/{self.dataset_name}/{self.filename}-normalized"
        self.world_npy_path = f"./output/{self.dataset_name}/{self.filename}-world"
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO
        )

    def __extract_landmarks(self):
        current_timestamp = 0
        with PoseLandmarker.create_from_options(self.options) as landmarker:
            # Capturing the video
            video_capture = cv2.VideoCapture(self.video_path)

            while video_capture.isOpened():
                has_frame, frame = video_capture.read()
                if not has_frame:
                    break

                cv2.imshow('Depth Frame', frame)

                # Convert the frame to a NumPy array
                numpy_frame = np.array(frame)

                # Transforming into an Image MediaPipe object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame)

                pose_landmarker_result = landmarker.detect_for_video(mp_image, current_timestamp)
                self.pose_world_landmarks += pose_landmarker_result.pose_world_landmarks
                self.pose_normalized_landmarks += pose_landmarker_result.pose_landmarks
                current_timestamp += 1

    def __export_normalized_landmarks_arrays(self):
        if not exists(f"./output/{self.dataset_name}"):
            os.mkdir(f"./output/{self.dataset_name}")
        final_dataframe_normalized = []
        for frame in self.pose_normalized_landmarks:
            time_df = []
            for coord in frame:
                x = coord.x
                y = coord.y
                z = coord.z
                time_df.append(x)
                time_df.append(y)
                time_df.append(z)
            final_dataframe_normalized.append(time_df)
        np.save(file=self.normalized_npy_path, arr=final_dataframe_normalized)

    def __export_world_landmarks_arrays(self):
        final_dataframe_world = []
        for frame in self.pose_world_landmarks:
            time_df = []
            for coord in frame:
                x = coord.x
                y = coord.y
                z = coord.z
                time_df.append(x)
                time_df.append(y)
                time_df.append(z)
            final_dataframe_world.append(time_df)

        np.save(file=self.world_npy_path, arr=final_dataframe_world)

    def extract(self):
        self.__extract_landmarks()
        self.__export_normalized_landmarks_arrays()
        self.__export_world_landmarks_arrays()


# Pose extraction - RGB data
for file_path in autsl_rgb_files:
    pose_landmarker_extractor = PoseLandmarkerExtractor(
        dataset_name='autsl_rgb',
        model_path=POSE_HEAVY_MODEL_PATH,
        video_path=f'dataset/autsl_depth/{file_path}'
    )
    pose_landmarker_extractor.extract()

# Pose extraction - Depth data
for file_path in autsl_depth_files:
    pose_landmarker_extractor = PoseLandmarkerExtractor(
        dataset_name='autsl_depth',
        model_path=POSE_HEAVY_MODEL_PATH,
        video_path=file_path
    )
    pose_landmarker_extractor.extract()
