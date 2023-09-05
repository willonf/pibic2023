import os
from os.path import exists

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode, HandLandmarker, HandLandmarkerOptions

import helpers


class HandLandmarkerExtractor:

    def __init__(self, dataset_name, model_path, video_path):
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.video_path = video_path
        self.filename = self.video_path.split('/')[-1].replace('.mp4', '')
        self.pose_world_landmarks = None
        self.pose_normalized_landmarks = None
        self.normalized_npy_path = f"{os.environ.get('OUTPUT_PATH')}/{self.dataset_name}/{self.filename}-normalized"
        self.world_npy_path = f"{os.environ.get('OUTPUT_PATH')}/{self.dataset_name}/{self.filename}-world"
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

                # Waiting 'q' key be pressed to cancel
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

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

    def __show_normalized_data(self):
        print(f"{10 * '+='} NORMALIZED DATA {10 * '+='}")
        helpers.show_npy_data(f'{self.normalized_npy_path}.npy')

    def __show_world_data(self):
        print(f"{10 * '+='} WORLD DATA {10 * '+='}")
        helpers.show_npy_data(f'{self.world_npy_path}.npy')

    def extract(self):
        self.__extract_landmarks()
        self.__export_landmarks_arrays()
        self.__show_normalized_data()
        self.__show_world_data()
