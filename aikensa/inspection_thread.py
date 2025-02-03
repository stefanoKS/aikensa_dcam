import inspect
from tabnanny import verbose
import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time
import logging
import sqlite3

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.camscripts.cam_hole_init import initialize_hole_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound, play_ok_sound, play_ng_sound

from ultralytics import YOLO
from aikensa.parts_config.P828XXW0X0P_CTRPLR import partcheck as P828XXW0X0P_check               #5

# from aikensa.parts_config.P5902A509 import dailyTenken01 as P5902A509_dailyTenken01
# from aikensa.parts_config.P5902A509 import dailyTenken02 as P5902A509_dailyTenken02
# from aikensa.parts_config.P5902A509 import dailyTenken03 as P5902A509_dailyTenken03

from PIL import ImageFont, ImageDraw, Image

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

    mapCalculated: list = field(default_factory=lambda: [False]*30) #for 10 cameras
    map1: list = field(default_factory=lambda: [None]*30) #for 10 cameras
    map2: list = field(default_factory=lambda: [None]*30) #for 10 cameras

    map1_downscaled: list = field(default_factory=lambda: [None]*30) #for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*30) #for 10 cameras

    doInspection: bool = False
    button_sensor: int = 0

    kensainNumber: str = None
    furyou_plus: bool = False
    furyou_minus: bool = False
    kansei_plus: bool = False
    kansei_minus: bool = False
    furyou_plus_10: bool = False #to add 10
    furyou_minus_10: bool = False
    kansei_plus_10: bool = False
    kansei_minus_10: bool = False

    counterReset: bool = False

    today_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])
    current_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])

class InspectionThread(QThread):

    partCam = pyqtSignal(QImage)
    partKatabuL = pyqtSignal(QImage)
    partKatabuR = pyqtSignal(QImage)

    clip1Signal = pyqtSignal(QImage)
    clip2Signal = pyqtSignal(QImage)
    clip3Signal = pyqtSignal(QImage)

    P82833W050P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W040P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82833W090P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W080P_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    
    P82833W050PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W040PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82833W090PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W080PKENGEN_InspectionResult_PitchMeasured = pyqtSignal(list, list)

    P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured = pyqtSignal(list, list)

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)

    ethernetStatus = pyqtSignal(list)

    def __init__(self, inspection_config: InspectionConfig = None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.multiCam_stream = False

        self.cap_cam = None
        self.cap_cam1 = None
        self.cap_cam2 = None

        self.emit = None

        self.mergeframe1 = None
        self.mergeframe2 = None

        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None

        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        # self.homography_matrix1_high = None
        # self.homography_matrix2_high = None

        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None
        # self.homography_matrix1_high_scaled= None
        # self.homography_matrix2_high_scaled = None

        self.H1 = None
        self.H2 = None
        # self.H1_high = None
        # self.H2_high = None

        self.H1_scaled = None
        self.H2_scaled = None
        # self.H1_high_scaled = None
        # self.H2_high_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.katabuImageL = None
        self.katabuImageR = None
        self.katabuImageL_scaled = None
        self.katabuImageR_scaled = None
        
        self.katabuImage = None
        self.katabuImage_init = None

        #Crop format: X Y W H OUTW OUTH
        self.katabuImageL_Crop = np.array([620, 360, 320, 160, 320, 160])
        self.katabuImageR_Crop = np.array([3800, 360, 320, 160, 320, 160])

        self.clipImage1 = None
        self.clipImage2 = None
        self.clipImage3 = None

        self.clipImage1_Crop = np.array([1750, 1600, 600, 600, 128, 128])
        self.clipImage2_Crop = np.array([600, 1600, 600, 600, 128, 128])
        self.clipImage3_Crop = np.array([1880, 1600, 600, 600, 128, 128])

        # self.combinedImage_narrow = None
        # self.combinedImage_narrow_scaled = None
        # self.combinedImage_wide = None
        # self.combinedImage_wide_scaled = None

        # self.combinedImage_high_narrow = None
        # self.combinedImage_high_narrow_scaled = None
        # self.combinedImage_high_wide = None
        # self.combinedImage_high_wide_scaled = None

        self.scale_factor = 5.0 #Scale Factor, might increase this later
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.narrow_planarize = (531, 2646)
        self.wide_planarize = (1342, 5672)

        self.planarizeTransform_narrow = None
        self.planarizeTransform_narrow_scaled = None
        self.planarizeTransform_high_narrow = None
        self.planarizeTransform_high_narrow_scaled = None

        self.planarizeTransform_wide = None
        self.planarizeTransform_wide_scaled = None
        self.planarizeTransform_high_wide = None
        self.planarizeTransform_high_wide_scaled = None

        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*1
        self.InspectionImages_bgr = [None]*1

        self.InspectionImagesKatabu = [None]*1

        self.InspectionImages_endSegmentation_Left = [None]*1
        self.InspectionImages_endSegmentation_Right = [None]*1

        self.InspectionResult_EndSegmentation_Left = [None]*5
        self.InspectionResult_EndSegmentation_Right = [None]*5

        self.InspectionResult_ClipDetection = [None]*30
        self.InspectionResult_KatabuDetection = [None]*30
        self.InspectionResult_Segmentation = [None]*30
        self.InspectionResult_Hanire = [None]*30

        self.InspectionResult_PitchMeasured = [None]*30
        self.InspectionResult_PitchResult = [None]*30
        self.InspectionResult_DetectionID = [None]*30
        self.InspectionResult_Status = [None]*30
        self.InspectionResult_DeltaPitch = [None]*30

        self.InspectionImages_prev = [None]*30
        self._test = [0]*30
        self.widget_dir_map = {
            5: "82833W050P",
            6: "82832W040P",
            7: "82833W090P",
            8: "82832W080P",
            9: "82833W050PKENGEN",
            10: "82832W040PKENGEN",
            11: "82833W090PKENGEN",
            12: "82832W080PKENGEN",
            13: "82833W050PCLIPSOUNYUUKI",
            14: "82832W040PCLIPSOUNYUUKI",
            15: "82833W090PCLIPSOUNYUUKI",
            16: "82832W080PCLIPSOUNYUUKI",
        }

        #for widget name map, append the string "P" to the initial widget dir map
        self.widget_name_map = {key: f"P{value}" for key, value in self.widget_dir_map.items()}

        self.InspectionWaitTime = 1.0
        self.InspectionTimeStart = None

        self.ethernetTrigger = [0]*5

        
        self.cam_config_file = "aikensa/camscripts/cam_config.yaml"
        
        with open(self.cam_config_file, 'r') as file:
            self.cam_map = yaml.safe_load(file)


    def release_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")

    def initialize_single_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.inspection_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            # self.frame = self.create_placeholder_image()
            self.cap_cam = initialize_camera(camID)
            if not self.cap_cam.isOpened():
                print(f"Failed to open camera with ID {camID}")
                self.cap_cam = None
            else:
                print(f"Initialized Camera on ID {camID}")

    def initialize_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")

        actual_camID = self.cam_map.get(0, -1)
        self.cap_cam1 = initialize_camera(actual_camID)

        actual_camID = self.cam_map.get(1, -1)
        self.cap_cam2 = initialize_camera(actual_camID)

        if not self.cap_cam1.isOpened():
            print(f"Failed to open camera with ID 1")
            self.cap_cam1 = None
        else:
            print(f"Initialized Camera on ID 1")

        if not self.cap_cam2.isOpened():
            print(f"Failed to open camera with ID 2")
            self.cap_cam2 = None
        else:
            print(f"Initialized Camera on ID 2")


    def run(self):
        #initialize the database
        if not os.path.exists("./aikensa/inspection_results"):
            os.makedirs("./aikensa/inspection_results")

        self.conn = sqlite3.connect('./aikensa/inspection_results/database_results.db')
        self.cursor = self.conn.cursor()

        # Create the table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partName TEXT,
            numofPart TEXT,
            currentnumofPart TEXT,
            timestampHour TEXT,
            timestampDate TEXT,
            deltaTime REAL,
            kensainName TEXT,
            detected_pitch TEXT,
            delta_pitch TEXT,
            total_length REAL
        )
        ''')

        self.conn.commit()

        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Models Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        # self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        for key, value in self.widget_dir_map.items():
            self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)
                print(f"Loaded homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)
                print(f"Loaded homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)
                print(f"Loaded scaled homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)
                print(f"Loaded scaled homography matrix for camera 2")
 

        # if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_high.yaml"):
        #     with open("./aikensa/cameracalibration/homography_param_cam1_high.yaml") as file:
        #         self.homography_matrix1_high = yaml.load(file, Loader=yaml.FullLoader)
        #         self.H1_high = np.array(self.homography_matrix1_high)

        # if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_high.yaml"):
        #     with open("./aikensa/cameracalibration/homography_param_cam2_high.yaml") as file:
        #         self.homography_matrix2_high = yaml.load(file, Loader=yaml.FullLoader)
        #         self.H2_high = np.array(self.homography_matrix2_high)

        # if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_high_scaled.yaml"):
        #     with open("./aikensa/cameracalibration/homography_param_cam1_high_scaled.yaml") as file:
        #         self.homography_matrix1_high_scaled = yaml.load(file, Loader=yaml.FullLoader)
        #         self.H1_high_scaled = np.array(self.homography_matrix1_high_scaled)

        # if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_high_scaled.yaml"):
        #     with open("./aikensa/cameracalibration/homography_param_cam2_high_scaled.yaml") as file:
        #         self.homography_matrix2_high_scaled = yaml.load(file, Loader=yaml.FullLoader)
        #         self.H2_high_scaled = np.array(self.homography_matrix2_high_scaled)


        # if os.path.exists("./aikensa/cameracalibration/planarizeTransform_narrow.yaml"):
        #     with open("./aikensa/cameracalibration/planarizeTransform_narrow.yaml") as file:
        #         transform_list = yaml.load(file, Loader=yaml.FullLoader)
        #         self.planarizeTransform_narrow = np.array(transform_list)

        # if os.path.exists("./aikensa/cameracalibration/planarizeTransform_narrow_scaled.yaml"):
        #     with open("./aikensa/cameracalibration/planarizeTransform_narrow_scaled.yaml") as file:
        #         transform_list = yaml.load(file, Loader=yaml.FullLoader)
        #         self.planarizeTransform_narrow_scaled = np.array(transform_list)
        
        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_wide.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_wide.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_wide_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_wide_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide_scaled = np.array(transform_list)

        # if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_narrow.yaml"):
        #     with open("./aikensa/cameracalibration/planarizeTransform_high_narrow.yaml") as file:
        #         transform_list = yaml.load(file, Loader=yaml.FullLoader)
        #         self.planarizeTransform_high_narrow = np.array(transform_list)
        
        # if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_narrow_scaled.yaml"):
        #     with open("./aikensa/cameracalibration/planarizeTransform_high_narrow_scaled.yaml") as file:
        #         transform_list = yaml.load(file, Loader=yaml.FullLoader)
        #         self.planarizeTransform_high_narrow_scaled = np.array(transform_list)

        # if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_wide.yaml"):
        #     with open("./aikensa/cameracalibration/planarizeTransform_high_wide.yaml") as file:
        #         transform_list = yaml.load(file, Loader=yaml.FullLoader)
        #         self.planarizeTransform_high_wide = np.array(transform_list)

        # if os.path.exists("./aikensa/cameracalibration/planarizeTransform_high_wide_scaled.yaml"):
        #     with open("./aikensa/cameracalibration/planarizeTransform_high_wide_scaled.yaml") as file:
        #         transform_list = yaml.load(file, Loader=yaml.FullLoader)
        #         self.planarizeTransform_high_wide_scaled = np.array(transform_list)


        while self.running:

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget > 0:

                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()
                    # print("initialize all camera")    

                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                
                #Downsampled the image
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)

                if self.inspection_config.mapCalculated[1] is False:  # Only checking the first camera for efficiency
                    for i in range(0, 2): #Make sure to check the camID
                        calib_file = self._save_dir + f"Calibration_camera_{i}.yaml"
                        if os.path.exists(calib_file):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(calib_file)
                            h, w = self.mergeframe1.shape[:2]
                            self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(
                                camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2
                            )
                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map calculated for Camera {i}")

                            scaled_file = self._save_dir + f"Calibration_camera_scaled_{i}.yaml"

                            if os.path.exists(scaled_file):
                                camera_matrix, dist_coeffs = self.load_matrix_from_yaml(scaled_file)
                                h, w = self.mergeframe1_scaled.shape[:2]
                                self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(
                                    camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2
                                )
                                print(f"Downscaled calibration map calculated for Camera {i}")
                            else:
                                print(f"Error: Scaled calibration file {scaled_file} does not exist.")

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

                    self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[0], self.inspection_config.map2_downscaled[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                    self.mergeframe1_scaled = cv2.rotate(self.mergeframe1_scaled, cv2.ROTATE_180)
                    self.mergeframe2_scaled = cv2.rotate(self.mergeframe2_scaled, cv2.ROTATE_180)

                    if self.inspection_config.widget in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_wide_scaled, (int(self.wide_planarize[1]/(self.scale_factor)), int(self.wide_planarize[0]/(self.scale_factor))))
                        if self.inspection_config.widget in [5, 6, 7, 8, 9, 10, 11, 12]: # emit katabu
                            if self.inspection_config.widget in [5, 7, 9, 11]:
                                #katabu L is blank
                                #katabu R is cropped image
                                self.katabuImageL_scaled = self.createBlackImage(width=256, height=128)
                                self.katabuImageR_scaled = self.frameCrop(self.combinedImage_scaled, self.katabuImageR_Crop[0]/self.scale_factor, self.katabuImageR_Crop[1]/self.scale_factor, self.katabuImageR_Crop[2]/self.scale_factor, self.katabuImageR_Crop[3]/self.scale_factor, self.katabuImageR_Crop[4], self.katabuImageR_Crop[5])
    
                            if self.inspection_config.widget in [6, 8, 10, 12]: 
                                #katabu L is cropped image
                                #katabu R is blank
                                self.katabuImageL_scaled = self.frameCrop(self.combinedImage_scaled, self.katabuImageL_Crop[0]/self.scale_factor, self.katabuImageL_Crop[1]/self.scale_factor, self.katabuImageL_Crop[2]/self.scale_factor, self.katabuImageL_Crop[3]/self.scale_factor, self.katabuImageL_Crop[4], self.katabuImageL_Crop[5])
                                self.katabuImageR_scaled = self.createBlackImage(width=256, height=128)

                            self.katabuImageL_scaled = self.convertQImage(self.katabuImageL_scaled)
                            self.katabuImageR_scaled = self.convertQImage(self.katabuImageR_scaled)

                            self.partKatabuL.emit(self.katabuImageL_scaled)
                            self.partKatabuR.emit(self.katabuImageR_scaled)

                            if self.inspection_config.widget in [5, 6, 7, 8]: # also emit clip
                                self.clipImage1 = self.frameCrop(self.mergeframe1, self.clipImage1_Crop[0], self.clipImage1_Crop[1], self.clipImage1_Crop[2], self.clipImage1_Crop[3], self.clipImage1_Crop[4], self.clipImage1_Crop[5])
                                self.clipImage2 = self.frameCrop(self.mergeframe1, self.clipImage2_Crop[0], self.clipImage2_Crop[1], self.clipImage2_Crop[2], self.clipImage2_Crop[3], self.clipImage2_Crop[4], self.clipImage2_Crop[5])
                                self.clipImage3 = self.frameCrop(self.mergeframe2, self.clipImage3_Crop[0], self.clipImage3_Crop[1], self.clipImage3_Crop[2], self.clipImage3_Crop[3], self.clipImage3_Crop[4], self.clipImage3_Crop[5])
                                
                                self.clipImage1 = cv2.rotate(self.clipImage1, cv2.ROTATE_180)
                                self.clipImage2 = cv2.rotate(self.clipImage2, cv2.ROTATE_180)
                                self.clipImage3 = cv2.rotate(self.clipImage3, cv2.ROTATE_180)

                                self.clipImage1 = self.convertQImage(self.clipImage1)
                                self.clipImage2 = self.convertQImage(self.clipImage2)
                                self.clipImage3 = self.convertQImage(self.clipImage3)

                                self.clip1Signal.emit(self.clipImage1)
                                self.clip2Signal.emit(self.clipImage2)
                                self.clip3Signal.emit(self.clipImage3)


                    self.InspectionResult_PitchMeasured = [None]*30
                    self.InspectionResult_PitchResult = [None]*30
                    self.InspectionResult_DeltaPitch = [None]*30

                    if self.combinedImage_scaled is not None:
                        #resize to 1791 x 428
                        self.combinedImage_scaled = self.downSampling(self.combinedImage_scaled, width=1791, height=428)
                        self.partCam.emit(self.convertQImage(self.combinedImage_scaled))
        
                    self.P82833W050P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P82833W050PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

                    self.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                    self.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

            #for the kengen
            if self.inspection_config.widget in [9, 10, 11, 12]:    

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")
                    
                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0)

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False

                    if self.inspection_config.kensainNumber != "10194":
                        imgresults = cv2.cvtColor(self.combinedImage_scaled, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(imgresults)
                        font = ImageFont.truetype(self.kanjiFontPath, 60)
                        draw = ImageDraw.Draw(img_pil)
                        centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                        draw.text((centerpos[0]-800, centerpos[1]+20), u"管理者権限が必要", font=font, fill=(160, 200, 10, 0))
                        imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        play_alarm_sound()
                        self.combinedImage_scaled = imgResult
                        self.partCam.emit(self.convertQImage(self.combinedImage_scaled))
                        time.sleep(2)
                        continue

                    if self.InspectionTimeStart is not None:

                        if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                            print("Inspection Time is over")
                            self.InspectionTimeStart = time.time()

                            self.emit = self.combinedImage_scaled
                            if self.emit is None:
                                self.emit = np.zeros((428, 1791, 3), dtype=np.uint8)

                            self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                            self.partCam.emit(self.convertQImage(self.emit))

                            self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                            self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                            self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                            self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                            self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                            self.InspectionImages[0] = self.combinedImage.copy()
                            self.InspectionImages_bgr[0] =self.combinedImage.copy()
                            self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                            if self.inspection_config.widget in [5, 6, 7, 8, 9, 10, 11, 12]: # emit katabu
                                if self.inspection_config.widget in [5, 7, 9, 11]:
                                    #katabu L is blank
                                    #katabu R is cropped image
                                    self.katabuImageL = self.createBlackImage(width=256, height=128)
                                    self.katabuImageR = self.frameCrop(self.combinedImage, self.katabuImageR_Crop[0], self.katabuImageR_Crop[1], self.katabuImageR_Crop[2], self.katabuImageR_Crop[3], self.katabuImageR_Crop[4], self.katabuImageR_Crop[5])
                                    self.katabuImage = self.katabuImageR.copy()
                                    self.katabuImage_init = self.katabuImageR.copy()
                                if self.inspection_config.widget in [6, 8, 10, 12]: 
                                    #katabu L is cropped image
                                    #katabu R is blank
                                    self.katabuImageL = self.frameCrop(self.combinedImage, self.katabuImageL_Crop[0], self.katabuImageL_Crop[1], self.katabuImageL_Crop[2], self.katabuImageL_Crop[3], self.katabuImageL_Crop[4], self.katabuImageL_Crop[5])
                                    self.katabuImageR = self.createBlackImage(width=256, height=128)
                                    self.katabuImage = self.katabuImageL.copy()
                                    self.katabuImage_init = self.katabuImageL.copy()

                                self.partKatabuL.emit(self.convertQImage(self.katabuImageL))
                                self.partKatabuR.emit(self.convertQImage(self.katabuImageR))

                            for i in range(len(self.InspectionImages)):
                                self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages_bgr[i], 
                                            self.P828XXW0X0P_CLIP_Model, 
                                            slice_height=1280, slice_width=1280, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.2,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=False
                                        )
                                if self.inspection_config.widget in [5, 7, 9, 11]:
                                    self.InspectionResult_KatabuDetection = self.P828XXW0X0P_KATABU_Model(cv2.cvtcolor(self.katabuImage, cv2.COLOR_BGR2RGB),
                                                                                                        stream=True,
                                                                                                        verbose=False,
                                                                                                        conf=0.1,
                                                                                                        iou=0.5)

                                if self.inspection_config.widget in [6, 8, 10, 12]: 
                                    self.InspectionResult_KatabuDetection = self.P828XXW0X0P_KATABU_Model(cv2.cvtColor(self.katabuImage, cv2.COLOR_BGR2RGB),
                                                                                                        stream=True,
                                                                                                        verbose=False,
                                                                                                        conf=0.1,
                                                                                                        iou=0.5)    
                                    
                                self.InspectionImages[i], self.InspectionImagesKatabu[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i] = P828XXW0X0P_check(self.InspectionImages[i], self.katabuImage,
                                                                                                                                                                                                                self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                self.InspectionResult_KatabuDetection,
                                                                                                                                                                                                                self.widget_name_map[self.inspection_config.widget])


                                for i in range(len(self.InspectionResult_Status)):
                                    if self.InspectionResult_Status[i] == "OK": 
                                        # Increment the 'OK' count at the appropriate index (1)
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1
                                        play_ok_sound()

                                    elif self.InspectionResult_Status[i] == "NG": 
                                        # Increment the 'NG' count at the appropriate index (0)
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1
                                        play_ng_sound()

                            self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])
                            self.save_image_result_withKatabu(self.combinedImage, self.InspectionImages[0], self.katabuImage_init, self.InspectionImagesKatabu[0], self.InspectionResult_Status[0])

                            self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                    numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                    currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                    deltaTime = 0.0,
                                    kensainName = self.inspection_config.kensainNumber, 
                                    detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                                    delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                                    total_length=0)




                            # print(f"Measured Pitch: {self.InspectionResult_PitchMeasured}")
                            # print(f"Delta Pitch: {self.InspectionResult_DeltaPitch}")
                            # print(f"Pirch Results: {self.InspectionResult_PitchResult}")

                            # #Add custom text to the image
                            # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 10 == 0 and self.InspectionResult_Status[0] == "OK" and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0 :
                            #     if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 150 == 0:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"ダンボールに入れてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_konpou_sound()
                            #         self.InspectionImages[0] = imgResult

                            #     else:
                            #         imgresults = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_BGR2RGB)
                            #         img_pil = Image.fromarray(imgresults)
                            #         font = ImageFont.truetype(self.kanjiFontPath, 120)
                            #         draw = ImageDraw.Draw(img_pil)
                            #         centerpos = (imgresults.shape[1] // 2, imgresults.shape[0] // 2) 
                            #         draw.text((centerpos[0]-900, centerpos[1]+20), u"束ねてください", font=font, fill=(5, 80, 160, 0))
                            #         imgResult = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            #         play_keisoku_sound()         
                            #         self.InspectionImages[0] = imgResult                         

                            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                            self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=428)

                            self.P82833W050PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82832W040PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82833W090PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                            self.P82832W080PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)


                            self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                            self.partCam.emit(self.converQImageRGB(self.InspectionImages[0]))

                            if self.inspection_config.widget in [5, 7, 9, 11]:
                                self.partKatabuR.emit(self.convertQImage(self.InspectionImagesKatabu[0]))
                            if self.inspection_config.widget in [6, 8, 10, 12]: 
                                self.partKatabuL.emit(self.convertQImage(self.InspectionImagesKatabu[0]))


                            
                            

                            time.sleep(1.5)

            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        self.msleep(5)

    def setCounterFalse(self):
        self.inspection_config.furyou_plus = False
        self.inspection_config.furyou_minus = False
        self.inspection_config.kansei_plus = False
        self.inspection_config.kansei_minus = False
        self.inspection_config.furyou_plus_10 = False
        self.inspection_config.furyou_minus_10 = False
        self.inspection_config.kansei_plus_10 = False
        self.inspection_config.kansei_minus_10 = False

    def manual_adjustment(self, currentPart, Totalpart,
                          furyou_plus, furyou_minus, 
                          furyou_plus_10, furyou_minus_10,
                          kansei_plus, kansei_minus,
                          kansei_plus_10, kansei_minus_10):
        
        ok_count_current = currentPart[0]
        ng_count_current = currentPart[1]
        ok_count_total = Totalpart[0]
        ng_count_total = Totalpart[1]
        
        if furyou_plus:
            ng_count_current += 1
            ng_count_total += 1

        if furyou_plus_10:
            ng_count_current += 10
            ng_count_total += 10

        if furyou_minus and ng_count_current > 0 and ng_count_total > 0:
            ng_count_current -= 1
            ng_count_total -= 1
        
        if furyou_minus_10 and ng_count_current > 9 and ng_count_total > 9:
            ng_count_current -= 10
            ng_count_total -= 10

        if kansei_plus:
            ok_count_current += 1
            ok_count_total += 1

        if kansei_plus_10:
            ok_count_current += 10
            ok_count_total += 10

        if kansei_minus and ok_count_current > 0 and ok_count_total > 0:
            ok_count_current -= 1
            ok_count_total -= 1

        if kansei_minus_10 and ok_count_current > 9 and ok_count_total > 9:
            ok_count_current -= 10
            ok_count_total -= 10

        self.setCounterFalse()

        self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                numofPart = [ok_count_total, ng_count_total], 
                currentnumofPart = [ok_count_current, ng_count_current],
                deltaTime = 0.0,
                kensainName = self.inspection_config.kensainNumber, 
                detected_pitch_str = "MANUAL", 
                delta_pitch_str = "MANUAL", 
                total_length=0)

        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, detected_pitch_str, 
                             delta_pitch_str, total_length):
        # Ensure all inputs are strings or compatible types

        timestamp = datetime.now()
        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H:%M:%S")

        partname = str(partname)
        numofPart = str(numofPart)
        currentnumofPart = str(currentnumofPart)
        timestamp_hour = str(timestamp_hour)
        timestamp_date = str(timestamp_date)
        deltaTime = float(deltaTime)  # Ensure this is a float
        kensainName = str(kensainName)
        detected_pitch_str = str(detected_pitch_str)
        delta_pitch_str = str(delta_pitch_str)
        total_length = float(total_length)  # Ensure this is a float

        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length))
        self.conn.commit()

    def get_last_entry_currentnumofPart(self, part_name):
        self.cursor.execute('''
        SELECT currentnumofPart 
        FROM inspection_results 
        WHERE partName = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name,))
        
        row = self.cursor.fetchone()
        if row:
            currentnumofPart = eval(row[0])
            return currentnumofPart
        else:
            return [0, 0]
            
    def get_last_entry_total_numofPart(self, part_name):
        # Get today's date in yyyymmdd format
        today_date = datetime.now().strftime("%Y%m%d")

        self.cursor.execute('''
        SELECT numofPart 
        FROM inspection_results 
        WHERE partName = ? AND timestampDate = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name, today_date))
        
        row = self.cursor.fetchone()
        if row:
            numofPart = eval(row[0])  # Convert the string tuple to an actual tuple
            return numofPart
        else:
            return [0, 0]  # Default values if no entry is found

    def draw_status_text_PIL(self, image, text, color, size = "normal", x_offset = 0, y_offset = 0):

        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2

        if size == "large":
            font_scale = 130.0

        if size == "normal":
            font_scale = 100.0

        elif size == "small":
            font_scale = 50.0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.kanjiFontPath, font_scale)

        draw.text((center_x + x_offset, center_y + y_offset), text, font=font, fill=color)  
        # Convert back to BGR for OpenCV compatibility
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return image

    def save_image(self, image):
        dir = "aikensa/inspection/" + self.widget_dir_map[self.inspection_config.widget]
        os.makedirs(dir, exist_ok=True)
        filename = dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        
        # Check if the file already exists and add an identifier if it does
        counter = 1
        while os.path.exists(filename):
            filename = dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{counter}.png"
            counter += 1
        
        cv2.imwrite(filename, image)

    def save_image_result(self, image_initial, image_result, result):
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_result)

    def save_image_result_withKatabu(self, image_initial, image_result, katabu_initial, katabu_result, result):
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_initial)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_katabu.png", katabu_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_result)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_katabu.png", katabu_result)

    def minitimerStart(self):
        self.timerStart_mini = time.time()
    
    def minitimerFinish(self, message = "OperationName"):
        self.timerFinish_mini = time.time()
        # self.fps_mini = 1/(self.timerFinish_mini - self.timerStart_mini)
        print(f"Time to {message} : {(self.timerFinish_mini - self.timerStart_mini) * 1000} ms")
        # print(f"FPS of {message} : {self.fps_mini}")

    def convertQImage(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image
    
    def convertQImageKatabu(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        # Ensure image data is converted to bytes
        processed_image = QImage(image.data.tobytes(), w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image

    def converQImageRGB(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image
    
    def downScaledImage(self, image, scaleFactor=1.0):
        #create a copy of the image
        resized_image = cv2.resize(image, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor, interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def downSampling(self, image, width=384, height=256):
        #create a copy of the image
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff

    def frameCrop(self,img, x=0, y=0, w=640, h=480, wout=640, hout=480):
        #crop and resize image to wout and hout
        #convert x y w h into int
        x, y, w, h, wout, hout = int(x), int(y), int(w), int(h), int(wout), int(hout)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        # print(f"X: {x}, Y: {y}, W: {w}, H: {h}")
        img = img[y:y+h, x:x+w]
        try:
            img = cv2.resize(img, (wout, hout), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print("An error occurred while cropping the image:", str(e))
        return img

    def createBlackImage(self, width, height): #create a black image with width and height
        return np.zeros((height, width, 3), dtype=np.uint8)

    def initialize_model(self):

        print("Model Dummy Loaded")

        # #Change based on the widget
        # For all CTR PLR AI model
        P828XXW0X0P_CLIP_Model = None
        P828XXW0X0P_KATABU_Model = None
        P828XXW0X0P_CLIPFLIP_Model= None #For clip yellow flip detection 
        P828XXW0X0P_SEGMENT_Model = None
        P828XXW0X0P_HAND_DETECT = None

        path_P828XXW0X0P_CLIP_Model = "./aikensa/models/P828XXW0X0P_detect.pt"
        path_P828XXW0X0P_KATABU_Model = "./aikensa/models/P828XXW0X0P_katabu.pt"
        path_P828XXW0X0P_CLIPFLIP_Model = "./aikensa/models/P828XXW0X0P_detect_flip.pt"
        path_P828XXW0X0P_SEGMENT_Model = "./aikensa/models/P828XXW0X0P_segment.pt"
        path_P828XXW0X0P_HAND_DETECT = "./aikensa/models/P828XXW0X0P_hand.pt"

        P828XXW0X0P_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",model_path=path_P828XXW0X0P_CLIP_Model,
                                                                            confidence_threshold=0.35,
                                                                            device="cuda:0")
        P828XXW0X0P_KATABU_Model = YOLO(path_P828XXW0X0P_KATABU_Model)
        P828XXW0X0P_SEGMENT_Model = YOLO(path_P828XXW0X0P_CLIPFLIP_Model)
        P828XXW0X0P_SEGMENT_Model = YOLO(path_P828XXW0X0P_SEGMENT_Model)

        self.P828XXW0X0P_CLIP_Model = P828XXW0X0P_CLIP_Model
        self.P828XXW0X0P_KATABU_Model = P828XXW0X0P_KATABU_Model
        self.P828XXW0X0P_CLIPFLIP_Model = P828XXW0X0P_CLIPFLIP_Model
        self.P828XXW0X0P_SEGMENT_Model = P828XXW0X0P_SEGMENT_Model

        print("Model Loaded")
        
    def stop(self):
        self.inspection_config.widget = -1
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        print("Inspection thread stopped.")