import inspect
from tabnanny import verbose
import cv2
import os
from datetime import datetime
import numpy as np
from scipy.fftpack import ifft
import yaml
import time
import logging
import sqlite3
import mysql.connector
import random

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.camscripts.cam_hole_init import initialize_hole_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from aikensa.scripts.scripts_method import scale_translation, load_register_map
from aikensa.scripts.scripts_img_processing import rescaled_image, resize_image, make_undistort_maps
from aikensa.scripts.scripts_img_processing import image_cropping


from dataclasses import dataclass, field
from typing import List

from aikensa.parts_config.sound import play_konpou_sound, play_keisoku_sound, play_ok_sound, play_ng_sound

from ultralytics import YOLO


from aikensa.parts_config.dailyTenken.dailyTenken import dailyTenken

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

    kensainNumber: str = None
    ppmsnumber : str = None
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

    P1_LH_Signal = pyqtSignal(QImage)
    P2_LH_Signal = pyqtSignal(QImage)
    P3_LH_Signal = pyqtSignal(QImage)
    P4_LH_Signal = pyqtSignal(QImage)
    P5_LH_Signal = pyqtSignal(QImage)

    P1_RH_Signal = pyqtSignal(QImage)
    P2_RH_Signal = pyqtSignal(QImage)
    P3_RH_Signal = pyqtSignal(QImage)
    P4_RH_Signal = pyqtSignal(QImage)
    P5_RH_Signal = pyqtSignal(QImage)

    P808387UA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P808397UA0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    
    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)

    requestModbusWrite = pyqtSignal(int, list)

    def __init__(self, inspection_config: InspectionConfig = None, modbus_client_thread=None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        self.modbus_client_thread = modbus_client_thread

        if self.modbus_client_thread is not None:
            self.modbus_client_thread.holdingUpdated.connect(self.on_holding_update)
            self.requestModbusWrite.connect(self.modbus_client_thread.write_holding_registers)
            self.modbus_client_thread.inputRead.connect(self.on_input_update)

        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.multiCam_stream = False

        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None
        self.cap_cam4 = None

        self.emit = None

        self.mergeframe1 = None
        self.mergeframe2 = None
        self.mergeframe3 = None
        self.mergeframe4 = None

        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None
        self.mergeframe3_scaled = None
        self.mergeframe4_scaled = None

        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None
        self.mergeframe3_downsampled = None
        self.mergeframe4_downsampled = None

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        self.homography_matrix3 = None
        self.homography_matrix4 = None

        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None
        self.homography_matrix3_scaled = None
        self.homography_matrix4_scaled = None

        self.H1 = None
        self.H2 = None
        self.H3 = None
        self.H4 = None

        self.H1_scaled = None
        self.H2_scaled = None
        self.H3_scaled = None
        self.H4_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.planarize = (1300, 3500)

        self.combinedImage_left = None
        self.combinedImage_left_scaled = None

        self.combinedImage_right = None
        self.combinedImage_right_scaled = None

        self.scale_factor = 5.0 #Scale Factor, might increase this later
        self.scale = 0.2

        self.P1_LH_image = None
        self.P2_LH_image = None
        self.P3_LH_image = None
        self.P4_LH_image = None
        self.P5_LH_image = None

        self.P1_RH_image = None
        self.P2_RH_image = None
        self.P3_RH_image = None
        self.P4_RH_image = None
        self.P5_RH_image = None

        #Scaled version
        self.P1_LH_image_scaled = None
        self.P2_LH_image_scaled = None
        self.P3_LH_image_scaled = None
        self.P4_LH_image_scaled = None
        self.P5_LH_image_scaled = None

        self.P1_RH_image_scaled = None
        self.P2_RH_image_scaled = None
        self.P3_RH_image_scaled = None
        self.P4_RH_image_scaled = None
        self.P5_RH_image_scaled = None

        #value for opencv cropping
        self.P1_LH_image_scaled_crop = [0, 9, 700, 34]
        self.P2_LH_image_scaled_crop = [0, 63, 700, 88]
        self.P3_LH_image_scaled_crop = [0, 117, 700, 142]
        self.P4_LH_image_scaled_crop = [0, 170, 700, 195]
        self.P5_LH_image_scaled_crop = [0, 225, 700, 250]

        self.P1_RH_image_scaled_crop = [0, 9, 700, 34]
        self.P2_RH_image_scaled_crop = [0, 63, 700, 88]
        self.P3_RH_image_scaled_crop = [0, 117, 700, 142]
        self.P4_RH_image_scaled_crop = [0, 170, 700, 195]
        self.P5_RH_image_scaled_crop = [0, 225, 700, 250]


        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.planarize = (1300, 3500)

        self.planarizeTransform_left = None
        self.planarizeTransform_right = None

        self.planarizeTransform_left_scaled = None
        self.planarizeTransform_right_scaled = None

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.InspectionImages = [None]*1
        self.InspectionImages_bgr = [None]*1

        self.InspectionImages_endSegmentation_Left = [None]*1
        self.InspectionImages_endSegmentation_Right = [None]*1

        self.InspectionResult_EndSegmentation_Left = [None]*5
        self.InspectionResult_EndSegmentation_Right = [None]*5

        self.InspectionSet = [None] * 10
        self.InspectionPitch = [None] * 10

        self.InspectionResult_PitchMeasured = [None]*30
        self.InspectionResult_PitchResult = [None]*30
        self.InspectionResult_DetectionID = [None]*30
        self.InspectionResult_Status = [None]*30
        self.InspectionResult_DeltaPitch = [None]*30
        self.InspectionResult_NGReason = [None]*30

        self.inspection_widget_indices = [7, 8, 21, 22, 23]
        self.inspection_widget_indices_without_dailytenken = [7, 8]

        self.widget_dir_map = {
            7: "808397UA0A",
            8: "808387UA0A",
            21: "dailyTenken_01",
            22: "dailyTenken_02",
            23: "dailyTenken_03",
        }

        #for widget name map, append the string "P" to the initial widget dir map
        self.widget_name_map = {key: f"P{value}" for key, value in self.widget_dir_map.items()}

        self.InspectionWaitTime = 1.0
        self.InspectionTimeStart = None

        self.ethernetTrigger = [0]*5
        
        this_dir = os.path.dirname(__file__)
        cam_config_path = os.path.abspath(os.path.join(this_dir, '..', 'config'))
        self.cam_config_file = os.path.join(cam_config_path, 'camera_config.yaml')
        
        with open(self.cam_config_file, 'r') as file:
            self.cam_map = yaml.safe_load(file)

        mysql_credentials_path = os.path.abspath(os.path.join(this_dir, '..', 'mysql'))
        self.mysql_credentials_file = os.path.join(mysql_credentials_path, 'id.yaml')

        if not os.path.exists(self.mysql_credentials_file):
            print(f"Error: MySQL credentials file {self.mysql_credentials_file} does not exist.")
            self.mysqlID = None
            self.mysqlPassword = None
            self.mysqlHost = None
            self.mysqlHostPort = None
            # Load MySQL credentials from the YAML file
        else:
            with open(self.mysql_credentials_file) as file:
                credentials = yaml.load(file, Loader=yaml.FullLoader)
                self.mysqlID = credentials["id"]
                self.mysqlPassword = credentials["pass"]
                self.mysqlHost = credentials["host"]
                self.mysqlHostPort = credentials["port"]

        self.holding_register_path = "./aikensa/modbus/holding_register_map.yaml"
        self.input_register_path = "./aikensa/modbus/input_register_map.yaml"

        self.holding_register_map = load_register_map(self.holding_register_path)
        self.input_register_map = load_register_map(self.input_register_path)

    @pyqtSlot(dict)
    def on_holding_update(self, reg_dict):
        # Only called whenever the Modbus thread emits new data.
        # self.partNumber = reg_dict.get(50, 0)
        # self.serialNumber_front = reg_dict.get(62, 0)
        # self.serialNumber_back  = reg_dict.get(63, 0)
        # self.InstructionCode    = reg_dict.get(100, 0)
        # print(f"Part Number: {self.partNumber}")
        # print(f"Serial Number Front: {self.serialNumber_front}")
        # print(f"Serial Number Back:  {self.serialNumber_back}")
        self.InstructionCode = 0 #for debug

    @pyqtSlot(dict)
    def on_input_update(self, reg_dict: dict):
        """
        reg_dict maps {address: value} for Input Registers (FC=4).
        Pull out your command & tray position here.
        """
        cmd_addr  = self.input_register_map["AIKENSACOMMAND"]  # e.g. 112
        tray_addr = self.input_register_map["TRAYPOSITION"]    # e.g. 113

        self.AIKENSA_COMMAND = reg_dict.get(cmd_addr, 0)
        self.TRAYPOSITION    = reg_dict.get(tray_addr,  0)
        print(f"AIKENSACOMMAND={self.AIKENSA_COMMAND}, TRAYPOSITION={self.TRAYPOSITION}")



    def release_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")

    def initialize_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")

        actual_camID = self.cam_map.get(0, -1)
        self.cap_cam1 = initialize_camera(actual_camID)

        actual_camID = self.cam_map.get(1, -1)
        self.cap_cam2 = initialize_camera(actual_camID)

        actual_camID = self.cam_map.get(2, -1)
        self.cap_cam3 = initialize_camera(actual_camID)

        actual_camID = self.cam_map.get(3, -1)
        self.cap_cam4 = initialize_camera(actual_camID)

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

        if not self.cap_cam3.isOpened():
            print(f"Failed to open camera with ID 3")
            self.cap_cam3 = None
        else:
            print(f"Initialized Camera on ID 3")

        if not self.cap_cam4.isOpened():
            print(f"Failed to open camera with ID 4")
            self.cap_cam4 = None
        else:
            print(f"Initialized Camera on ID 4")


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
            total_length REAL,
            resultpitch TEXT,
            status TEXT,
            NGreason TEXT,
            ClipInsertionMachine TEXT,
            PPMS TEXT
        )
        ''')

        self.conn.commit()

        #Initialize connection to mysql server if available
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.mysqlHost,
                user=self.mysqlID,
                password=self.mysqlPassword,
                port=self.mysqlHostPort,
                database="AIKENSAresults"
            )
            print(f"Connected to MySQL database at {self.mysqlHost}")
        except Exception as e:
            print(f"Error connecting to MySQL database: {e}")
            self.mysql_conn = None

        #try adding data to the schema in mysql
        if self.mysql_conn is not None:
            self.mysql_cursor = self.mysql_conn.cursor()
            self.mysql_cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_results (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                partName TEXT,
                numofPart TEXT,
                currentnumofPart TEXT,
                timestampHour TEXT,
                timestampDate TEXT,
                deltaTime REAL,
                kensainName TEXT,
                detected_pitch TEXT,
                delta_pitch TEXT,
                total_length REAL,
                resultpitch TEXT,
                status TEXT,
                NGreason TEXT,
                ClipInsertionMachine TEXT,
                PPMS TEXT
            )
            ''')
            self.mysql_conn.commit()

        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Models Initialized")

        self.current_cameraID = self.inspection_config.cameraID
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
                self.H1_scaled = scale_translation(self.H1, self.scale )
                print(f"Loaded homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)
                self.H2_scaled = scale_translation(self.H2, self.scale )
                print(f"Loaded homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                self.H3 = np.array(self.homography_matrix3)
                self.H3_scaled = scale_translation(self.H3, self.scale )
                print(f"Loaded homography matrix for camera 3")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                self.H4 = np.array(self.homography_matrix4)
                self.H4_scaled = scale_translation(self.H4, self.scale )
                print(f"Loaded homography matrix for camera 4")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_left.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_left.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_left = np.array(transform_list)
                self.planarizeTransform_left_scaled = scale_translation(self.planarizeTransform_left, self.scale )

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_right.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_right.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_right = np.array(transform_list)
                self.planarizeTransform_right_scaled = scale_translation(self.planarizeTransform_right, self.scale )

        while self.running:

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget > 0:

                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()

                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()

                self.mergeframe1_scaled = rescaled_image(self.mergeframe1, scale_factor=self.scale)
                self.mergeframe2_scaled = rescaled_image(self.mergeframe2, scale_factor=self.scale)
                self.mergeframe3_scaled = rescaled_image(self.mergeframe3, scale_factor=self.scale)
                self.mergeframe4_scaled = rescaled_image(self.mergeframe4, scale_factor=self.scale)


                self.mergeframe1_scaled = cv2.rotate(self.mergeframe1_scaled, cv2.ROTATE_180)
                self.mergeframe2_scaled = cv2.rotate(self.mergeframe2_scaled, cv2.ROTATE_180)
                self.mergeframe3_scaled = cv2.rotate(self.mergeframe3_scaled, cv2.ROTATE_180)
                self.mergeframe4_scaled = cv2.rotate(self.mergeframe4_scaled, cv2.ROTATE_180)

                # self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                # self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
                # self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                # self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
                
                if self.inspection_config.mapCalculated[1] is False:  # Only checking the first camera for efficiency
                    for i in range(0, 4): #Make sure to check the camID
                        calib_file = self._save_dir + f"Calibration_camera_{i}.yaml"
                        if os.path.exists(calib_file):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(calib_file)
                            h, w = self.mergeframe1.shape[:2]
                            # self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(
                            #     camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2
                            # )
                            (map1_full, map2_full), (map1_ds, map2_ds) = make_undistort_maps(
                                camera_matrix, dist_coeffs, (w, h), scale=self.scale
                            )

                            # store them
                            self.inspection_config.map1[i] = map1_full
                            self.inspection_config.map2[i] = map2_full
                            self.inspection_config.map1_downscaled[i] = map1_ds
                            self.inspection_config.map2_downscaled[i] = map2_ds

                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map calculated for Camera {i}")

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

                    self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[0], self.inspection_config.map2_downscaled[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


                    if self.inspection_config.widget in self.inspection_widget_indices_without_dailytenken:
                        self.combinedImage_left_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_left_scaled = warpTwoImages_template(self.combinedImage_left_scaled, self.mergeframe2_scaled, self.H2_scaled)

                        self.combinedImage_right_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe3_scaled, self.H3_scaled)
                        self.combinedImage_right_scaled = warpTwoImages_template(self.combinedImage_right_scaled, self.mergeframe4_scaled, self.H4_scaled)

                        self.combinedImage_left_scaled = cv2.warpPerspective(self.combinedImage_left_scaled, self.planarizeTransform_left_scaled, (int(self.planarize[1]*(self.scale)), int(self.planarize[0]*(self.scale))))
                        self.combinedImage_right_scaled = cv2.warpPerspective(self.combinedImage_right_scaled, self.planarizeTransform_right_scaled, (int(self.planarize[1]*(self.scale)), int(self.planarize[0]*(self.scale))))

                        self.P1_LH_image_scaled = image_cropping(self.combinedImage_left_scaled, self.P1_LH_image_scaled_crop)
                        self.P2_LH_image_scaled = image_cropping(self.combinedImage_left_scaled, self.P2_LH_image_scaled_crop)
                        self.P3_LH_image_scaled = image_cropping(self.combinedImage_left_scaled, self.P3_LH_image_scaled_crop)
                        self.P4_LH_image_scaled = image_cropping(self.combinedImage_left_scaled, self.P4_LH_image_scaled_crop)
                        self.P5_LH_image_scaled = image_cropping(self.combinedImage_left_scaled, self.P5_LH_image_scaled_crop)

                        self.P1_RH_image_scaled = image_cropping(self.combinedImage_right_scaled, self.P1_RH_image_scaled_crop)
                        self.P2_RH_image_scaled = image_cropping(self.combinedImage_right_scaled, self.P2_RH_image_scaled_crop)
                        self.P3_RH_image_scaled = image_cropping(self.combinedImage_right_scaled, self.P3_RH_image_scaled_crop)
                        self.P4_RH_image_scaled = image_cropping(self.combinedImage_right_scaled, self.P4_RH_image_scaled_crop)
                        self.P5_RH_image_scaled = image_cropping(self.combinedImage_right_scaled, self.P5_RH_image_scaled_crop)

                        #resize all to 1791x71
                        self.P1_LH_image_scaled = resize_image(self.P1_LH_image_scaled, width=1791, height=71)
                        self.P2_LH_image_scaled = resize_image(self.P2_LH_image_scaled, width=1791, height=71)
                        self.P3_LH_image_scaled = resize_image(self.P3_LH_image_scaled, width=1791, height=71)
                        self.P4_LH_image_scaled = resize_image(self.P4_LH_image_scaled, width=1791, height=71)
                        self.P5_LH_image_scaled = resize_image(self.P5_LH_image_scaled, width=1791, height=71)

                        self.P1_RH_image_scaled = resize_image(self.P1_RH_image_scaled, width=1791, height=71)
                        self.P2_RH_image_scaled = resize_image(self.P2_RH_image_scaled, width=1791, height=71)
                        self.P3_RH_image_scaled = resize_image(self.P3_RH_image_scaled, width=1791, height=71)
                        self.P4_RH_image_scaled = resize_image(self.P4_RH_image_scaled, width=1791, height=71)
                        self.P5_RH_image_scaled = resize_image(self.P5_RH_image_scaled, width=1791, height=71)



                        # cv2.imwrite("./P1_LH_image_scaled.png", self.P1_LH_image_scaled)
                        # cv2.imwrite("./P2_LH_image_scaled.png", self.P2_LH_image_scaled)
                        # cv2.imwrite("./P3_LH_image_scaled.png", self.P3_LH_image_scaled)
                        # cv2.imwrite("./P4_LH_image_scaled.png", self.P4_LH_image_scaled)
                        # cv2.imwrite("./P5_LH_image_scaled.png", self.P5_LH_image_scaled)

                        # cv2.imwrite("./P1_RH_image_scaled.png", self.P1_RH_image_scaled)
                        # cv2.imwrite("./P2_RH_image_scaled.png", self.P2_RH_image_scaled)
                        # cv2.imwrite("./P3_RH_image_scaled.png", self.P3_RH_image_scaled)
                        # cv2.imwrite("./P4_RH_image_scaled.png", self.P4_RH_image_scaled)
                        # cv2.imwrite("./P5_RH_image_scaled.png", self.P5_RH_image_scaled)
                        
                    self.InspectionResult_PitchMeasured = [None]*30
                    self.InspectionResult_PitchResult = [None]*30
                    self.InspectionResult_DeltaPitch = [None]*30

                    
                    if self.P1_LH_image_scaled is not None:
                        self.P1_LH_Signal.emit(self.convertQImage(self.P1_LH_image_scaled))
                    if self.P2_LH_image_scaled is not None:
                        self.P2_LH_Signal.emit(self.convertQImage(self.P2_LH_image_scaled))
                    if self.P3_LH_image_scaled is not None:
                        self.P3_LH_Signal.emit(self.convertQImage(self.P3_LH_image_scaled))
                    if self.P4_LH_image_scaled is not None:
                        self.P4_LH_Signal.emit(self.convertQImage(self.P4_LH_image_scaled))
                    if self.P5_LH_image_scaled is not None:
                        self.P5_LH_Signal.emit(self.convertQImage(self.P5_LH_image_scaled))

                    if self.P1_RH_image_scaled is not None:
                        self.P1_RH_Signal.emit(self.convertQImage(self.P1_RH_image_scaled))
                    if self.P2_RH_image_scaled is not None:
                        self.P2_RH_Signal.emit(self.convertQImage(self.P2_RH_image_scaled))
                    if self.P3_RH_image_scaled is not None:
                        self.P3_RH_Signal.emit(self.convertQImage(self.P3_RH_image_scaled))
                    if self.P4_RH_image_scaled is not None:
                        self.P4_RH_Signal.emit(self.convertQImage(self.P4_RH_image_scaled))
                    if self.P5_RH_image_scaled is not None:
                        self.P5_RH_Signal.emit(self.convertQImage(self.P5_RH_image_scaled))

                    # self.P82832W080P_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)

            if self.inspection_config.widget in [7]:    

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
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET")

                if self.InspectionTimeStart is None:
                    self.InspectionTimeStart = time.time()

                if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                    self.inspection_config.doInspection = False

                if self.AIKENSA_COMMAND == 1:
                    #This is JAKA asking for part set inspection
                    if self.TRAYPOSITION == 1:
                        #This means the tray is on the left side
                        #Generate random 0 and 1 for the inspection result
                        for i in range(len(self.InspectionSet)):
                            self.InspectionSet[i] = random.randint(0, 1)
                        
                        self.requestModbusWrite.emit(self.holding_register_map["P1_EXIST"], [self.InspectionSet[0]])
                        self.requestModbusWrite.emit(self.holding_register_map["P2_EXIST"], [self.InspectionSet[1]])
                        self.requestModbusWrite.emit(self.holding_register_map["P3_EXIST"], [self.InspectionSet[2]])
                        self.requestModbusWrite.emit(self.holding_register_map["P4_EXIST"], [self.InspectionSet[3]])
                        self.requestModbusWrite.emit(self.holding_register_map["P5_EXIST"], [self.InspectionSet[4]])
                        self.requestModbusWrite.emit(self.holding_register_map["AIKENSA_STATUS"], [1])
                        time.sleep(0.5)
                        self.requestModbusWrite.emit(self.holding_register_map["AIKENSA_STATUS"], [0])

                    if self.TRAYPOSITION == 2:
                        #This means the tray is on the right side
                        for i in range(len(self.InspectionSet)):
                            self.InspectionSet[i] = random.randint(0, 1)

                        self.requestModbusWrite.emit(self.holding_register_map["P1_EXIST"], [self.InspectionSet[0]])
                        self.requestModbusWrite.emit(self.holding_register_map["P2_EXIST"], [self.InspectionSet[1]])
                        self.requestModbusWrite.emit(self.holding_register_map["P3_EXIST"], [self.InspectionSet[2]])
                        self.requestModbusWrite.emit(self.holding_register_map["P4_EXIST"], [self.InspectionSet[3]])
                        self.requestModbusWrite.emit(self.holding_register_map["P5_EXIST"], [self.InspectionSet[4]])
                        self.requestModbusWrite.emit(self.holding_register_map["AIKENSA_STATUS"], [1])
                        time.sleep(0.5)
                        self.requestModbusWrite.emit(self.holding_register_map["AIKENSA_STATUS"], [0])

                    if self.TRAYPOSITION == 0:
                        #This means that the tray is in the wrong position
                        print("Tray Position is not set correctly. Please set the tray to the left or right side.")
                        #Need to print in the status bar so user can see and notice it clearly


                # if self.inspection_config.doInspection is True:
                #     self.inspection_config.doInspection = False
                #     print("Inspection Started")
                #     print(self.inspection_config.kensainNumber)


                #     if self.InspectionTimeStart is not None:

                #         if time.time() - self.InspectionTimeStart > self.InspectionWaitTime:
                #             print("Inspection Time is over")
                #             self.InspectionTimeStart = time.time()

                #             self.emit = self.combinedImage_scaled
                #             if self.emit is None:
                #                 self.emit = np.zeros((428, 1791, 3), dtype=np.uint8)

                #             self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                #             self.partCam.emit(self.convertQImage(self.emit))

                #             self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[0], self.inspection_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                #             self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                #             self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                #             self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                #             self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                #             self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                #             self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_wide, (int(self.wide_planarize[1]), int(self.wide_planarize[0])))

                #             self.InspectionImages[0] = self.combinedImage.copy()
                #             self.InspectionImages_bgr[0] =self.combinedImage.copy()
                #             self.InspectionImages_bgr[0] = cv2.cvtColor(self.InspectionImages_bgr[0], cv2.COLOR_BGR2RGB)

                #             if self.inspection_config.widget in [5, 6, 7, 8, 9, 10, 11, 12]: # emit katabu
                #                 if self.inspection_config.widget in [5, 7, 9, 11]:
                #                     #katabu L is blank
                #                     #katabu R is cropped image
                #                     self.katabuImageL = self.createBlackImage(width=256, height=128)
                #                     self.katabuImageR = self.frameCrop(self.combinedImage, self.katabuImageR_Crop[0], self.katabuImageR_Crop[1], self.katabuImageR_Crop[2], self.katabuImageR_Crop[3], self.katabuImageR_Crop[4], self.katabuImageR_Crop[5])
                #                     self.katabuImage = self.katabuImageR.copy()
                #                     self.katabuImage_init = self.katabuImageR.copy()
                #                 if self.inspection_config.widget in [6, 8, 10, 12]: 
                #                     #katabu L is cropped image
                #                     #katabu R is blank
                #                     self.katabuImageL = self.frameCrop(self.combinedImage, self.katabuImageL_Crop[0], self.katabuImageL_Crop[1], self.katabuImageL_Crop[2], self.katabuImageL_Crop[3], self.katabuImageL_Crop[4], self.katabuImageL_Crop[5])
                #                     self.katabuImageR = self.createBlackImage(width=256, height=128)
                #                     self.katabuImage = self.katabuImageL.copy()
                #                     self.katabuImage_init = self.katabuImageL.copy()

                #                 self.partKatabuL.emit(self.convertQImage(self.katabuImageL))
                #                 self.partKatabuR.emit(self.convertQImage(self.katabuImageR))

                #             for i in range(len(self.InspectionImages)):
                #                 self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                #                             self.InspectionImages_bgr[i], 
                #                             self.P828XXW0X0P_CLIP_Model, 
                #                             slice_height=1280, slice_width=1280, 
                #                             overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                #                             postprocess_match_metric="IOS",
                #                             postprocess_match_threshold=0.2,
                #                             postprocess_class_agnostic=True,
                #                             postprocess_type="GREEDYNMM",
                #                             verbose=0,
                #                             perform_standard_pred=False
                #                         )
                #                 if self.inspection_config.widget in [5, 7, 9, 11]:
                #                     self.InspectionResult_KatabuDetection = self.P828XXW0X0P_KATABU_Model(cv2.cvtColor(self.katabuImage, cv2.COLOR_BGR2RGB),
                #                                                                                         stream=True,
                #                                                                                         verbose=False,
                #                                                                                         conf=0.1,
                #                                                                                         iou=0.5)

                #                 if self.inspection_config.widget in [6, 8, 10, 12]: 
                #                     self.InspectionResult_KatabuDetection = self.P828XXW0X0P_KATABU_Model(cv2.cvtColor(self.katabuImage, cv2.COLOR_BGR2RGB),
                #                                                                                         stream=True,
                #                                                                                         verbose=False,
                #                                                                                         conf=0.1,
                #                                                                                         iou=0.5)    
                                    
                #                 self.InspectionImages[i], self.InspectionImagesKatabu[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i]  = P828XXW0X0P_check(self.InspectionImages[i], self.katabuImage,
                #                                                                                                                                                                                                 self.InspectionResult_ClipDetection[i].object_prediction_list,
                #                                                                                                                                                                                                 self.InspectionResult_KatabuDetection,
                #                                                                                                                                                                                                 self.widget_name_map[self.inspection_config.widget])


                #                 for i in range(len(self.InspectionResult_Status)):
                #                     if self.InspectionResult_Status[i] == "OK": 
                #                         # Increment the 'OK' count at the appropriate index (1)
                #                         self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                #                         self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1
                #                         play_ok_sound()

                #                     elif self.InspectionResult_Status[i] == "NG": 
                #                         # Increment the 'NG' count at the appropriate index (0)
                #                         self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                #                         self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1
                #                         play_ng_sound()

                #             self.save_image_result(self.combinedImage, self.InspectionImages[0], self.InspectionResult_Status[0])
                #             self.save_image_result_withKatabu(self.combinedImage, self.InspectionImages[0], self.katabuImage_init, self.InspectionImagesKatabu[0], self.InspectionResult_Status[0])

                #             self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                #                     numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                #                     currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                #                     deltaTime = 0.0,
                #                     kensainName = self.inspection_config.kensainNumber, 
                #                     detected_pitch_str = self.InspectionResult_PitchMeasured[0], 
                #                     delta_pitch_str = self.InspectionResult_DeltaPitch[0], 
                #                     total_length=0,
                #                     resultPitch = self.InspectionResult_PitchResult[0], 
                #                     status = self.InspectionResult_Status[0], 
                #                     NGreason = self.InspectionResult_NGReason[0])


                #             self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                #             self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
                #             self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1791, height=428)

                #             self.P82833W050PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                #             self.P82832W040PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                #             self.P82833W090PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
                #             self.P82832W080PKENGEN_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)


                #             self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                #             self.partCam.emit(self.converQImageRGB(self.InspectionImages[0]))

                #             if self.inspection_config.widget in [5, 7, 9, 11]:
                #                 self.partKatabuR.emit(self.convertQImage(self.InspectionImagesKatabu[0]))
                #             if self.inspection_config.widget in [6, 8, 10, 12]: 
                #                 self.partKatabuL.emit(self.convertQImage(self.InspectionImagesKatabu[0]))
                            
                            

                #             time.sleep(1.5)


            self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
            self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)

        # self.msleep(5)
        time.sleep(0.02)

    def add_admin_warning(self, image, text, font_path):
        try:
            print("Image shape:", image.shape)
            print("Data type:", image.dtype)
            img_results = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_results = np.ascontiguousarray(img_results)
            img_pil = Image.fromarray(img_results)
            font = ImageFont.truetype(font_path, 60)
            draw = ImageDraw.Draw(img_pil)
            center_pos = (img_results.shape[1] // 2, img_results.shape[0] // 2) 
            draw.text((center_pos[0]-800, center_pos[1]+20), text, font=font, fill=(160, 200, 10, 0))
            img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_result
        except Exception as e:
            print(f"Error adding admin warning: {str(e)}")
            return image  # Return the original image if an error occurs

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
                total_length=0,
                resultPitch = "MANUAL",
                status = "MANUAL",
                NGreason = "MANUAL",
                PPMS="MANUAL")

        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, detected_pitch_str, 
                             delta_pitch_str, total_length, 
                             resultPitch, status, NGreason, PPMS="Null"):
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
        resultPitch = str(resultPitch)
        status = str(status)
        NGreason = str(NGreason)

        if PPMS != "Null":
            PPMS = str(PPMS)
        else:
            PPMS = "Null"


        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, PPMS)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason, PPMS))
        self.conn.commit()

        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        #Also save to mysql cursor
        self.mysql_cursor.execute('''
        INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason, PPMS)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason, PPMS))
        self.mysql_conn.commit()

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

    def save_image_result(self, image_initial, image_result, result):
        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image_result)

    def convertQImage(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image
    
    def converQImageRGB(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image
    

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

    def createBlankGreenImage (self, width=1791, height=169):
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)
        blank_image[:] = (0, 255, 0)
        return blank_image

    def initialize_model(self):
        # Define model paths
        path_P8083X7UA0A_CLIP_Model = "./aikensa/models/P8083X7UA0A_detect.pt"
        path_P8083X7UA0A_SEGMENT_Model = "./aikensa/models/P8083X7UA0A_segment.pt"
        path_NICHIJOU_TENKEN_Model = "./aikensa/models/AIKENSA23GO_NICHIJOU_TENKEN.pt"

        # Initialize each model with existence check
        if os.path.exists(path_P8083X7UA0A_CLIP_Model):
            self.P8083X7UA0A_CLIP_Model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=path_P8083X7UA0A_CLIP_Model,
                confidence_threshold=0.5,
                device="cuda:0"
            )
        else:
            print(f"Model file {path_P8083X7UA0A_CLIP_Model} does not exist. Initializing as None.")
            self.P8083X7UA0A_CLIP_Model = None

        if os.path.exists(path_NICHIJOU_TENKEN_Model):
            self.NICHIJOU_TENKEN_Model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=path_NICHIJOU_TENKEN_Model,
                confidence_threshold=0.5,
                device="cuda:0"
            )
        else:
            print(f"Model file {path_NICHIJOU_TENKEN_Model} does not exist. Initializing as None.")
            self.NICHIJOU_TENKEN_Model = None

        print("Model Loaded")
        
    def stop(self):
        self.inspection_config.widget = -1
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        print("Inspection thread stopped.")

    def add_columns(self, cursor, table_name, columns):
        for column_name, column_type in columns:
            try:
                cursor.execute(f'''
                ALTER TABLE {table_name}
                ADD COLUMN {column_name} {column_type};
                ''')
                print(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                print(f"Could not add column {column_name}: {e}")
