from tabnanny import verbose
import cv2
import os
from datetime import datetime
from networkx import center, jaccard_coefficient
import numpy as np
from shapely import length
from sympy import fu
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
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound

from ultralytics import YOLO
# from aikensa.parts_config.ctrplr_8283XW0W0P import partcheck as ctrplrCheck
# from aikensa.parts_config.ctrplr_8283XW0W0P import dailytenkencheck
from aikensa.parts_config.P658207LE0A import partcheck as P658207LE0A_check

from PIL import ImageFont, ImageDraw, Image

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

    mapCalculated: list = field(default_factory=lambda: [False]*10) #for 10 cameras
    map1: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    map1_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras

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

    today_numofPart: list = field(default_factory=lambda: [0, 0] * 10)
    current_numofPart: list = field(default_factory=lambda: [0, 0] * 10)


class InspectionThread(QThread):

    part1Cam = pyqtSignal(QImage)


    P5819A107_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P5902A509_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P5902A510_InspectionResult_PitchMeasured = pyqtSignal(list, list)
    P658207LE0A_InspectionResult_PitchMeasured = pyqtSignal(list, list)



    # ethernet_status_red_tenmetsu = pyqtSignal(list)
    # ethernet_status_green_hold = pyqtSignal(list)
    # ethernet_status_red_hold = pyqtSignal(list)

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

        self.bottomframe = None
        self.mergeframe1 = None
        self.mergeframe2 = None
        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None
        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None
        self.H1 = None
        self.H2 = None
        self.H1_scaled = None
        self.H2_scaled = None

        self.part1Crop = None
        self.part2Crop = None
        
        self.part1Crop_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.combinedImage_narrow = None
        self.combinedImage_narrow_scaled = None
        self.combinedImage_wide = None
        self.combinedImage_wide_scaled = None

        self.scale_factor = 5.0 #Scale Factor, might increase this later
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.narrow_planarize = (512, 2644)
        self.wide_planarize = (512, 4840)

        self.planarizeTransform = None
        self.planarizeTransform_scaled = None

        self.planarizeTransform_narrow = None
        self.planarizeTransform_narrow_scaled = None
        self.planarizeTransform_wide = None
        self.planarizeTransform_wide_scaled = None

        
        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*1

        self.InspectionResult_ClipDetection = [None]*10
        self.InspectionResult_Segmentation = [None]*10

        self.InspectionResult_PitchMeasured = [None]*10
        self.InspectionResult_PitchResult = [None]*10
        self.InspectionResult_DetectionID = [None]*10
        self.InspectionResult_Status = [None]*10

        self.DetectionResult_HoleDetection = [None]*10

        self.InspectionImages_prev = [None]*10
        self._test = [0]*10
        self.widget_dir_map = {
            5: "5902A509",
            6: "5902A510",
            7: "658207LE0A",
            8: "5819A107",
        }

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

        self.cap_cam1 = initialize_camera(4)
        self.cap_cam2 = initialize_camera(5)

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
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        for key, value in self.widget_dir_map.items():
            self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(key)
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(key)



        #INIT all variables

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

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_narrow.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_narrow.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_narrow = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_narrow_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_narrow_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_narrow_scaled = np.array(transform_list)
        
        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_wide.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_wide.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_wide_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_wide_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_wide_scaled = np.array(transform_list)


        while self.running:
            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget == 7:
                # self.timerStart = time.time()

                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()

                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()

                #Downsampled the image
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)

                if self.inspection_config.mapCalculated[1] is False: #Just checking the first camera to reduce loop time
                    for i in range(1, 3):
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                            #Also do the map for the scaled image
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                            h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 scaled and map2 scaled value is calculated for scaled image")
                            print(f"Calibration map is calculated for Camera {i} for scaled image")

                            #Not idea but the condition use the bigger image

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time


                    self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                    self.mergeframe1_scaled = cv2.rotate(self.mergeframe1_scaled, cv2.ROTATE_180)
                    self.mergeframe2_scaled = cv2.rotate(self.mergeframe2_scaled, cv2.ROTATE_180)

                    self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                    self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)


                    self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_narrow_scaled, (int(self.narrow_planarize[1]/(self.scale_factor)), int(self.narrow_planarize[0]/(self.scale_factor))))
                    self.combinedImage_scaled = self.downScaledImage(self.combinedImage_scaled, scaleFactor=0.303)

                    self.InspectionResult_PitchMeasured = [None]*10
                    self.InspectionResult_PitchResult = [None]*10

                    if self.combinedImage_scaled is not None:
                        self.part1Cam.emit(self.convertQImage(self.combinedImage_scaled))
        
                    self.P658207LE0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)
  

                    if self.inspection_config.doInspection is True:

                        self.inspection_config.doInspection = False

                        self.emit = self.combinedImage_scaled
                        if self.emit is None:
                            self.emit = np.zeros((337, 1742, 3), dtype=np.uint8)

                        self.emit = self.draw_status_text_PIL(self.emit, "検査中", (50,150,10), size="large", x_offset = -200, y_offset = -100)
                        self.part1Cam.emit(self.convertQImage(self.emit))

                        self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[2], self.inspection_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                        self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)

                        self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                        self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                        self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_narrow, (int(self.narrow_planarize[1]), int(self.narrow_planarize[0])))

                        self.InspectionImages[0] = self.combinedImage

                        for i in range(len(self.InspectionImages)):
                            self.InspectionResult_ClipDetection[i] = self.P658207LE0A_CLIP_Model(source=self.InspectionImages[i], conf=0.7, imgsz=2500, iou=0.7, verbose=False)
                            self.InspectionResult_Segmentation[i] = self.P658207LE0A_SEGMENT_Model(source=self.InspectionImages[i], conf=0.3, imgsz=1280, verbose=False)
                            self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_Status[i] = P658207LE0A_check(self.InspectionImages[i], self.InspectionResult_ClipDetection[i], self.InspectionResult_Segmentation[i])

                            self.save_result_database(self.widget_dir_map[self.inspection_config.widget],
                                  self.inspection_config.today_numofPart[self.inspection_config.widget], self.inspection_config.current_numofPart[self.inspection_config.widget],
                                  self.inspection_config.kensainNumber, self.InspectionResult_PitchMeasured[i], 
                                  self.InspectionResult_PitchResult[i], total_length=0)

                        self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1742, height=337)
                        self.P658207LE0A_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured, self.InspectionResult_PitchResult)


                        # self.InspectionImages_prev[0] = self.InspectionImages[0]
          
                        # self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                        # self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()

                        self.InspectionImages[0] = cv2.cvtColor(self.InspectionImages[0], cv2.COLOR_RGB2BGR)
                        self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))

                        time.sleep(3)


        self.msleep(1)


    #database save method
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
            return (0, 0) 
            
    def get_last_entry_total_numofPart(self, part_name):
        # Get today's date in yyyymmdd format
        today_date = datetime.now().strftime("%Y%m%d")

        print (today_date)

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
            return (0, 0)  # Default values if no entry is found

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
        cv2.imwrite(dir + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image)

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

    def initialize_model(self):
        #Change based on the widget
        P658207LE0A_CLIP_Model = None
        P658207LE0A_SEGMENT_Model = None
        P5902A509_CLIP_Model = None
        P5902A509_SEGMENT_Model = None
        P5819A107_CLIP_Model = None
        P5819A107_SEGMENT_Model = None

        #Detection Model
        path_P658207LE0A_CLIP_Model = "./aikensa/models/P658207LE0A_CLIP.pt"
        path_P658207LE0A_SEGMENT_Model = "./aikensa/models/P658207LE0A_SEGMENT.pt"

        path_P5902A509_CLIP_Model = "./aikensa/models/P5902A509_CLIP.pt"
        path_P5902A509_SEGMENT_Model = "./aikensa/models/P5902A509_SEGMENT.pt"

        #Larger image inference, need SAHI
        path_P5819A107_CLIP_Model = "./aikensa/models/P5819A107_CLIP.pt"
        path_P5819A107_SEGMENT_Model = "./aikensa/models/P5819A107_SEGMENT.pt"

        if os.path.exists(path_P658207LE0A_CLIP_Model):
            P658207LE0A_CLIP_Model = YOLO(path_P658207LE0A_CLIP_Model)

        if os.path.exists(path_P658207LE0A_SEGMENT_Model):
            P658207LE0A_SEGMENT_Model = YOLO(path_P658207LE0A_SEGMENT_Model)

        if os.path.exists(path_P5902A509_CLIP_Model):
            P5902A509_CLIP_Model = YOLO(path_P5902A509_CLIP_Model)

        if os.path.exists(path_P5902A509_SEGMENT_Model):
            P5902A509_SEGMENT_Model = YOLO(path_P5902A509_SEGMENT_Model)
        
        if os.path.exists(path_P5819A107_CLIP_Model):
            P5819A107_CLIP_Model = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                            model_path=path_P5819A107_CLIP_Model,
                                                                            confidence_threshold=0.9,
                                                                            device="cuda:0")
        if os.path.exists(path_P5819A107_SEGMENT_Model):
            P5902A509_SEGMENT_Model = YOLO(path_P5902A509_SEGMENT_Model)
            #Segmentation is not supported by SAHI yet


        self.P658207LE0A_CLIP_Model = P658207LE0A_CLIP_Model
        self.P658207LE0A_SEGMENT_Model = P658207LE0A_SEGMENT_Model

        self.P5902A509_CLIP_Model = P5902A509_CLIP_Model
        self.P5902A509_SEGMENT_Model = P5902A509_SEGMENT_Model

        self.P5819A107_CLIP_Model = P5819A107_CLIP_Model
        self.P5819A107_SEGMENT_Model = P5819A107_SEGMENT_Model

        if self.P658207LE0A_CLIP_Model is not None:
            print("P658207LE0A_CLIP_Model loaded")
        if self.P658207LE0A_SEGMENT_Model is not None:
            print("P658207LE0A_SEGMENT_Model loaded")
        if self.P5902A509_CLIP_Model is not None:
            print("P5902A509_CLIP_Model loaded")
        if self.P5902A509_SEGMENT_Model is not None:
            print("P5902A509_SEGMENT_Model loaded")
        if self.P5819A107_CLIP_Model is not None:
            print("P5819A107_CLIP_Model loaded")
        if self.P5819A107_SEGMENT_Model is not None:
            print("P5819A107_SEGMENT_Model loaded")
        

    def stop(self):
        self.inspection_config.widget = -1
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        print("Inspection thread stopped.")