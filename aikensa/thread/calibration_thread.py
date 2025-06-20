import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, detectCharucoBoard_scaledImage, calculatecameramatrix, calculatecameramatrix_scaledImage, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.scripts.scripts_img_processing import resize_image
from aikensa.opencv_imgprocessing.arucoplanarize import planarize_image, planarize_image_4567
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CalibrationConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected
    
    calculateSingeFrameMatrix: bool = False
    calculateCamMatrix: bool = False
    delCamMatrix: bool = False
    savecalculatedCamImage: bool = False

    calibrationMatrix: np.ndarray = field(default=None)
    mapCalculated: list = field(default_factory=lambda: [False]*10)     #max for 10 cameras
    map1: list = field(default_factory=lambda: [None]*10)               #max for 10 cameras
    map2: list = field(default_factory=lambda: [None]*10)               #max for 10 cameras

    calibrationMatrix_scaled: np.ndarray = field(default=None)
    map1_downscaled: list = field(default_factory=lambda: [None]*10)    #max for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*10)    #max for 10 cameras

    calculateHomo_cam1: bool = False
    calculateHomo_cam2: bool = False
    calculateHomo_cam3: bool = False
    calculateHomo_cam4: bool = False


    
    deleteHomo: bool = False

    mergeCam: bool = False
    saveImage: bool = False

    savePlanarize_left: bool = False
    savePlanarize_right: bool = False

    delPlanarize: bool = False

class CalibrationThread(QThread):

    CalibCamStream = pyqtSignal(QImage)
    CamMerge1 = pyqtSignal(QImage)
    CamMerge2 = pyqtSignal(QImage)
    CamMerge3 = pyqtSignal(QImage)
    CamMerge4 = pyqtSignal(QImage)
    CamMergeAll = pyqtSignal(QImage)

    def __init__(self, calib_config: CalibrationConfig = None):
        super(CalibrationThread, self).__init__()
        self.running = True
        self.frames = {}  # logical_id → np.ndarray
        
        if calib_config is None:
            self.calib_config = CalibrationConfig()    
        else:
            self.calib_config = calib_config

        self.cameraMatrix = None
        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"
        self.cap_cam = None
        self.frame = None
        self.frame_resized = None

        self.multiCam_stream = False
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None
        self.cap_cam4 = None

        self.mergeframe1 = None
        self.mergeframe2 = None

        self.mergeframe1_resized = None
        self.mergeframe2_resized = None

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        self.homography_matrix3 = None
        self.homography_matrix4 = None

        self.H1 = None
        self.H2 = None
        self.H3 = None
        self.H4 = None

        self.homography_size = None
        self.homography_blank_canvas = None

        self.planarize = (1300, 3500)

        self.combinedImage = None
        self.combinedImage_left = None
        self.combinedImage_right = None
        self.combinedImage_left_resized = None
        self.combinedImage_right_resized = None

        self.scale_factor = 5.0
        
        self.frame_width = 3072
        self.frame_height = 2048

        self.planarizeTransform_left = None
        self.planarizeTransform_right = None

        this_dir = os.path.dirname(__file__)
        cam_config_path = os.path.abspath(os.path.join(this_dir, '..', 'config'))
        self.cam_config_file = os.path.join(cam_config_path, 'camera_config.yaml')
        
        with open(self.cam_config_file, 'r') as file:
            self.cam_map = yaml.safe_load(file)

        self.widget_to_cam_map = {1: 0, 2: 1, 4: 2, 5: 3}  

    def initialize_single_camera(self, camID):

        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.calib_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
        else:
            print(f"Requested Camera ID: {camID}")
            actual_camID = self.cam_map.get(camID, -1)
            print("Actual Camera ID from map:", actual_camID)
            print(f"Initialized Camera on ID {actual_camID}")
            self.cap_cam = initialize_camera(actual_camID)

    def release_all_camera(self):
        if self.cap_cam is not None:
            self.cap_cam.release()
            print(f"Camera {self.calib_config.cameraID} released.")
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

    def initialize_all_left_camera(self):
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

    def initialize_all_right_camera(self):
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")

        actual_camID = self.cam_map.get(2, -1)
        self.cap_cam3 = initialize_camera(actual_camID)

        actual_camID = self.cam_map.get(3, -1)
        self.cap_cam4 = initialize_camera(actual_camID)

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

        #print thread started
        print("Calibration Thread Started")

        self.current_cameraID = self.calib_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        #INIT all variables
        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_left.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_left.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_left = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_right.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_right.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_right = np.array(transform_list)

        while self.running:

            if self.calib_config.widget == 0:
                self.calib_config.cameraID = -1

            self.calib_config.cameraID = self.widget_to_cam_map.get(self.calib_config.widget, -1)

            if self.calib_config.widget in [1, 2, 4, 5]:
                if self.calib_config.cameraID != self.current_cameraID:
                    # Camera ID has changed, reinitialize the camera
                    if self.current_cameraID != -1:
                        self.cap_cam.release()
                        print(f"Camera {self.current_cameraID} released.")
                    self.current_cameraID = self.calib_config.cameraID

                    if self.calib_config.widget == 1:
                        self.initialize_single_camera(0)
                        print("Initializing Camera 0")
                    if self.calib_config.widget == 2:
                        self.initialize_single_camera(1)
                        print("Initializing Camera 1")
                    if self.calib_config.widget == 4:
                        self.initialize_single_camera(2)
                        print("Initializing Camera 2")
                    if self.calib_config.widget == 5:
                        self.initialize_single_camera(3)
                        print("Initializing Camera 3")
                  
                if self.cap_cam is not None:
                    try:
                        ret, self.frame = self.cap_cam.read()
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

                        if not ret:
                            print("Failed to capture frame")
                            continue
                    except cv2.error as e:
                        print("An error occurred while reading frames from the cameras:", str(e))


                if self.calib_config.calculateSingeFrameMatrix:
                    self.frame, _, _ = detectCharucoBoard(self.frame)
                    self.calib_config.calculateSingeFrameMatrix = False

                if self.calib_config.calculateCamMatrix:
                    self.calib_config.calibrationMatrix = calculatecameramatrix()

                    print(f"Calibration Matrix Value: {self.calib_config.calibrationMatrix}")
                    
                    os.makedirs(self._save_dir, exist_ok=True)
                    self.save_calibration_to_yaml(self.calib_config.calibrationMatrix, self._save_dir + f"Calibration_camera_{self.calib_config.cameraID}.yaml")
                    self.calib_config.calculateCamMatrix = False

                if self.frame is not None:
                    self.frame_resized = resize_image(self.frame.copy(), width=1024, height=683)
                    self.CalibCamStream.emit(self.convertQImage(self.frame_resized))
                    # print("Frame emitted to CalibCamStream")
            
            if self.calib_config.widget == 3:
                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_left_camera()
                    
                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()

                #Calculate all map from calibration matrix for 5 cameras, thus i in range(1, 6)
                for i in range(0, 2):
                    if self.calib_config.mapCalculated[i] is False:
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.calib_config.map1[i], self.calib_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.calib_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                self.mergeframe1 = cv2.cvtColor(self.mergeframe1, cv2.COLOR_BGR2RGB)
                self.mergeframe2 = cv2.cvtColor(self.mergeframe2, cv2.COLOR_BGR2RGB)

                self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)


                if all(self.calib_config.mapCalculated[i] for i in range(0, 2)):
                    # print("All calibration maps are calculated.")
                    self.mergeframe1 = cv2.remap(self.mergeframe1, self.calib_config.map1[0], self.calib_config.map2[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2 = cv2.remap(self.mergeframe2, self.calib_config.map1[1], self.calib_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                #Calculate Homography matrix
                if self.calib_config.calculateHomo_cam1 is True:

                    self.calib_config.calculateHomo_cam1 = False
                    if self.mergeframe1 is not None:
                        cv2.imwrite("mergeframe1.png", self.mergeframe1)
                        _, self.homography_matrix1 = calculateHomography_template(self.homography_template, self.mergeframe1)
                        #save _
                        cv2.imwrite("resultmergeframe1.png", _)
                        self.H1 = np.array(self.homography_matrix1)
                        print(f"Homography matrix is calculated for Camera 1 with value {self.homography_matrix1}")
                        os.makedirs(self._save_dir, exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam1.yaml", "w") as file:
                            yaml.dump(self.homography_matrix1.tolist(), file)
                    else:
                        ("mergeframe1 is empty")

                if self.calib_config.calculateHomo_cam2 is True:

                    self.calib_config.calculateHomo_cam2 = False
                    if self.mergeframe2 is not None:
                            
                        _, self.homography_matrix2 = calculateHomography_template(self.homography_template, self.mergeframe2)
                        cv2.imwrite("resultmergeframe2.png", _)
                        self.H2 = np.array(self.homography_matrix2)
                        print(f"Homography matrix is calculated for Camera 2 with value {self.homography_matrix2}")
                        os.makedirs(self._save_dir, exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam2.yaml", "w") as file:
                            yaml.dump(self.homography_matrix2.tolist(), file)
                    else:
                        print("mergeframe2 is empty")

                if self.H1 is None:
                    # print("H1 is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                            print("Loading H1 from file")
                            self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H1 = np.array(self.homography_matrix1)

                if self.H2 is None:
                    # print("H2 is None")  
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                            print("Loading H2 from file")
                            self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H2 = np.array(self.homography_matrix2)

                if self.H1 is not None and self.H2 is not None:
                    print("Both H1 and H2 are not None, warping images")
                    self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                    self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                else:
                    self.combinedImage = self.homography_blank_canvas

                if self.calib_config.savePlanarize_left is True:
                    self.calib_config.savePlanarize = False
                    print("Saving planarize")

                    cv2.imwrite("beforePlanarized.png", self.combinedImage)
                    self.combinedImage_left, self.planarizeTransform_left = planarize_image(self.combinedImage, 
                                                                                  target_width=self.planarize[1], target_height=self.planarize[0], 
                                                                                  top_offset=0, bottom_offset=0)
                    cv2.imwrite("afterPlanarized.png", self.combinedImage_left)
                    
                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/planarizeTransform_left.yaml", "w") as file:
                        yaml.dump(self.planarizeTransform_left.tolist(), file)

                    
                if self.planarizeTransform_left is not None:
                    self.combinedImage_left = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_left, (self.planarize[1],self.planarize[0]))

                if self.mergeframe1 is not None:
                    self.mergeframe1_resized = resize_image(self.mergeframe1, 246, 163)
                    # print("Emitting mergeframe1 resized")
                    self.CamMerge1.emit(self.convertQImage(self.mergeframe1_resized))
                if self.mergeframe2 is not None:
                    self.mergeframe2_resized = resize_image(self.mergeframe2, 246, 163)
                    # print("Emitting mergeframe2 resized")
                    self.CamMerge2.emit(self.convertQImage(self.mergeframe2_resized))
                if self.combinedImage is not None:
                    self.combinedImage_resized = resize_image(self.combinedImage, 1167, 433)
                    # print("Emitting combinedImage resized")
                    self.CamMergeAll.emit(self.convertQImage(self.combinedImage_resized))
                if self.combinedImage_left is not None:
                    self.combinedImage_left_resized = resize_image(self.combinedImage_left, 1167, 433)
                    # print("Emitting combinedImage_left resized")
                    self.CamMergeAll.emit(self.convertQImage(self.combinedImage_left_resized))

            if self.calib_config.widget == 6:
                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_right_camera()  

                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()

                #Calculate all map from calibration matrix for 5 cameras, thus i in range(1, 6)
                for i in range(2, 4):
                    if self.calib_config.mapCalculated[i] is False:
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe3.shape[:2] #use mergeframe3 as reference
                            self.calib_config.map1[i], self.calib_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.calib_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                self.mergeframe3 = cv2.cvtColor(self.mergeframe3, cv2.COLOR_BGR2RGB)
                self.mergeframe4 = cv2.cvtColor(self.mergeframe4, cv2.COLOR_BGR2RGB)

                self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)

                if all(self.calib_config.mapCalculated[i] for i in range(2, 4)):
                    # print("All calibration maps are calculated.")
                    self.mergeframe3 = cv2.remap(self.mergeframe3, self.calib_config.map1[2], self.calib_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe4 = cv2.remap(self.mergeframe4, self.calib_config.map1[3], self.calib_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                #Calculate Homography matrix
                if self.calib_config.calculateHomo_cam3 is True:

                    self.calib_config.calculateHomo_cam3 = False
                    if self.mergeframe3 is not None:
                        cv2.imwrite("mergeframe3.png", self.mergeframe3)
                        _, self.homography_matrix3 = calculateHomography_template(self.homography_template, self.mergeframe3)
                        #save _
                        cv2.imwrite("resultmergeframe3.png", _)
                        self.H3 = np.array(self.homography_matrix3)
                        print(f"Homography matrix is calculated for Camera 3 with value {self.homography_matrix3}")
                        os.makedirs(self._save_dir, exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam3.yaml", "w") as file:
                            yaml.dump(self.homography_matrix3.tolist(), file)
                    else:
                        print("mergeframe3 is empty")

                if self.calib_config.calculateHomo_cam4 is True:

                    self.calib_config.calculateHomo_cam4 = False
                    if self.mergeframe4 is not None:

                        _, self.homography_matrix4 = calculateHomography_template(self.homography_template, self.mergeframe4)
                        cv2.imwrite("resultmergeframe4.png", _)
                        self.H4 = np.array(self.homography_matrix4)
                        print(f"Homography matrix is calculated for Camera 4 with value {self.homography_matrix4}")
                        os.makedirs(self._save_dir, exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam4.yaml", "w") as file:
                            yaml.dump(self.homography_matrix4.tolist(), file)
                    else:
                        print("mergeframe4 is empty")

                if self.H3 is None:
                    # print("H1 is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                            print("Loading H3 from file")
                            self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H3 = np.array(self.homography_matrix3)

                if self.H4 is None:
                    # print("H2 is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                            print("Loading H4 from file")
                            self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H4 = np.array(self.homography_matrix4)

                if self.H3 is not None and self.H4 is not None:
                    print("Both H3 and H4 are not None, warping images")
                    self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe3, self.H3)
                    self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe4, self.H4)
                else:
                    self.combinedImage = self.homography_blank_canvas

                if self.calib_config.savePlanarize_right is True:
                    self.calib_config.savePlanarize_right = False
                    print("Saving planarize")

                    cv2.imwrite("beforePlanarized.png", self.combinedImage)
                    self.combinedImage_right, self.planarizeTransform_right = planarize_image_4567(self.combinedImage, 
                                                                                  target_width=self.planarize[1], target_height=self.planarize[0], 
                                                                                  top_offset=0, bottom_offset=0)
                    cv2.imwrite("afterPlanarized.png", self.combinedImage_right)

                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/planarizeTransform_right.yaml", "w") as file:
                        yaml.dump(self.planarizeTransform_right.tolist(), file)

                if self.planarizeTransform_right is not None:
                    self.combinedImage_right = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_right, (self.planarize[1],self.planarize[0]))

                if self.mergeframe3 is not None:
                    self.mergeframe3_resized = resize_image(self.mergeframe3, 246, 163)
                    # print("Emitting mergeframe3 resized")
                    self.CamMerge1.emit(self.convertQImage(self.mergeframe3_resized))
                if self.mergeframe4 is not None:
                    self.mergeframe4_resized = resize_image(self.mergeframe4, 246, 163)
                    # print("Emitting mergeframe4 resized")
                    self.CamMerge2.emit(self.convertQImage(self.mergeframe4_resized))
                if self.combinedImage is not None:
                    self.combinedImage_resized = resize_image(self.combinedImage, 1167, 433)
                    # print("Emitting combinedImage resized")
                    self.CamMergeAll.emit(self.convertQImage(self.combinedImage_resized))
                if self.combinedImage_right is not None:
                    self.combinedImage_right_resized = resize_image(self.combinedImage_right, 1167, 433)
                    # print("Emitting combinedImage_right resized")
                    self.CamMergeAll.emit(self.convertQImage(self.combinedImage_right_resized))

            self.msleep(10)
            
        print(f"Camera {self.calib_config.cameraID} released.")

    def create_placeholder_image(self):
        # Create a small black image with a white dot in the center
        size = 100
        placeholder = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(placeholder, (size // 2, size // 2), 10, (255, 255, 255), -1)
        return placeholder

    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def undistortFrame(self, frame,cameraMatrix, distortionCoeff):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.undistort(frame, cameraMatrix, distortionCoeff, None, cameraMatrix)
        return frame

    def stop(self):
        self.running = False
        self.release_all_camera()
        print("Calibration thread stopped.")
    
    def save_calibration_to_yaml(self, calibrationMatrix, filename):
        with open(filename, 'w') as file:
            yaml.dump(calibrationMatrix, file)

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff
        
    def initialize_maps(camera_matrix, dist_coeffs, image_size):
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, image_size, cv2.CV_16SC2)
        return map1, map2