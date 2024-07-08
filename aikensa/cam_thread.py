import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time
import csv

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix, calculateHomography, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize

from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_fa_sound, play_sol_sound, play_la_sound, play_si_sound, play_alarm_sound

from ultralytics import YOLO
from aikensa.parts_config.ctrplr_8283XW0W0P import partcheck as ctrplrCheck

from PIL import ImageFont, ImageDraw, Image

@dataclass
class CameraConfig:
    widget: int = 0
    
    calculateCamMatrix1: bool = False
    calculateCamMatrix2: bool = False
    captureCam1: bool = False
    captureCam2: bool = False
    captureClip1: bool = False
    captureClip2: bool = False
    captureClip3: bool = False
    delCamMatrix1: bool = False
    delCamMatrix2: bool = False
    checkUndistort1: bool = False
    checkUndistort2: bool = False
    calculateHomo: bool = False

    calculateHomo_cam1: bool = False
    calculateHomo_cam2: bool = False

    deleteHomo: bool = False

    mergeCam: bool = False
    saveImage: bool = False

    savePlanarize: bool = False
    delPlanarize: bool = False

    opacity: float = 0.5
    blur: int = 10
    lower_canny: int = 100
    upper_canny: int = 200
    contrast: int = 200
    brightness: int = 0
    savecannyparams: bool = False

    #General Functions
    furyou_plus: bool = False
    furyou_minus: bool = False
    kansei_plus: bool = False
    kansei_minus: bool = False

    kensainName: str = None

    HDRes: bool = False
    triggerKensa: bool = False
    kensaReset: bool = False

    ctrplrpitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    ctrplrWorkOrder : List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    ctrplrLHpitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    ctrplrLHnumofPart: Tuple[int, int] = (0, 0)
    resetCounter: bool = False

    ctrplrRHpitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    ctrplrRHnumofPart: Tuple[int, int] = (0, 0)
    ctrplrRH_resetCounter: bool = False
    



class CameraThread(QThread):

    camFrame1 = pyqtSignal(QImage)
    camFrame2 = pyqtSignal(QImage)
    mergeFrame = pyqtSignal(QImage)
    kata1Frame = pyqtSignal(QImage)
    kata2Frame = pyqtSignal(QImage)
    clip1Frame = pyqtSignal(QImage)
    clip2Frame = pyqtSignal(QImage)
    clip3Frame = pyqtSignal(QImage)

    handFrame1 = pyqtSignal(int)
    handFrame2 = pyqtSignal(int)
    handFrame3 = pyqtSignal(int)

    ctrplrworkorderSignal = pyqtSignal(list)

    ctrplrLH_pitch_updated = pyqtSignal(list)
    ctrplrRH_pitch_updated = pyqtSignal(list)

    ctrplrLH_numofPart_updated = pyqtSignal(tuple)
    ctrplrRH_numofPart_updated = pyqtSignal(tuple)

    def __init__(self, cam_config: CameraConfig = None):
        super(CameraThread, self).__init__()
        self.running = True
        self.charucoTimer = None
        self.kensatimer = None

        if cam_config is None:
            self.cam_config = CameraConfig()    
        else:
            self.cam_config = cam_config

        self.widget_dir_map={
            3: "82833W050P",
            4: "82833W040P"
        }

        self.previous_HDRes = self.cam_config.HDRes
        self.scale_factor = 5.0

        self.handClassificationModel = None

        self.clipHandWaitTime = 0.8 
        self.inspection_delay = 3.0

        self.handinFrame1 = False
        self.handinFrame2 = False
        self.handinFrame3 = False

        self.result_handframe1 = None
        self.result_handframe2 = None
        self.result_handframe3 = None

        self.result_clip = None

        self.handinFrame1Timer = None
        self.handinFrame2Timer = None
        self.handinFrame3Timer = None
        self.oneLoop = False

        self.cameraMatrix1 = None
        self.distortionCoeff1 = None
        self.cameraMatrix2 = None
        self.distortionCoeff2 = None
        self.H = None

        self.clip_detection = None
        self.marking_detection = None

        self.kensa_cycle = False
        self.kensa_order = []

        self.musicPlay = False
        self.handinFrameTimer = None

        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.last_inspection_time = 0
        self.prev_timestamp = None

    def run(self):

        cap_cam1 = initialize_camera(2)
        print(f"Initiliazing Camera 1.... Located on {cap_cam1}")
        cap_cam2 = initialize_camera(0)
        print(f"Initiliazing Camera 2.... Located on {cap_cam2}")

        #Read the yaml param once
        if os.path.exists("./aikensa/cameracalibration/cam1calibration_param.yaml"):
            with open("./aikensa/cameracalibration/cam1calibration_param.yaml") as file:
                cam1calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                cameraMatrix1 = np.array(cam1calibration_param.get('camera_matrix'))
                distortionCoeff1 = np.array(cam1calibration_param.get('distortion_coefficients'))

        if os.path.exists("./aikensa/cameracalibration/cam2calibration_param.yaml"):
            with open("./aikensa/cameracalibration/cam2calibration_param.yaml") as file:
                cam2calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                cameraMatrix2 = np.array(cam2calibration_param.get('camera_matrix'))
                distortionCoeff2 = np.array(cam2calibration_param.get('distortion_coefficients'))

        # if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
        #     with open("./aikensa/cameracalibration/homography_param.yaml") as file:
        #         homography_param = yaml.load(file, Loader=yaml.FullLoader)
        #         H = np.array(homography_param)

        # if os.path.exists("./aikensa/cameracalibration/homography_param_lowres.yaml"):
        #     with open("./aikensa/cameracalibration/homography_param_lowres.yaml") as file:
        #         homography_param_lowres = yaml.load(file, Loader=yaml.FullLoader)
        #         H_lowres = np.array(homography_param_lowres)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                homography_param1 = yaml.load(file, Loader=yaml.FullLoader)
                H1 = np.array(homography_param1)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                homography_param2 = yaml.load(file, Loader=yaml.FullLoader)
                H2 = np.array(homography_param2)


        if os.path.exists("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml") and os.path.exists("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml") as file:
                homography_param_lowres_cam1 = yaml.load(file, Loader=yaml.FullLoader)
                H1_lowres = np.array(homography_param_lowres_cam1)

            with open("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml") as file:
                homography_param_lowres_cam2 = yaml.load(file, Loader=yaml.FullLoader)
                H2_lowres = np.array(homography_param_lowres_cam2)

        self.cameraMatrix1 = self.adjust_camera_matrix(cameraMatrix1, self.scale_factor)
        self.cameraMatrix2 = self.adjust_camera_matrix(cameraMatrix2, self.scale_factor)
        self.distortionCoeff1 = distortionCoeff1
        self.distortionCoeff2 = distortionCoeff2
        # self.H = self.adjust_transform_matrix(H, self.scale_factor)

        self.flexibleH1 = H1_lowres
        self.flexibleH2 = H2_lowres

        self.H1 = H1
        self.H2 = H2

        self.H1_lowres = H1_lowres
        self.H2_lowres = H2_lowres

        homography_template = cv2.imread("./aikensa/homography_template/charuco_template.png")
        #print read image size
        homography_size = (homography_template.shape[0], homography_template.shape[1])

        #make dark blank image with same size as homography_template
        homography_blank_canvas = np.zeros(homography_size, dtype=np.uint8)
        homography_blank_canvas = cv2.cvtColor(homography_blank_canvas, cv2.COLOR_GRAY2RGB)

        while self.running is True:
            current_time = time.time()

            try:
                ret1, frame1 = cap_cam1.read()
                ret2, frame2 = cap_cam2.read()
                
            except cv2.error as e:
                print("An error occurred while reading frames from the cameras:", str(e))

            if self.cam_config.widget == 1:

                if frame1 is None:
                    frame1 = np.zeros((2048, 3072, 3), dtype=np.uint8)
                if frame2 is None:
                    frame2 = np.zeros((2048, 3072, 3), dtype=np.uint8) 

                else:
                    frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
                    frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
                    
                    if os.path.exists("./aikensa/cameracalibration/cam1calibration_param.yaml"):
                        with open("./aikensa/cameracalibration/cam1calibration_param.yaml") as file:
                            cam1calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                            cameraMatrix1 = np.array(cam1calibration_param.get('camera_matrix'))
                            distortionCoeff1 = np.array(cam1calibration_param.get('distortion_coefficients'))

                        frame1raw = frame1.copy()
                        frame1 = cv2.undistort(frame1, cameraMatrix1, distortionCoeff1, None, cameraMatrix1)

                        if self.cam_config.checkUndistort1 == True:
                            cv2.imwrite("camu1ndistorted.jpg", frame1)
                            cv2.imwrite("cam1raw.jpg", frame1raw)
                            self.cam_config.checkUndistort1 = False

                    if os.path.exists("./aikensa/cameracalibration/cam2calibration_param.yaml"):
                        with open("./aikensa/cameracalibration/cam2calibration_param.yaml") as file:
                            cam2calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                            cameraMatrix2 = np.array(cam2calibration_param.get('camera_matrix'))
                            distortionCoeff2 = np.array(cam2calibration_param.get('distortion_coefficients'))

                            frame2raw = frame2.copy()
                            frame2 = cv2.undistort(frame2, cameraMatrix2, distortionCoeff2, None, cameraMatrix2)
                            if self.cam_config.checkUndistort2 == True:
                                cv2.imwrite("camu2ndistorted.jpg", frame2)
                                cv2.imwrite("cam2raw.jpg", frame2raw)
                                self.cam_config.checkUndistort2 = False

                    if self.cam_config.delCamMatrix1 == True:
                        if os.path.exists("./aikensa/cameracalibration/cam1calibration_param.yaml"):
                            os.remove("./aikensa/cameracalibration/cam1calibration_param.yaml")
                        self.cam_config.delCamMatrix1 = False

                    if self.cam_config.delCamMatrix2 == True:
                        if os.path.exists("./aikensa/cameracalibration/cam2calibration_param.yaml"):
                            os.remove("./aikensa/cameracalibration/cam2calibration_param.yaml")
                        self.cam_config.delCamMatrix2 = False
                    
                    if ret1 and ret2:
                        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                        image1 = self.downSampling(frame1)
                        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                        image2 = self.downSampling(frame2)

                    if self.cam_config.captureCam1 == True:
                        frame1, _, _ = detectCharucoBoard(frame1)

                        arucoFrame1 = frame1.copy()
                        self.charucoTimer = current_time

                        self.cam_config.captureCam1 = False

                        if self.charucoTimer and current_time - self.charucoTimer < 1:
                            image1 = self.downSampling(arucoFrame1)
                        elif self.charucoTimer and current_time - self.charucoTimer >= 1.2:
                            self.charucoTimer = None
                    
                    if self.cam_config.captureCam2 == True:
                        frame2, _, _ = detectCharucoBoard(frame2)

                        arucoFrame2 = frame2.copy()
                        self.charucoTimer = current_time

                        self.cam_config.captureCam2 = False

                        if self.charucoTimer and current_time - self.charucoTimer > 1:
                            image2 = self.downSampling(arucoFrame2)
                        else:
                            self.charucoTimer = None

                    if self.cam_config.calculateCamMatrix1 == True:
                        calibration_matrix = calculatecameramatrix()
                        if not os.path.exists("./aikensa/cameracalibration"):
                            os.makedirs("./aikensa/cameracalibration")
                        with open("./aikensa/cameracalibration/cam1calibration_param.yaml", "w") as file:
                            yaml.dump(calibration_matrix, file)


                        print("Camera matrix 1 calculated.")
                        self.cam_config.calculateCamMatrix1 = False

                    if self.cam_config.calculateCamMatrix2 == True:
                        calibration_matrix = calculatecameramatrix()
                        if not os.path.exists("./aikensa/cameracalibration"):
                            os.makedirs("./aikensa/cameracalibration")
                        with open("./aikensa/cameracalibration/cam2calibration_param.yaml", "w") as file:
                            yaml.dump(calibration_matrix, file)


                        print("Camera matrix 2 calculated.")
                        
                        self.cam_config.calculateCamMatrix2 = False

                    if self.cam_config.calculateHomo == True:
                        combineImage, homographyMatrix = calculateHomography(frame1, frame2)
                        combineImage_lowres, homographyMatrix_lowres = calculateHomography(self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)),self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)))

                        os.makedirs("./aikensa/cameracalibration", exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param.yaml", "w") as file:
                            yaml.dump(homographyMatrix.tolist(), file)
                        with open("./aikensa/cameracalibration/homography_param_lowres.yaml", "w") as file:
                            yaml.dump(homographyMatrix_lowres.tolist(), file)
                        
                        self.cam_config.calculateHomo = False


                    if self.cam_config.calculateHomo_cam1 == True:
                        _, homographyMatrix1 = calculateHomography_template(homography_template, frame1)
                        _, homographyMatrix1_lowres = calculateHomography_template(self.resizeImage(homography_template, int(homography_template.shape[1]//self.scale_factor), int(homography_template.shape[0]//self.scale_factor)),
                                                                                   self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)))

                        print(f"Homography Matrix 1: {homographyMatrix1}")
                        print(f"Homography Matrix 1 Lowres: {homographyMatrix1_lowres}")

                        os.makedirs("./aikensa/cameracalibration", exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam1.yaml", "w") as file:
                            yaml.dump(homographyMatrix1.tolist(), file)
                        with open("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml", "w") as file:
                            yaml.dump(homographyMatrix1_lowres.tolist(), file)

                        self.cam_config.calculateHomo_cam1 = False

                    if self.cam_config.calculateHomo_cam2 == True:
                        _, homographyMatrix2 = calculateHomography_template(homography_template, frame2)
                        _, homographyMatrix2_lowres = calculateHomography_template(self.resizeImage(homography_template, int(homography_template.shape[1]//self.scale_factor), int(homography_template.shape[0]//self.scale_factor)),
                                                                                   self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)))

                        print(f"Homography Matrix 2: {homographyMatrix2}")
                        print(f"Homography Matrix 2 Lowres: {homographyMatrix2_lowres}")

                        os.makedirs("./aikensa/cameracalibration", exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam2.yaml", "w") as file:
                            yaml.dump(homographyMatrix2.tolist(), file)
                        with open("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml", "w") as file:
                            yaml.dump(homographyMatrix2_lowres.tolist(), file)

                        self.cam_config.calculateHomo_cam2 = False

                    

                    if self.cam_config.deleteHomo == True:
                        if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
                            os.remove("./aikensa/cameracalibration/homography_param.yaml")
                        if os.path.exists("./aikensa/cameracalibration/homography_param_lowres.yaml"):
                            os.remove("./aikensa/cameracalibration/homography_param_lowres.yaml")
                        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
                            os.remove("./aikensa/cameracalibration/homography_param_cam1.yaml")
                        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
                            os.remove("./aikensa/cameracalibration/homography_param_cam2.yaml")
                        self.cam_config.deleteHomo = False



                    # #blank combinedImage (placeholder)
                    # combinedImage = homography_blank_canvas.copy()

                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                            homography_param1 = yaml.load(file, Loader=yaml.FullLoader)
                            H1 = np.array(homography_param1)
                            combinedImage = warpTwoImages_template(homography_blank_canvas, frame1, H1)

                            # cv2.imwrite("combinedImage_cam1.jpg", combinedImage)

                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                            homography_param2 = yaml.load(file, Loader=yaml.FullLoader)
                            H2 = np.array(homography_param2)
                            combinedImage = warpTwoImages_template(combinedImage, frame2, H2)

                            # cv2.imwrite("combinedImage_cam2.jpg", combinedImage)

                    
                    if os.path.exists("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml") and os.path.exists("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_lowres_cam1.yaml") as file:
                            homography_param_lowres_cam1 = yaml.load(file, Loader=yaml.FullLoader)
                            H1_lowres = np.array(homography_param_lowres_cam1)
                            combinedImage_lowres = warpTwoImages_template(self.resizeImage(homography_blank_canvas, int(homography_blank_canvas.shape[1]//self.scale_factor), int(homography_blank_canvas.shape[0]//self.scale_factor)),
                                                                                   self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)), 
                                                                                   H1_lowres)
                            
                            # cv2.imwrite("combinedImage_lowres_cam1.jpg", combinedImage_lowres)

                        with open("./aikensa/cameracalibration/homography_param_lowres_cam2.yaml") as file:
                            homography_param_lowres_cam2 = yaml.load(file, Loader=yaml.FullLoader)
                            H2_lowres = np.array(homography_param_lowres_cam2)
                            combinedImage_lowres = warpTwoImages_template(combinedImage_lowres, 
                                                                          self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)), H2_lowres)

                            # cv2.imwrite("combinedImage_lowres_cam2.jpg", combinedImage_lowres)

                    combinedImage, _ = planarize(combinedImage, scale_factor=1.0)
                    combinedImage_lowres, _t = planarize(combinedImage_lowres, self.scale_factor)

                    # print("warp Transform;", _)
                    # print("warp Transform Lowres;", _t)

                    # cv2.imwrite("combinedImage.jpg", combinedImage)
                    # cv2.imwrite("combinedImage_lowres.jpg", combinedImage_lowres)

                    if self.cam_config.savePlanarize == True:
                        os.makedirs("./aikensa/param", exist_ok=True)
                        with open('./aikensa/param/warptransform.yaml', 'w') as file:
                            yaml.dump(_.tolist(), file)
                        with open('./aikensa/param/warptransform_lowres.yaml', 'w') as file:
                            yaml.dump(_t.tolist(), file)
                        self.cam_config.savePlanarize = False

                    if self.cam_config.delPlanarize == True:
                        if os.path.exists("./aikensa/param/warptransform.yaml"):
                            os.remove("./aikensa/param/warptransform.yaml")
                        if os.path.exists("./aikensa/param/warptransform_lowres.yaml"):
                            os.remove("./aikensa/param/warptransform_lowres.yaml")
                        self.cam_config.delPlanarize = False

                    if self.cam_config.saveImage == True:
                        os.makedirs("./aikensa/capturedimages", exist_ok=True)
                        os.makedirs("./aikensa/capturedimages/combinedImage", exist_ok=True)
                        os.makedirs("./aikensa/capturedimages/croppedFrame1", exist_ok=True)
                        os.makedirs("./aikensa/capturedimages/croppedFrame2", exist_ok=True)

                        combinedImage_dump = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f"./aikensa/capturedimages/combinedImage/capturedimage_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", combinedImage_dump)

                        croppedFrame1_dump = cv2.cvtColor(croppedFrame1, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f"./aikensa/capturedimages/croppedFrame1/croppedFrame1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", croppedFrame1_dump)

                        croppedFrame2_dump = cv2.cvtColor(croppedFrame2, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f"./aikensa/capturedimages/croppedFrame2/croppedFrame2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", croppedFrame2_dump)
                        print("Images saved.")
                        self.cam_config.saveImage = False


                    clipFrame1 = self.frameCrop(frame1, x=590, y=0, w=600, h=600, wout = 128, hout = 128)
                    clipFrame2 = self.frameCrop(frame1, x=1900, y=0, w=600, h=600, wout = 128, hout = 128)
                    clipFrame3 = self.frameCrop(frame2, x=600, y=0, w=600, h=600, wout = 128, hout = 128)

                    if self.cam_config.captureClip1:
                        clipFrame1_dump = cv2.cvtColor(clipFrame1, cv2.COLOR_BGR2RGB)
                        os.makedirs("./aikensa/capturedimages/clip1", exist_ok=True)
                        cv2.imwrite(f"./aikensa/capturedimages/clip1/clip1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", clipFrame1_dump)
                        self.cam_config.captureClip1 = False
                        print("Clip 1 captured.")
                    if self.cam_config.captureClip2:
                        clipFrame2_dump = cv2.cvtColor(clipFrame2, cv2.COLOR_BGR2RGB)
                        os.makedirs("./aikensa/capturedimages/clip2", exist_ok=True)
                        cv2.imwrite(f"./aikensa/capturedimages/clip2/clip2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", clipFrame2_dump)
                        self.cam_config.captureClip2 = False
                        print("Clip 2 captured.")
                    if self.cam_config.captureClip3:
                        clipFrame3_dump = cv2.cvtColor(clipFrame3, cv2.COLOR_BGR2RGB)
                        os.makedirs("./aikensa/capturedimages/clip3", exist_ok=True)
                        cv2.imwrite(f"./aikensa/capturedimages/clip3/clip3_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", clipFrame3_dump)
                        print("Clip 3 captured.")
                        self.cam_config.captureClip3 = False

                    combinedImage_raw = combinedImage.copy()
                    combinedImage = self.resizeImage(combinedImage, 1521, 363)
                    
                    croppedFrame1 = self.frameCrop(combinedImage_raw, x=450, y=260, w=320, h=160, wout = 320, hout = 160)
                    croppedFrame2 = self.frameCrop(combinedImage_raw, x=3800, y=260, w=320, h=160, wout = 320, hout = 160)
                    
                    
                    self.kata1Frame.emit(self.convertQImage(croppedFrame1))
                    self.kata2Frame.emit(self.convertQImage(croppedFrame2))

                    frame1_downres = self.resizeImage(frame1)
                    frame2_downres = self.resizeImage(frame2)

                    self.camFrame1.emit(self.convertQImage(frame1_downres))
                    self.camFrame2.emit(self.convertQImage(frame2_downres))

                    self.mergeFrame.emit(self.convertQImage(combinedImage))

                    self.clip1Frame.emit(self.convertQImage(clipFrame1))
                    self.clip2Frame.emit(self.convertQImage(clipFrame2))
                    self.clip3Frame.emit(self.convertQImage(clipFrame3))


            if self.cam_config.widget == 3 or self.cam_config.widget == 4:

                if frame1 is None:
                    frame1 = np.zeros((2048, 3072, 3), dtype=np.uint8)
                if frame2 is None:
                    frame2 = np.zeros((2048, 3072, 3), dtype=np.uint8) 
                homography_blank = homography_blank_canvas.copy()

                if ret1 and ret2:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                if self.cam_config.HDRes != self.previous_HDRes:
                    if not self.cam_config.HDRes:
                        self.cameraMatrix1 = self.adjust_camera_matrix(self.cameraMatrix1, self.scale_factor)
                        self.cameraMatrix2 = self.adjust_camera_matrix(self.cameraMatrix2, self.scale_factor)
                        self.flexibleH1 = self.H1_lowres
                        self.flexibleH2 = self.H2_lowres
                    else:
                        self.cameraMatrix1 = self.adjust_camera_matrix(self.cameraMatrix1, 1/self.scale_factor)
                        self.cameraMatrix2 = self.adjust_camera_matrix(self.cameraMatrix2, 1/self.scale_factor)
                        self.flexibleH1 = self.H1
                        self.flexibleH2 = self.H2

                    self.previous_HDRes = self.cam_config.HDRes  


                if self.cam_config.HDRes == False:
                    frame1 = self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor))
                    frame2 = self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor))
                    homography_blank = self.resizeImage(homography_blank, 
                                                        int(homography_blank.shape[1]//self.scale_factor), 
                                                        int(homography_blank.shape[0]//self.scale_factor))

                
                frame1 = self.undistortFrame(frame1, self.cameraMatrix1, self.distortionCoeff1)
                frame2 = self.undistortFrame(frame2, self.cameraMatrix1, self.distortionCoeff1)

                # combinedFrame_raw, combinedImage, croppedFrame1, croppedFrame2 = self.combineFrames(frame1, frame2, self.flexibleH)
                combinedFrame_raw, combinedImage, croppedFrame1, croppedFrame2 = self.combineFrames_template(frame1, frame2, homography_blank, self.flexibleH1, self.flexibleH2)

                if self.cam_config.HDRes == False:
                    clipFrame1 = self.frameCrop(frame1, x=int(590/self.scale_factor), y=int(0/self.scale_factor), w=int(600/self.scale_factor), h=int(500/self.scale_factor), wout = 128, hout = 128)
                    clipFrame2 = self.frameCrop(frame1, x=int(1900/self.scale_factor), y=int(0/self.scale_factor), w=int(600/self.scale_factor), h=int(500/self.scale_factor), wout = 128, hout = 128)
                    clipFrame3 = self.frameCrop(frame2, x=int(600/self.scale_factor), y=int(0/self.scale_factor), w=int(600/self.scale_factor), h=int(500/self.scale_factor), wout = 128, hout = 128)

                if self.cam_config.HDRes == True:
                    clipFrame1 = self.frameCrop(frame1, x=590, y=0, w=600, h=600, wout = 128, hout = 128)
                    clipFrame2 = self.frameCrop(frame1, x=1900, y=0, w=600, h=600, wout = 128, hout = 128)
                    clipFrame3 = self.frameCrop(frame2, x=600, y=0, w=600, h=600, wout = 128, hout = 128)
                    

                if self.handClassificationModel is not None and self.cam_config.HDRes == False:
                    frame1_handClassify = self.handClassificationModel(cv2.cvtColor(clipFrame1, cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                    frame2_handClassify = self.handClassificationModel(cv2.cvtColor(clipFrame2, cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                    frame3_handClassify = self.handClassificationModel(cv2.cvtColor(clipFrame3, cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                    
                    self.result_handframe1 = list(frame1_handClassify)[0].probs.data.argmax().item()
                    self.result_handframe2 = list(frame2_handClassify)[0].probs.data.argmax().item()
                    self.result_handframe3 = list(frame3_handClassify)[0].probs.data.argmax().item()
                    # 0 for hand in frame, 1 for hand not in frame. It's flipped, I know
                    # print(f"HandFrame1,2,and 3: {self.result_handframe1}, {self.result_handframe2}, {self.result_handframe3}")
                
                if self.musicPlay == True:

                    if self.result_handframe1 == 0:
                        self.handinFrame1 = True
                        if self.handinFrame1Timer is None:
                            self.handinFrame1Timer = time.time()
                            play_do_sound()
                    elif self.handinFrame1 and time.time() - self.handinFrame1Timer > self.clipHandWaitTime:
                        self.handinFrame1 = False
                        self.handinFrame1Timer = None

                    if self.result_handframe2 == 0:
                        self.handinFrame2 = True
                        if self.handinFrame2Timer is None:
                            self.handinFrame2Timer = time.time()
                            play_re_sound()
                    elif self.handinFrame2 and time.time() - self.handinFrame2Timer > self.clipHandWaitTime:
                        self.handinFrame2 = False
                        self.handinFrame2Timer = None

                    if self.result_handframe3 == 0:
                        self.handinFrame3 = True
                        if self.handinFrame3Timer is None:
                            self.handinFrame3Timer = time.time()
                            play_mi_sound()
                    elif self.handinFrame3 and time.time() - self.handinFrame3Timer > self.clipHandWaitTime:
                        self.handinFrame3 = False
                        self.handinFrame3Timer = None

                if self.musicPlay == False:

                    #Logic for work order
                    if self.handinFrameTimer is None:

                        if self.kensa_cycle is False and self.result_handframe1 == 0:
                            self.handinFrameTimer = time.time()
                            play_picking_sound()
                            self.kensa_cycle = True
                            self.cam_config.ctrplrWorkOrder = [1, 0, 0, 0, 0]

                        elif self.kensa_cycle is True and self.result_handframe1 == 0 and self.cam_config.ctrplrWorkOrder == [1, 0, 0, 0, 0]:
                            self.handinFrameTimer = time.time()
                            play_picking_sound()
                            self.cam_config.ctrplrWorkOrder = [1, 1, 0, 0, 0]

                        elif self.kensa_cycle is True and self.result_handframe2 == 0 and self.cam_config.ctrplrWorkOrder == [1, 1, 0, 0, 0]:
                            self.handinFrameTimer = time.time()
                            play_picking_sound()
                            self.cam_config.ctrplrWorkOrder = [1, 1, 1, 0, 0]

                        elif self.kensa_cycle is True and self.result_handframe2 == 0 and self.cam_config.ctrplrWorkOrder == [1, 1, 1, 0, 0]:
                            self.handinFrameTimer = time.time()
                            play_picking_sound()
                            self.cam_config.ctrplrWorkOrder = [1, 1, 1, 1, 0]                    

                        elif self.kensa_cycle is True and self.result_handframe3 == 0 and self.cam_config.ctrplrWorkOrder == [1, 1, 1, 1, 0]:
                            self.handinFrameTimer = time.time()
                            play_picking_sound()
                            self.cam_config.ctrplrWorkOrder = [1, 1, 1, 1, 1] 

                        elif self.result_handframe1 == 1 and self.result_handframe2 == 1 and self.result_handframe3 == 1:
                            None

                        elif self.result_handframe1 is None and self.result_handframe2 is None and self.result_handframe3 is None:
                            None

                        else:
                            play_alarm_sound()
                            self.handinFrameTimer = time.time()
                    else:
                        self.result_handframe1 = 1
                        self.result_handframe2 = 1
                        self.result_handframe3 = 1
                    
                    if self.handinFrameTimer:
                        if time.time() - self.handinFrameTimer > self.clipHandWaitTime:
                            self.handinFrameTimer = None



                if self.cam_config.kensaReset == True:
                    self.kensa_order = []
                    self.kensa_cycle = False
                    self.cam_config.ctrplrWorkOrder = [0, 0, 0, 0, 0]
                    self.cam_config.kensaReset = False

                if self.cam_config.widget == 3:
                    ok_count, ng_count = self.cam_config.ctrplrLHnumofPart
                    self.cam_config.ctrplrLHnumofPart = self.manual_adjustment(ok_count, 
                                                                            ng_count, 
                                                                            self.cam_config.furyou_plus, 
                                                                            self.cam_config.furyou_minus, 
                                                                            self.cam_config.kansei_plus, 
                                                                            self.cam_config.kansei_minus)
                    if self.cam_config.resetCounter == True:
                        ok_count = 0
                        ng_count = 0
                        self.cam_config.ctrplrLHnumofPart = (ok_count, ng_count)
                        self.cam_config.resetCounter = False

                if self.cam_config.widget == 4:
                    ok_count, ng_count = self.cam_config.ctrplrRHnumofPart
                    self.cam_config.ctrplrRHnumofPart = self.manual_adjustment(ok_count, 
                                                                ng_count, 
                                                                self.cam_config.furyou_plus, 
                                                                self.cam_config.furyou_minus, 
                                                                self.cam_config.kansei_plus, 
                                                                self.cam_config.kansei_minus)

                    if self.cam_config.resetCounter == True:
                        ok_count = 0
                        ng_count = 0
                        self.cam_config.ctrplrRHnumofPart = (ok_count, ng_count)
                        self.cam_config.resetCounter = False

                


            
                    ##To manually set the work order
                self.cam_config.ctrplrWorkOrder = [1, 1, 1, 1, 1]

                if self.cam_config.triggerKensa == True or self.oneLoop == True:
                    current_time = time.time()

                    if current_time - self.last_inspection_time < self.inspection_delay: #extra 3 sec after inspection to prevent multiple inspection
                        self.cam_config.triggerKensa = False
                        self.oneLoop = False
                        continue

                    if self.prev_timestamp == None:
                        self.prev_timestamp = datetime.now()

                    timestamp = datetime.now() #datetime.now().strftime('%Y%m%d_%H%M%S')

                    deltaTime = timestamp - self.prev_timestamp
                    self.prev_timestamp = timestamp

                    if self.cam_config.ctrplrWorkOrder != [1, 1, 1, 1, 1]:
                        play_alarm_sound()
                        self.cam_config.triggerKensa = False
                        self.oneLoop = False
                        continue

                    # if self.cam_config.ctrplrWorkOrder == [0,0,0,0,0]:
                    if self.cam_config.ctrplrWorkOrder == [1, 1, 1, 1, 1]:
                        self.cam_config.HDRes = True

                        #Append a word "kensaChuu" to the combined image and emit it
                        combinedImage_wait = combinedImage.copy()
                        text = '検査実施中'
                        font_path = self.kanjiFontPath
                        font_size = 80

                        combinedImage_wait = self.add_text_to_image(combinedImage_wait, text, font_path, font_size)
                        self.mergeFrame.emit(self.convertQImage(combinedImage_wait))

                        combinedFrame_raw_copy = cv2.cvtColor(combinedFrame_raw, cv2.COLOR_BGR2RGB)


                        if self.oneLoop == True:
                            #Detect Clip
                            self.clip_detection = get_sliced_prediction(combinedFrame_raw, 
                                                                        self.ctrplr_clipDetectionModel, 
                                                                        slice_height=968, slice_width=968, 
                                                                        overlap_height_ratio=0.3, overlap_width_ratio=0.2,
                                                                        postprocess_match_metric = "IOS",
                                                                        postprocess_match_threshold = 0.2,
                                                                        postprocess_class_agnostic = True,
                                                                        postprocess_type = "GREEDYNMM",
                                                                        verbose = 0,
                                                                        perform_standard_pred = False)
                            
                            # self.clip_detection.export_visuals(export_dir="./demo_data/")
                            #Detect Katabu Marking 
                            if self.cam_config.widget == 3:
                                self.marking_detection  = self.ctrplr_markingDetectionModel(cv2.cvtColor(croppedFrame2, cv2.COLOR_BGR2RGB), 
                                                                                            stream=True, 
                                                                                            verbose=False,
                                                                                            conf=0.1, iou=0.5)
                                self.hanire_detections = None
                                imgResult, katabumarkingResult, pitch_results, detected_pitch, delta_pitch, hanire, status = ctrplrCheck(combinedFrame_raw, croppedFrame2,
                                                                                                                                        self.clip_detection.object_prediction_list, 
                                                                                                                                        self.marking_detection, 
                                                                                                                                        self.hanire_detections, 
                                                                                                                                        partid="LH")
                                dir_part = self.widget_dir_map.get(self.cam_config.widget)
                                self.save_result_csv("82833W050P", dir_part, 
                                                    self.cam_config.ctrplrLHnumofPart, self.cam_config.ctrplrLHnumofPart, 
                                                    timestamp, deltaTime, 
                                                    self.cam_config.kensainName, 
                                                    pitch_results, delta_pitch, 
                                                    total_length=0)
                       
                                if status == "OK":
                                    ok_count += 1
                                elif status == "NG":
                                    ng_count += 1

                            if self.cam_config.widget == 4:
                                self.marking_detection  = self.ctrplr_markingDetectionModel(cv2.cvtColor(croppedFrame1, cv2.COLOR_BGR2RGB), 
                                                                                            stream=True, 
                                                                                            verbose=False,
                                                                                            conf=0.1, iou=0.5)
                                self.hanire_detections = None
                                imgResult, katabumarkingResult, pitch_results, detected_pitch, delta_pitch, hanire, status = ctrplrCheck(combinedFrame_raw, croppedFrame1,
                                                                                                                                        self.clip_detection.object_prediction_list, 
                                                                                                                                        self.marking_detection, 
                                                                                                                                        self.hanire_detections, 
                                                                                                                                        partid="RH")
                                dir_part = self.widget_dir_map.get(self.cam_config.widget)
                                self.save_result_csv("82833W040P", dir_part, 
                                                    self.cam_config.ctrplrRHnumofPart, self.cam_config.ctrplrRHnumofPart, 
                                                    timestamp, deltaTime, 
                                                    self.cam_config.kensainName, 
                                                    pitch_results, delta_pitch, 
                                                    total_length=0)
                                
                                                            
                                if status == "OK":
                                    ok_count += 1
                                elif status == "NG":
                                    ng_count += 1
                                
        
                            imgResult_copy = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
                            

                            os.makedirs(f"./aikensa/inspection_results/{dir_part}/nama/{timestamp.strftime('%Y%m%d')}", exist_ok=True)
                            os.makedirs(f"./aikensa/inspection_results/{dir_part}/kekka/{timestamp.strftime('%Y%m%d')}", exist_ok=True)
                            
                            cv2.imwrite(f"./aikensa/inspection_results/{dir_part}/nama/{timestamp.strftime('%Y%m%d')}/{timestamp.strftime('%Y%m%d%H%M%S')}.png", combinedFrame_raw_copy)
                            cv2.imwrite(f"./aikensa/inspection_results/{dir_part}/kekka/{timestamp.strftime('%Y%m%d')}/{timestamp.strftime('%Y%m%d%H%M%S')}.png", imgResult_copy)

                            self.cam_config.ctrplrLHnumofPart = (ok_count, ng_count)
                            self.cam_config.ctrplrRHnumofPart = (ok_count, ng_count)

                            combinedImage = self.resizeImage(imgResult, 1791, 428)


                            self.mergeFrame.emit(self.convertQImage(combinedImage))
                            if self.cam_config.widget == 3:
                                self.kata2Frame.emit(self.convertQImage(katabumarkingResult))
                                self.ctrplrLH_numofPart_updated.emit(self.cam_config.ctrplrLHnumofPart)
                                self.ctrplrLH_pitch_updated.emit(pitch_results)

                            if self.cam_config.widget == 4:
                                self.kata1Frame.emit(self.convertQImage(katabumarkingResult))
                                self.ctrplrRH_numofPart_updated.emit(self.cam_config.ctrplrRHnumofPart)
                                self.ctrplrRH_pitch_updated.emit(pitch_results)

                            self.ctrplrworkorderSignal.emit(self.cam_config.ctrplrWorkOrder)


                            #sleep for self.inspection_delay
                            time.sleep(self.inspection_delay)

                            self.last_inspection_time = time.time()

                            self.clip_detection = None
                            self.oneLoop = False
                            self.cam_config.HDRes = False
                            self.cam_config.ctrplrWorkOrder = [0, 0, 0, 0, 0]
                            self.kensa_order = [] #reinitialize the kensa order
                            self.kensa_cycle = False #reinitialize the kensa cycle
                            self.clip1Frame.emit(self.convertQImage(clipFrame1))
                            self.clip2Frame.emit(self.convertQImage(clipFrame2))
                            self.clip3Frame.emit(self.convertQImage(clipFrame3))
                            continue

                        self.oneLoop = True
                        self.cam_config.triggerKensa = False

                self.mergeFrame.emit(self.convertQImage(combinedImage))

                if self.cam_config.widget == 3:
                    self.kata2Frame.emit(self.convertQImage(croppedFrame2))
                    #emit blank image for kata1Frame
                    blankFrame = np.zeros((160, 320, 3), dtype=np.uint8)
                    self.kata1Frame.emit(self.convertQImage(blankFrame))
                if self.cam_config.widget == 4:
                    self.kata1Frame.emit(self.convertQImage(croppedFrame1))
                    #emit blank image for kata2Frame
                    blankFrame = np.zeros((160, 320, 3), dtype=np.uint8)
                    self.kata2Frame.emit(self.convertQImage(blankFrame))


                self.clip1Frame.emit(self.convertQImage(clipFrame1))
                self.clip2Frame.emit(self.convertQImage(clipFrame2))
                self.clip3Frame.emit(self.convertQImage(clipFrame3)) 

                self.handFrame1.emit(not self.handinFrame1)
                self.handFrame2.emit(not self.handinFrame2)
                self.handFrame3.emit(not self.handinFrame3)
                
                self.ctrplrworkorderSignal.emit(self.cam_config.ctrplrWorkOrder)

                if self.cam_config.widget == 3:
                    self.ctrplrLH_numofPart_updated.emit(self.cam_config.ctrplrLHnumofPart)
                    self.ctrplrLH_pitch_updated.emit(self.cam_config.ctrplrLHpitch)
                if self.cam_config.widget == 4:
                    self.ctrplrRH_numofPart_updated.emit(self.cam_config.ctrplrRHnumofPart)
                    self.ctrplrRH_pitch_updated.emit(self.cam_config.ctrplrRHpitch)


        cap_cam1.release()
        print("Camera 1 released.")
        cap_cam2.release()
        print("Camera 2 released.")

    def save_result_csv(self, part_name, dir_part, 
                        numofPart, current_numofPart, 
                        timestamp, deltaTime, kensainName, 
                        detected_pitch, delta_pitch, 
                        total_length):
        
        
        detected_pitch_str = str(detected_pitch).replace('[', '').replace(']', '')
        delta_pitch_str = str(delta_pitch).replace('[', '').replace(']', '')

        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H:%M:%S")
        deltaTime = deltaTime.total_seconds()

        base_dir = f"./aikensa/inspection_results/{dir_part}/{timestamp_date}/results"
        os.makedirs(base_dir, exist_ok=True)


        if not os.path.exists(f"{base_dir}/inspection_results.csv"):
            with open(f"{base_dir}/inspection_results.csv", mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["PartName", 'KensaResult(OK,/NG)', 
                                 "CurrentKensaResult(OK/NG)", 'KensaTime', 
                                 "KensaDate",  "KensaTimeLength", 
                                 'KensaSagyoushaName',
                                'DetectedPitch', "DeltaPitch", 
                                'TotalLength'])

                writer.writerow([part_name, numofPart, 
                                 current_numofPart, timestamp_hour, 
                                 timestamp_date, deltaTime, 
                                 kensainName, detected_pitch_str, 
                                 delta_pitch_str, total_length
                                 ])
                
        else:
            with open(f"{base_dir}/inspection_results.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([part_name, numofPart, 
                                 current_numofPart, timestamp_hour, 
                                 timestamp_date, deltaTime, 
                                 kensainName, detected_pitch_str, 
                                 delta_pitch_str, total_length
                                 ])
                
            # Call the method to save data to the database
        # self.save_result_database(part_name, numofPart, 
        #                           current_numofPart, timestamp_hour, 
        #                           timestamp_date, deltaTime, 
        #                           kensainName, detected_pitch_str, 
        #                           delta_pitch_str, total_length
        #                           )

    def adjust_camera_matrix(self, camera_matrix, scale_factor):
        camera_matrix[0][0] /= scale_factor
        camera_matrix[1][1] /= scale_factor
        camera_matrix[0][2] /= scale_factor
        camera_matrix[1][2] /= scale_factor
        return camera_matrix

    def adjust_transform_matrix(self, matrix, scale_factor):
        # matrix[0, 0] /= scale_factor  # Adjust sx
        # matrix[1, 1] /= scale_factor  # Adjust sy

        # matrix[0, 1] /= scale_factor  # Adjust shear in x
        # matrix[1, 0] /= scale_factor  # Adjust shear in y

        matrix[0, 2] /= scale_factor  # Adjust tx
        matrix[1, 2] /= scale_factor  # Adjust ty
        # matrix[2, 2] /= scale_factor

        return matrix

    def undistortFrame(self, frame,cameraMatrix, distortionCoeff):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.undistort(frame, cameraMatrix, distortionCoeff, None, cameraMatrix)
        return frame

    def combineFrames(self, frame1, frame2, H):
        combinedFrame = warpTwoImages(frame2, frame1, H)

        croppedFrame1 = None
        croppedFrame2 = None

        combinedFrame, _ = planarize(combinedFrame, self.scale_factor if not self.cam_config.HDRes else 1.0)

        combinedFrame_raw = combinedFrame.copy()
        combinedFrame = self.resizeImage(combinedFrame, 1791, 428)
       
        
        if self.cam_config.HDRes == False:
            croppedFrame1 = self.frameCrop(combinedFrame_raw, x=int(450/self.scale_factor), y=int(260/self.scale_factor), w=int(320/self.scale_factor), h=int(160/self.scale_factor), wout = int(320), hout = int(160))
            croppedFrame2 = self.frameCrop(combinedFrame_raw, x=int(3800/self.scale_factor), y=int(260/self.scale_factor), w=int(320/self.scale_factor), h=int(160/self.scale_factor), wout = int(320), hout = int(160))
        if self.cam_config.HDRes == True:
            croppedFrame1 = self.frameCrop(combinedFrame_raw, x=450, y=260, w=320, h=160, wout = 320, hout = 160)
            croppedFrame2 = self.frameCrop(combinedFrame_raw, x=3800, y=260, w=320, h=160, wout = 320, hout = 160)


        if croppedFrame1 is None:
            croppedFrame1 = np.zeros((160, 320, 3), dtype=np.uint8)
        if croppedFrame2 is None:
            croppedFrame2 = np.zeros((160, 320, 3), dtype=np.uint8)

        return combinedFrame_raw, combinedFrame, croppedFrame1, croppedFrame2
    
    def add_text_to_image(self, combinedImage, text, font_path, font_size):
        pil_image = Image.fromarray(cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, font_size)
        
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (pil_image.width - text_width) // 2
        y = (pil_image.height + text_height) // 2
        
        draw.text((x, y), text, font=font, fill=(255, 50, 100))
        
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return open_cv_image

    
    def combineFrames_template(self, frame1, frame2, template, H1, H2):
        combinedFrame = warpTwoImages_template(template, frame1, H1)
        combinedFrame = warpTwoImages_template(combinedFrame, frame2, H2)


        croppedFrame1 = None
        croppedFrame2 = None

        combinedFrame, _ = planarize(combinedFrame, self.scale_factor if not self.cam_config.HDRes else 1.0)

        # cv2.imwrite("combinedFrametemplate.jpg", combinedFrame)


        combinedFrame_raw = combinedFrame.copy()
        combinedFrame = self.resizeImage(combinedFrame, 1791, 428)
        
        if self.cam_config.HDRes == False:
            croppedFrame1 = self.frameCrop(combinedFrame_raw, x=int(450/self.scale_factor), y=int(260/self.scale_factor), w=int(320/self.scale_factor), h=int(160/self.scale_factor), wout = int(320), hout = int(160))
            croppedFrame2 = self.frameCrop(combinedFrame_raw, x=int(3800/self.scale_factor), y=int(260/self.scale_factor), w=int(320/self.scale_factor), h=int(160/self.scale_factor), wout = int(320), hout = int(160))
        if self.cam_config.HDRes == True:
            croppedFrame1 = self.frameCrop(combinedFrame_raw, x=450, y=260, w=320, h=160, wout = 320, hout = 160)
            croppedFrame2 = self.frameCrop(combinedFrame_raw, x=3800, y=260, w=320, h=160, wout = 320, hout = 160)

        if croppedFrame1 is None:
            croppedFrame1 = np.zeros((160, 320, 3), dtype=np.uint8)
        if croppedFrame2 is None:
            croppedFrame2 = np.zeros((160, 320, 3), dtype=np.uint8)

        return combinedFrame_raw, combinedFrame, croppedFrame1, croppedFrame2

    def stop(self):
        self.running = False
        time.sleep(0.5)
    
    def read_calibration_params(self, path):
        with open(path) as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff

    def resizeImage(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def stop(self):
        self.running = False
        print(f"Running is set to {self.running}")
  
    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def downSampling(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert resized cv2 image to QImage
        h, w, ch = resized_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(resized_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def frameCrop(self,img, x=0, y=0, w=640, h=480, wout=640, hout=480):
        #crop and resize image to wout and hout
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        img = img[y:y+h, x:x+w]
        try:
            img = cv2.resize(img, (wout, hout), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print("An error occurred while cropping the image:", str(e))
        return img
    
    def manual_adjustment(self, ok_count, ng_count, furyou_plus, furyou_minus, kansei_plus, kansei_minus):
        if furyou_plus:
            ng_count += 1
            self.cam_config.furyou_plus = False

        if furyou_minus and ng_count > 0:
            ng_count -= 1
            self.cam_config.furyou_minus = False

        if kansei_plus:
            ok_count += 1
            self.cam_config.kansei_plus = False

        if kansei_minus and ok_count > 0:
            ok_count -= 1
            self.cam_config.kansei_minus = False
        return ok_count, ng_count

    def initialize_model(self):
        #Change based on the widget
        handClassificationModel = None
        ctrplr_clipDetectionModel = None
        ctrplr_hanireDetectionModel = None
        ctrplr_markingDetectionModel = None

        if self.cam_config.widget == 3 or self.cam_config.widget == 4:
            handClassificationModel = YOLO("./aikensa/custom_weights/handClassify.pt")
            ctrplr_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                           model_path="./aikensa/custom_weights/weights_5755A49X.pt",
                                                                           confidence_threshold=0.5,
                                                                           device="cuda:0",
            )
            ctrplr_markingDetectionModel = YOLO("./aikensa/custom_weights/weights_5755A49X_marking.pt")
            

        self.handClassificationModel = handClassificationModel
        self.ctrplr_clipDetectionModel = ctrplr_clipDetectionModel
        self.ctrplr_markingDetectionModel = ctrplr_markingDetectionModel

        print("HandClassificationModel initialized.")
        
