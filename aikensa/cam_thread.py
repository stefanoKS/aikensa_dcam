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
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix, calculateHomography, warpTwoImages
from aikensa.opencv_imgprocessing.arucoplanarize import planarize

from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_re_sound, play_mi_sound, play_fa_sound, play_sol_sound, play_la_sound, play_si_sound, play_alarm_sound

from ultralytics import YOLO
from aikensa.parts_config.ctrplr_8283XW0W0P import partcheck as ctrplrCheck

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

    HDRes: bool = False
    triggerKensa: bool = False
    kensaReset: bool = False

    ctrplrpitch: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    ctrplrWorkOrder : List[int] = field(default_factory=lambda: [0, 0, 0])




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
            3: "5755A491",
            4: "5755A492"
        }

        self.previous_HDRes = self.cam_config.HDRes
        self.scale_factor = 5.0

        self.handClassificationModel = None

        self.clipHandWaitTime = 2.5
        self.inspection_delay = 5.0

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

        self.kensa_cycle = False
        self.kensa_order = []

        self.musicPlay = False

    def run(self):

        cap_cam1 = initialize_camera(0)
        print(f"Initiliazing Camera 1.... Located on {cap_cam1}")
        cap_cam2 = initialize_camera(2)
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

        if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
            with open("./aikensa/cameracalibration/homography_param.yaml") as file:
                homography_param = yaml.load(file, Loader=yaml.FullLoader)
                H = np.array(homography_param)

        if os.path.exists("./aikensa/cameracalibration/homography_param_lowres.yaml"):
            with open("./aikensa/cameracalibration/homography_param_lowres.yaml") as file:
                homography_param_lowres = yaml.load(file, Loader=yaml.FullLoader)
                H_lowres = np.array(homography_param_lowres)

        self.cameraMatrix1 = self.adjust_camera_matrix(cameraMatrix1, self.scale_factor)
        self.cameraMatrix2 = self.adjust_camera_matrix(cameraMatrix2, self.scale_factor)
        self.distortionCoeff1 = distortionCoeff1
        self.distortionCoeff2 = distortionCoeff2
        self.H = self.adjust_transform_matrix(H, self.scale_factor)
        self.H_lowres = H_lowres

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

                    if self.cam_config.deleteHomo == True:
                        if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
                            os.remove("./aikensa/cameracalibration/homography_param.yaml")
                        if os.path.exists("./aikensa/cameracalibration/homography_param_lowres.yaml"):
                            os.remove("./aikensa/cameracalibration/homography_param_lowres.yaml")
                        self.cam_config.deleteHomo = False

                    if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
                        with open("./aikensa/cameracalibration/homography_param.yaml") as file:
                            homography_param = yaml.load(file, Loader=yaml.FullLoader)
                            H = np.array(homography_param)
                            combinedImage = warpTwoImages(frame2, frame1, H)
                    
                    if os.path.exists("./aikensa/cameracalibration/homography_param_lowres.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_lowres.yaml") as file:
                            homography_param_lowres = yaml.load(file, Loader=yaml.FullLoader)
                            H_lowres = np.array(homography_param_lowres)
                            combinedImage_lowres = warpTwoImages(self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor)), self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor)), H_lowres)

                    else :
                        combinedImage = np.zeros((363, 1521, 3), dtype=np.uint8)

                    combinedImage, _ = planarize(combinedImage, scale_factor=1.0)
                    combinedImage_lowres, _t = planarize(combinedImage_lowres, scale_factor = self.scale_factor)

                    cv2.imwrite("combinedImage.jpg", combinedImage)
                    cv2.imwrite("combinedImage_lowres.jpg", combinedImage_lowres)

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


            if self.cam_config.widget == 3:

                if frame1 is None:
                    frame1 = np.zeros((2048, 3072, 3), dtype=np.uint8)
                if frame2 is None:
                    frame2 = np.zeros((2048, 3072, 3), dtype=np.uint8) 

                if ret1 and ret2:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                if self.cam_config.HDRes != self.previous_HDRes:
                    if not self.cam_config.HDRes:
                        self.cameraMatrix1 = self.adjust_camera_matrix(self.cameraMatrix1, self.scale_factor)
                        self.cameraMatrix2 = self.adjust_camera_matrix(self.cameraMatrix2, self.scale_factor)
                        self.H = self.H_lowres
                    else:
                        self.cameraMatrix1 = self.adjust_camera_matrix(self.cameraMatrix1, 1/self.scale_factor)
                        self.cameraMatrix2 = self.adjust_camera_matrix(self.cameraMatrix2, 1/self.scale_factor)
                        self.H = self.adjust_transform_matrix(self.H, 1 / self.scale_factor)

                    self.previous_HDRes = self.cam_config.HDRes  


                if self.cam_config.HDRes == False:
                    frame1 = self.resizeImage(frame1, int(3072//self.scale_factor), int(2048//self.scale_factor))
                    frame2 = self.resizeImage(frame2, int(3072//self.scale_factor), int(2048//self.scale_factor))

                
                frame1 = self.undistortFrame(frame1, self.cameraMatrix1, self.distortionCoeff1)
                frame2 = self.undistortFrame(frame2, self.cameraMatrix1, self.distortionCoeff1)

                combinedFrame_raw, combinedImage, croppedFrame1, croppedFrame2 = self.combineFrames(frame1, frame2, self.H)

                # cv2.imwrite("frame1.jpg", frame1)
                # cv2.imwrite("frame2.jpg", frame2)
                
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
                    if self.result_handframe1 == 0:
                        self.handinFrame1 = True
                        if self.handinFrame1Timer is None:
                            self.handinFrame1Timer = time.time()
                            
                            if self.kensa_cycle and self.kensa_order == ["a"]:
                                self.kensa_order.append("a")
                                self.cam_config.ctrplrWorkOrder = [1, 1, 0]
                                play_re_sound()

                            
                            if self.kensa_cycle == False:
                                self.kensa_cycle = True
                                self.kensa_order.append("a")
                                self.cam_config.ctrplrWorkOrder = [1, 0, 0]
                                play_do_sound()

                    if self.handinFrame1 and time.time() - self.handinFrame1Timer > self.clipHandWaitTime:
                        self.handinFrame1 = False
                        self.handinFrame1Timer = None

                    if self.result_handframe2 == 0:
                        self.handinFrame2 = True
                        if self.handinFrame2Timer is None:
                            self.handinFrame2Timer = time.time()

                            if self.kensa_cycle and self.kensa_order == ["a", "a"]:
                                self.kensa_order.append("b")
                                self.cam_config.ctrplrWorkOrder = [1, 1, 1]
                                play_mi_sound()
                            
                    elif self.handinFrame2 and time.time() - self.handinFrame2Timer > self.clipHandWaitTime:
                        self.handinFrame2 = False
                        self.handinFrame2Timer = None

                if self.cam_config.kensaReset == True:
                    self.kensa_order = []
                    self.kensa_cycle = False
                    self.cam_config.ctrplrWorkOrder = [0, 0, 0]
                    self.cam_config.kensaReset = False
            
                if self.cam_config.triggerKensa == True or self.oneLoop == True:
                    print (self.cam_config.ctrplrWorkOrder)

                    if self.cam_config.ctrplrWorkOrder != [1, 1, 1]:
                        play_alarm_sound()
                        self.cam_config.triggerKensa = False
                        self.oneLoop = False
                        continue

                    if self.cam_config.ctrplrWorkOrder == [1, 1, 1]:
                        self.cam_config.HDRes = True

                        if self.oneLoop == True:
                            # cv2.imwrite("combinedbeforeImage.jpg", combinedFrame_raw)
                            self.clip_detection = get_sliced_prediction(combinedFrame_raw, self.ctrplr_clipDetectionModel, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
                            self.hanire_detections = None
                            # print("Clip Detection: ", self.clip_detection.object_prediction_list)
                            # cv2.imwrite("combinedImage.jpg", combinedFrame_raw)
                            imgResult, pitch_results, detected_pitch, delta_pitch, hanire = ctrplrCheck(combinedFrame_raw, self.clip_detection.object_prediction_list, self.hanire_detections, partid="LH")
                            _imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
                            cv2.imwrite("imgResult.jpg", _imgResult)
                            combinedImage = self.resizeImage(imgResult, 1791, 428)
                            print("Inference done.")
                            self.mergeFrame.emit(self.convertQImage(combinedImage))
                            self.kata1Frame.emit(self.convertQImage(croppedFrame1))
                            self.kata2Frame.emit(self.convertQImage(croppedFrame2))
                            self.ctrplrworkorderSignal.emit(self.cam_config.ctrplrWorkOrder)
                            #sleep for self.inspection_delay
                            self.clip_detection = None
                            self.oneLoop = False
                            self.cam_config.HDRes = False
                            self.cam_config.ctrplrWorkOrder = [0, 0, 0]
                            self.kensa_order = [] #reinitialize the kensa order
                            self.kensa_cycle = False #reinitialize the kensa cycle
                            time.sleep(self.inspection_delay)
                            continue

                        self.oneLoop = True
                        self.cam_config.triggerKensa = False

                self.mergeFrame.emit(self.convertQImage(combinedImage))
                self.kata1Frame.emit(self.convertQImage(croppedFrame1))
                self.kata2Frame.emit(self.convertQImage(croppedFrame2))

                self.clip1Frame.emit(self.convertQImage(clipFrame1))
                self.clip2Frame.emit(self.convertQImage(clipFrame2))
                # self.clip3Frame.emit(self.convertQImage(clipFrame3)) #only 2 clips for this part

                self.handFrame1.emit(not self.handinFrame1)
                self.handFrame2.emit(not self.handinFrame2)
                # self.handFrame3.emit(not self.handinFrame3) #only 2 clips for this part
                
                self.ctrplrworkorderSignal.emit(self.cam_config.ctrplrWorkOrder)



        cap_cam1.release()
        print("Camera 1 released.")
        cap_cam2.release()
        print("Camera 2 released.")

    # def partCheck(self, frame1, frame2, ret1, ret2, widget):



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

        # cv2.imwrite("frame1.jpg", frame1)
        # cv2.imwrite("frame2.jpg", frame2)
        # cv2.imwrite("combinedImage.jpg", combinedFrame)

        combinedFrame, _ = planarize(combinedFrame, self.scale_factor if not self.cam_config.HDRes else 1.0)

        # cv2.imwrite("combinedImage.jpg", combinedFrame)

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

        # cv2.imwrite("combinedImage_raw.jpg", combinedFrame_raw)
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
    

    def initialize_model(self):
        #Change based on the widget
        handClassificationModel = None
        ctrplr_clipDetectionModel = None
        ctrplr_hanireDetectionModel = None

        if self.cam_config.widget == 3:
            handClassificationModel = YOLO("./aikensa/custom_weights/handClassify.pt")
            ctrplr_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                           model_path="./aikensa/custom_weights/weights_5755A49X.pt",
                                                                           confidence_threshold=0.6,
                                                                           device="cuda:0",
            )
            

        self.handClassificationModel = handClassificationModel
        self.ctrplr_clipDetectionModel = ctrplr_clipDetectionModel

        print("HandClassificationModel initialized.")