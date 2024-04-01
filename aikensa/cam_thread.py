import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time
import csv


from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix, calculateHomography, warpTwoImages
from aikensa.opencv_imgprocessing.arucoplanarize import planarize

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    calculateCamMatrix1: bool = False
    calculateCamMatrix2: bool = False
    captureCam1: bool = False
    captureCam2: bool = False
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

class CameraThread(QThread):

    camFrame1 = pyqtSignal(QImage)
    camFrame2 = pyqtSignal(QImage)
    mergeFrame = pyqtSignal(QImage)
    kata1Frame = pyqtSignal(QImage)
    kata2Frame = pyqtSignal(QImage)

    def __init__(self, cam_config: CameraConfig = None):
        super(CameraThread, self).__init__()
        self.running = True
        self.charucoTimer = None

        if cam_config is None:
            self.cam_config = CameraConfig()    
        else:
            self.cam_config = cam_config

        self.widget_dir_map={
            3: "5755A491",
            4: "5755A492"

        }

    def run(self):
        cap_cam1 = initialize_camera(0)
        cap_cam2 = initialize_camera(2)

        while self.running:

            current_time = time.time()

            ret1, frame1 = cap_cam1.read()
            frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
            ret2, frame2 = cap_cam2.read()
            frame2 = cv2.rotate(frame2, cv2.ROTATE_180)



            if frame1 is not None:
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
            if frame2 is not None:
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
                if not os.path.exists("./aikensa/cameracalibration"):
                    os.makedirs("./aikensa/cameracalibration")
                with open("./aikensa/cameracalibration/homography_param.yaml", "w") as file:
                    yaml.dump(homographyMatrix.tolist(), file)
                self.cam_config.calculateHomo = False

            if self.cam_config.deleteHomo == True:
                if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
                    os.remove("./aikensa/cameracalibration/homography_param.yaml")
                self.cam_config.deleteHomo = False

            if os.path.exists("./aikensa/cameracalibration/homography_param.yaml"):
                with open("./aikensa/cameracalibration/homography_param.yaml") as file:
                    homography_param = yaml.load(file, Loader=yaml.FullLoader)
                    H = np.array(homography_param)

                    combinedImage = warpTwoImages(frame2, frame1, H)
                    # cv2.imwrite("combinedImage.jpg", combinedImage)
                    # combinedImage_raw = combinedImage.copy()
                    # combinedImage = self.resizeImage(combinedImage, 1521, 521)

            else :
                combinedImage = np.zeros((363, 1521, 3), dtype=np.uint8)

            #check ./aikensa/param/warptransform.yaml, if exists use that value
            combinedImage, _ = planarize(combinedImage)

            if self.cam_config.savePlanarize == True:
                os.makedirs("./aikensa/param", exist_ok=True)
                #Save planarize transform to warptransform.yaml (from the arucoplanarize.py)
                with open('./aikensa/param/warptransform.yaml', 'w') as file:
                    yaml.dump(_.tolist(), file)
                self.cam_config.savePlanarize = False

            if self.cam_config.saveImage == True:
                os.makedirs("./aikensa/capturedimages", exist_ok=True)
                #conver the image to RGB

                combinedImage_dump = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"./aikensa/capturedimages/capturedimage_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", combinedImage_dump)

                croppedFrame1_dump = cv2.cvtColor(croppedFrame1, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"./aikensa/capturedimages/croppedFrame1_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", croppedFrame1_dump)

                croppedFrame2_dump = cv2.cvtColor(croppedFrame2, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"./aikensa/capturedimages/croppedFrame2_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", croppedFrame2_dump)
                print("Images saved.")
                self.cam_config.saveImage = False

            combinedImage_raw = combinedImage.copy()
            combinedImage = self.resizeImage(combinedImage, 1521, 363)

            croppedFrame1 = self.frameCrop(combinedImage_raw, x=450, y=260, w=320, h=160, wout = 320, hout = 160)
            croppedFrame2 = self.frameCrop(combinedImage_raw, x=3800, y=260, w=320, h=160, wout = 320, hout = 160)

            self.camFrame1.emit(image1)
            self.camFrame2.emit(image2)
            self.mergeFrame.emit(self.convertQImage(combinedImage))

            self.kata1Frame.emit(self.convertQImage(croppedFrame1))
            self.kata2Frame.emit(self.convertQImage(croppedFrame2))


        cap_cam1.release()
        cap_cam2.release()

    def resizeImage(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized_image
    
    def stop(self):
        self.running = False
        
    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def downSampling(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # Convert resized cv2 image to QImage
        h, w, ch = resized_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(resized_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def frameCrop(self,img=None, x=0, y=0, w=640, h=480, wout=640, hout=480):
        #crop and resize image to wout and hout
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (wout, hout), interpolation=cv2.INTER_AREA)
        return img