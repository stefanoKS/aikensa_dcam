import re
import cv2
import sys
import yaml
import os
from enum import Enum
import time
import datetime

from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.calibration_thread import CalibrationThread, CalibrationConfig
from aikensa.inspection_thread import InspectionThread, InspectionConfig

from aikensa.sio_thread import ServerMonitorThread
from aikensa.time_thread import TimeMonitorThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui', # index 0
    'aikensa/qtui/calibration_cam1.ui', # index 1
    'aikensa/qtui/calibration_cam2.ui', # index 2         
    'aikensa/qtui/camera_merge.ui',            # index 3
    "aikensa/qtui/empty.ui", #empty 4
    "aikensa/qtui/P658107YA0A.ui",        # index 5
    "aikensa/qtui/P808387YA0A.ui",        # index 6
    "aikensa/qtui/P828387YA0A.ui", #empty 7
    "aikensa/qtui/P828387YA6A.ui", #empty 8
    "aikensa/qtui/P828397YA6A.ui", #empty 9
    "aikensa/qtui/P828387YA1A.ui", #empty 10
    "aikensa/qtui/P828397YA1A.ui", #empty 11
    "aikensa/qtui/P731957YA0A.ui", #empty 12
    "aikensa/qtui/P8462284S00.ui", #empty 13
    "aikensa/qtui/empty.ui", #empty 14
    "aikensa/qtui/empty.ui", #empty 15
    "aikensa/qtui/empty.ui", #empty 16
    "aikensa/qtui/empty.ui", #empty 17
    "aikensa/qtui/empty.ui", #empty 18
    "aikensa/qtui/empty.ui", #empty 19
    "aikensa/qtui/empty.ui", #empty 20
    "aikensa/qtui/P5902A509_dailyTenken_01.ui",  # index 21
    "aikensa/qtui/P5902A509_dailyTenken_02.ui",  # index 22
    "aikensa/qtui/P5902A509_dailyTenken_03.ui",  # index 23
    "aikensa/qtui/P658207LE0A_dailyTenken_01.ui",  # index 24
    "aikensa/qtui/P658207LE0A_dailyTenken_02.ui",  # index 25
]


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.calibration_thread = CalibrationThread(CalibrationConfig())
        self.inspection_thread = InspectionThread(InspectionConfig())   
        self._setup_ui()

        # self.cam_thread = CameraThread(CameraConfig())
        # self._setup_ui()
        # self.cam_thread.start()

        # Thread for SiO
        HOST = '192.168.0.100'  # Use the IP address from SiO settings
        PORT = 30001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.08)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.start()

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.initial_colors = {}#store initial colors of the labels

        self.widget_dir_map = {
            5: "658107YA0A",
            6: "808387YA0A",
            7: "828387YA0A",
            8: "828387YA6A",
            9: "828397YA6A",
            10: "828387YA1A",
            11: "828397YA1A",
            12: "731957YA0A",
            13: "8462284S00"
        }

        self.prevTriggerStates = 0
        self.TriggerWaitTime = 2.0
        self.currentTime = time.time()

    def timeUpdate(self, time):
        for label in self.timeLabel:
            if label:
                label.setText(time)

    def handle_server_status(self, is_up):
        status_text = "ON" if is_up else "OFF"
        status_color = "green" if is_up else "red"

        #to show the label later. Implement later

        for label in self.siostatus_server:
            if label:  # Check if the label was found correctly
                label.setText(status_text)
                label.setStyleSheet(f"color: {status_color};")


    def handle_input_states(self, input_states):
        # print(f"Input states: {input_states}")
        if input_states:
            if input_states[0] == 1 and self.prevTriggerStates == 0:
                self.trigger_kensa()
                self.prevTriggerStates = input_states[0]
                # print("Triggered Kensa")
            if time.time() - self.currentTime > self.TriggerWaitTime:
                # print("timePassed")
                self.prevTriggerStates = 0
                self.currentTime = time.time()
            else:
                pass

    def trigger_kensa(self):
        self.Inspect_button.click()
        # self.button_kensa4.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):

        self.calibration_thread.CalibCamStream.connect(self._setCalibFrame)

        self.calibration_thread.CamMerge1.connect(self._setMergeFrame1)
        self.calibration_thread.CamMerge2.connect(self._setMergeFrame2)
        self.calibration_thread.CamMergeAll.connect(self._setMergeFrameAll)

        self.inspection_thread.part1Cam.connect(self._setPartFrame1)
        # self.inspection_thread.P5902A509_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P5902A509)
        # self.inspection_thread.P658207LE0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658207LE0A)

        self.inspection_thread.P658107YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P658107YA0A)
        self.inspection_thread.P808387YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P808387YA0A)
        self.inspection_thread.P828387YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828387YA0A)
        self.inspection_thread.P828387YA6A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828387YA6A)
        self.inspection_thread.P828397YA6A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828397YA6A)
        self.inspection_thread.P828387YA1A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828387YA1A)
        self.inspection_thread.P828397YA1A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P828397YA1A)
        self.inspection_thread.P731957YA0A_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P731957YA0A)
        self.inspection_thread.P8462284S00_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P8462284S00)


        self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        dailytenken01_P5902A509_widget = self.stackedWidget.widget(21)
        dailytenken02_P5902A509_widget = self.stackedWidget.widget(22)
        dailytenken03_P5902A509_widget = self.stackedWidget.widget(23)
        dailytenken01_P658207LE0A_widget = self.stackedWidget.widget(24)
        dailytenken02_P658207LE0A_widget = self.stackedWidget.widget(25)

        cameraCalibration1_widget = self.stackedWidget.widget(1)
        cameraCalibration2_widget = self.stackedWidget.widget(2)
        mergeCamera_widget = self.stackedWidget.widget(3)

        partInspection_P5902A509 = self.stackedWidget.widget(5)
        partInspection_P5902A510 = self.stackedWidget.widget(6)
        partInspection_P658207LE0A = self.stackedWidget.widget(7)
        partInspection_P5819A107 = self.stackedWidget.widget(8)

        cameraCalibration1_button = main_widget.findChild(QPushButton, "camcalibrationbutton1")
        cameraCalibration2_button = main_widget.findChild(QPushButton, "camcalibrationbutton2")
        mergeCamera_button = main_widget.findChild(QPushButton, "cameraMerge")


        dailytenken01_P5902A509_button = main_widget.findChild(QPushButton, "dailytenkenbutton_P5902A509")
        dailytenken02_P5902A509_button = dailytenken01_P5902A509_widget.findChild(QPushButton, "nextButton")
        dailytenken03_P5902A509_button = dailytenken02_P5902A509_widget.findChild(QPushButton, "nextButton")
        dailytenken_kanryou_P5902A509_button = dailytenken03_P5902A509_widget.findChild(QPushButton, "finishButton")

        dailytenken01_P658207LE0A_button= main_widget.findChild(QPushButton, "dailytenkenbutton_P658207LE0A")
        dailytenken02_P658207LE0A_button = dailytenken01_P658207LE0A_widget.findChild(QPushButton, "nextButton")
        dailytenken_kanryou_P658207LE0A_button = dailytenken02_P658207LE0A_widget.findChild(QPushButton, "finishButton")

        if cameraCalibration1_button:
            cameraCalibration1_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            cameraCalibration1_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 1))
            cameraCalibration1_button.clicked.connect(self.calibration_thread.start)

        if cameraCalibration2_button:
            cameraCalibration2_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
            cameraCalibration2_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 2))
            cameraCalibration2_button.clicked.connect(self.calibration_thread.start)

        if mergeCamera_button:
            mergeCamera_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
            mergeCamera_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 3))
            mergeCamera_button.clicked.connect(self.calibration_thread.start)

        for i in range(1, 3):
            CalibrateSingleFrame = self.stackedWidget.widget(i).findChild(QPushButton, "calibSingleFrame")
            CalibrateSingleFrame.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateSingeFrameMatrix", True))

            CalibrateFinalCameraMatrix = self.stackedWidget.widget(i).findChild(QPushButton, "calibCam")
            CalibrateFinalCameraMatrix.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateCamMatrix", True))

        calcHomoCam1 = mergeCamera_widget.findChild(QPushButton, "calcH_cam1")
        calcHomoCam2 = mergeCamera_widget.findChild(QPushButton, "calcH_cam2")
        calcHHomoCam1_high = mergeCamera_widget.findChild(QPushButton, "calcH_cam1_high")
        calcHHomoCam2_high = mergeCamera_widget.findChild(QPushButton, "calcH_cam2_high")

        calcHomoCam1.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam1", True))
        calcHomoCam2.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam2", True))
        calcHHomoCam1_high.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam1_high", True))
        calcHHomoCam2_high.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam2_high", True))

        planarize_combined = mergeCamera_widget.findChild(QPushButton, "planarize")
        planarize_combined_high = mergeCamera_widget.findChild(QPushButton, "planarize_high")

        planarize_combined.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize", True))
        planarize_combined_high.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarizeHigh", True))
        
        button_config = {
            "P658107YA0Abutton": {"widget_index": 5, "inspection_param": 5},
            "P808387YA0Abutton": {"widget_index": 6, "inspection_param": 6},
            "P828387YA0Abutton": {"widget_index": 7, "inspection_param": 7},
            "P828387YA6Abutton": {"widget_index": 8, "inspection_param": 8},
            "P828397YA6Abutton": {"widget_index": 9, "inspection_param": 9},
            "P828387YA1Abutton": {"widget_index": 10, "inspection_param": 10},
            "P828397YA1Abutton": {"widget_index": 11, "inspection_param": 11},
            "P731957YA0Abutton": {"widget_index": 12, "inspection_param": 12},
            "P8462284S00button": {"widget_index": 13, "inspection_param": 13},
        }

        for button_name, config in button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                # Connect each signal with the necessary parameters
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
                button.clicked.connect(self.calibration_thread.stop)



        # partInspection_P658107YA0A_button = main_widget.findChild(QPushButton, "P658107YA0Abutton")
        # partInspection_P658107YA0A_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(5))
        # partInspection_P658107YA0A_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 5))
        # partInspection_P658107YA0A_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        # partInspection_P658107YA0A_button.clicked.connect(self.calibration_thread.stop)

        # partInspection_P808387YA0A_button = main_widget.findChild(QPushButton, "P808387YA0Abutton")
        # partInspection_P808387YA0A_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        # partInspection_P808387YA0A_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 6))
        # partInspection_P808387YA0A_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        # partInspection_P808387YA0A_button.clicked.connect(self.calibration_thread.stop)

        # partInspection_P828387YA0A_button = main_widget.findChild(QPushButton, "P828387YA0Abutton")


        # partInspection_P828387YA6A_button = main_widget.findChild(QPushButton, "P828387YA6Abutton")
        # partInspection_P828397YA6A_button = main_widget.findChild(QPushButton, "P828397YA6Abutton")
        # partInspection_P828387YA1A_button = main_widget.findChild(QPushButton, "P828387YA1Abutton")
        # partInspection_P828397YA1A_button = main_widget.findChild(QPushButton, "P828397YA1Abutton")
        # partInspection_P731957YA0A_button = main_widget.findChild(QPushButton, "P731957YA0Abutton")

        # partInspection_P658207LE0A_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(7))
        # partInspection_P658207LE0A_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 7))
        # partInspection_P658207LE0A_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        # partInspection_P658207LE0A_button.clicked.connect(self.calibration_thread.stop)

        # partInspection_P5819A107_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(8))
        # partInspection_P5819A107_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 8))
        # partInspection_P5819A107_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        # partInspection_P5819A107_button.clicked.connect(self.calibration_thread.stop)

        dailytenken01_P5902A509_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        dailytenken01_P5902A509_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))
        dailytenken01_P5902A509_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken01_P5902A509_button.clicked.connect(self.calibration_thread.stop)

        dailytenken02_P5902A509_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        dailytenken02_P5902A509_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 22))
        dailytenken02_P5902A509_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken02_P5902A509_button.clicked.connect(self.calibration_thread.stop)

        dailytenken03_P5902A509_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(23))
        dailytenken03_P5902A509_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 23))
        dailytenken03_P5902A509_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken03_P5902A509_button.clicked.connect(self.calibration_thread.stop)

        dailytenken01_P658207LE0A_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(24))
        dailytenken01_P658207LE0A_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 24))
        dailytenken01_P658207LE0A_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken01_P658207LE0A_button.clicked.connect(self.calibration_thread.stop)

        dailytenken02_P658207LE0A_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(25))
        dailytenken02_P658207LE0A_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 25))
        dailytenken02_P658207LE0A_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken02_P658207LE0A_button.clicked.connect(self.calibration_thread.stop)


        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25]]

        self.siostatus_server = [self.stackedWidget.widget(i).findChild(QLabel, "status_sio") for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25]]

        self.inspection_widget_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25]

        for i in self.inspection_widget_indices:
            self.Inspect_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectButton")
            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))

        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
            self.connect_inspectionConfig_button(i, "kansei_plus", "kansei_plus", True)
            self.connect_inspectionConfig_button(i, "kansei_minus", "kansei_minus", True)
            self.connect_inspectionConfig_button(i, "furyou_plus", "furyou_plus", True)
            self.connect_inspectionConfig_button(i, "furyou_minus", "furyou_minus", True)
            self.connect_inspectionConfig_button(i, "kansei_plus_10", "kansei_plus_10", True)
            self.connect_inspectionConfig_button(i, "kansei_minus_10", "kansei_minus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_plus_10", "furyou_plus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_minus_10", "furyou_minus_10", True)
            #connect reset button
            self.connect_inspectionConfig_button(i, "counterReset", "counterReset", True)

        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 0))

                # dailytenken_kanryou_P5902A509_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                # dailytenken_kanryou_P658207LE0A_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                # dailytenken_kanryou_P5902A509_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 0))
                # dailytenken_kanryou_P658207LE0A_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 0))


    #     #kensabutton for dailytenken
    #     self.button_dailyTenken01 = self.stackedWidget.widget(21).findChild(QPushButton, "checkButton")
    #     self.button_dailyTenken02 = self.stackedWidget.widget(22).findChild(QPushButton, "checkButton")
    #     self.button_dailyTenken03 = self.stackedWidget.widget(23).findChild(QPushButton, "checkButton")

    #     self.button_dailyTenken01.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "kensaButton", True))
    #     self.button_dailyTenken02.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "kensaButton", True))
    #     self.button_dailyTenken03.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "kensaButton", True))

    #     self.button_dailyTenken01.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))
    #     self.button_dailyTenken02.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))
    #     self.button_dailyTenken03.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "triggerKensa", True))

        # self.stackedWidget.currentChanged.connect(self._on_widget_changed)

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()

    def connect_button_font_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: black")
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)
                new_color = "red" if new_value else "black"
                button.setStyleSheet(f"color: {new_color}")
            button.pressed.connect(toggle_font_color_and_param)
        else:
            print(f"Button '{qtbutton}' not found.")

    def connect_button_label_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: red")
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)
                new_color = "green" if new_value else "red"
                button.setStyleSheet(f"color: {new_color}")

            button.pressed.connect(toggle_font_color_and_param)
        else:
            print(f"Button '{qtbutton}' not found.")

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_cam_params(self.cam_thread, cam_param, text))

    def connect_inspectionConfig_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_inspection_params(self.inspection_thread, cam_param, value))
            # print(f"Button '{button_name}' connected to cam_param '{cam_param}' with value '{value}' in widget {widget_index}")

    def _close_app(self):
        # self.cam_thread.stop()
        self.calibration_thread.stop()
        self.inspection_thread.stop()
        self.server_monitor_thread.stop()
        time.sleep(1.0)
        QCoreApplication.instance().quit()

    def _load_ui(self, filename):
        widget = QMainWindow()
        loadUi(filename, widget)
        return widget

    def _set_frame_raw(self, image):
        for i in [1, 2]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_frame_inference(self, image):
        for i in [3, 4]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "cameraFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_cam_params(self, thread, key, value):
        setattr(thread.cam_config, key, value)

    def _toggle_param_and_update_label(self, param, label):
        new_value = not getattr(self.cam_thread.cam_config, param)
        self._set_cam_params(self.cam_thread, param, new_value)

        color = "green" if new_value else "red"
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _update_OKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "current_kansei")
                    current_furyou_label = widget.findChild(QLabel, "current_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for numofPart")

    def _update_todayOKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "status_kansei")
                    current_furyou_label = widget.findChild(QLabel, "status_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for todaynumofPart")

#5
    def _outputMeasurementText_P658107YA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [5]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#6
    def _outputMeasurementText_P808387YA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label"]
        for widget_index in [6]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#7
    def _outputMeasurementText_P828387YA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label"]
        for widget_index in [7]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#8
    def _outputMeasurementText_P828387YA6A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [8]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#9
    def _outputMeasurementText_P828397YA6A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [9]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#10
    def _outputMeasurementText_P828387YA1A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label"]
        for widget_index in [10]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#11
    def _outputMeasurementText_P828397YA1A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label"]
        for widget_index in [11]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#12
    def _outputMeasurementText_P731957YA0A(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label", "P12label"]
        for widget_index in [12]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")
#13
    def _outputMeasurementText_P8462284S00(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label"]
        for widget_index in [13]:
            # Loop through the label names (P1label, P2label, etc.)
            for label_index, label_name in enumerate(label_names_part):
                # Find the QLabel in the specified widget
                label = self.stackedWidget.widget(widget_index).findChild(QLabel, label_name)
                if label:
                    # Get the measurement value for this label
                    if (measurementValue and isinstance(measurementValue, list) and len(measurementValue) > 0 
                        and isinstance(measurementValue[0], list) and len(measurementValue[0]) > label_index):
                        
                        value = measurementValue[0][label_index] if measurementValue[0][label_index] is not None else "None"
                    else:
                        value = "None"  # Fallback to "None" or "0"
                    
                    # Set text for the label
                    label.setText(str(value))

                    # Get the measurement result for this label
                    if (measurementResult and isinstance(measurementResult, list) and len(measurementResult) > 0 
                        and isinstance(measurementResult[0], list) and len(measurementResult[0]) > label_index):
                        result = measurementResult[0][label_index] if measurementResult[0][label_index] is not None else "None"
                    else:
                        result = "None"  # Fallback to "None" or "0"

                    # Set label background color based on result
                    if result == 1:  # OK result (1)
                        label.setStyleSheet("background-color: green;")
                    elif result == 0:  # NG result (0)
                        label.setStyleSheet("background-color: red;")
                    else:
                        label.setStyleSheet("background-color: white;")

    def _set_labelFrame(self, widget, paramValue, label_names):
        colorOK = "blue"
        colorNG = "black"
        label = widget.findChild(QLabel, label_names) 
        color = colorNG if paramValue else colorOK
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")
        
    def _set_button_color(self, pitch_data):
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")


    def _setCalibFrame(self, image):
        for i in [1, 2 ]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame1(self, image):
        widget = self.stackedWidget.widget(3)
        label = widget.findChild(QLabel, "camMerge1")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame2(self, image):
        widget = self.stackedWidget.widget(3)
        label = widget.findChild(QLabel, "camMerge2")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrameAll(self, image):
        widget = self.stackedWidget.widget(3)
        label = widget.findChild(QLabel, "camMergeAll")
        label.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame1(self, image):
        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "framePart")
            label.setPixmap(QPixmap.fromImage(image))

    def _extract_color(self, stylesheet):
        # Extracts the color value from the stylesheet string
        start = stylesheet.find("background-color: ") + len("background-color: ")
        end = stylesheet.find(";", start)
        return stylesheet[start:end].strip()

    def _store_initial_colors(self, widget_index, label_names):
        if widget_index not in self.initial_colors:
            self.initial_colors[widget_index] = {}
        labels = [self.stackedWidget.widget(widget_index).findChild(QLabel, name) for name in label_names]
        for label in labels:
            color = self._extract_color(label.styleSheet())
            self.initial_colors[widget_index][label.objectName()] = color
            # print(f"Stored initial color for {label.objectName()} in widget {widget_index}: {color}")

    def _set_calib_params(self, thread, key, value):
        setattr(thread.calib_config, key, value)

    def _set_inspection_params(self, thread, key, value):
        setattr(thread.inspection_config, key, value)


def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()