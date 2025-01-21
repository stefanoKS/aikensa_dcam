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
    'aikensa/qtui/mainPage.ui',         #index 0
    'aikensa/qtui/calibration_cam1.ui', #index 1
    'aikensa/qtui/calibration_cam2.ui', #index 2         
    'aikensa/qtui/camera_merge.ui',     #index 3
    "aikensa/qtui/empty.ui",            #empty 4
    "aikensa/qtui/P82833W050P.ui",      #index 5
    "aikensa/qtui/P82832W040P.ui",      #index 6
    "aikensa/qtui/P82833W090P.ui",      #index 7
    "aikensa/qtui/P82832W080P.ui",      #index 8
    "aikensa/qtui/P82833W050PKENGEN.ui",      #index 9
    "aikensa/qtui/P82832W040PKENGEN.ui",      #index 10
    "aikensa/qtui/P82833W090PKENGEN.ui",      #index 11
    "aikensa/qtui/P82832W080PKENGEN.ui",      #index 12
    "aikensa/qtui/P82833W050PCLIPSOUNYUUKI.ui",      #index 13
    "aikensa/qtui/P82832W040PCLIPSOUNYUUKI.ui",      #index 14
    "aikensa/qtui/P82833W090PCLIPSOUNYUUKI.ui",      #index 15
    "aikensa/qtui/P82832W080PCLIPSOUNYUUKI.ui",      #index 16
    "aikensa/qtui/empty.ui", #empty 17
    "aikensa/qtui/empty.ui", #empty 18
    "aikensa/qtui/empty.ui", #empty 19
    "aikensa/qtui/empty.ui", #empty 20
    "aikensa/qtui/dailyTenken2go3go_01.ui",  # index 21
    "aikensa/qtui/dailyTenken2go3go_02.ui",  # index 22
    "aikensa/qtui/dailyTenken2go3go_03.ui",  # index 23
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
        PORT = 40001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.05)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.start()

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.initial_colors = {}#store initial colors of the labels

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

        self.prevTriggerStates = 0
        self.TriggerWaitTime = 3.0
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

        self.inspection_thread.partCam.connect(self._setPartFrame)
        self.inspection_thread.partKatabuL.connect(self._setFrameKatabuL)
        self.inspection_thread.partKatabuR.connect(self._setFrameKatabuR)

        self.inspection_thread.clip1Signal.connect(self._setClip1Frame)
        self.inspection_thread.clip2Signal.connect(self._setClip2Frame)
        self.inspection_thread.clip3Signal.connect(self._setClip3Frame)

        self.inspection_thread.ethernetStatus.connect(self._setEthernetStatus)

        self.inspection_thread.P82833W050P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050P)
        self.inspection_thread.P82832W040P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040P)
        self.inspection_thread.P82833W090P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090P)
        self.inspection_thread.P82832W080P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080P)

        self.inspection_thread.P82833W050PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050PKENGEN)
        self.inspection_thread.P82832W040PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040PKENGEN)
        self.inspection_thread.P82833W090PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090PKENGEN)
        self.inspection_thread.P82832W080PKENGEN_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080PKENGEN)  

        self.inspection_thread.P82833W050PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050PCLIPSOUNYUUKI)
        self.inspection_thread.P82832W040PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040PCLIPSOUNYUUKI)
        self.inspection_thread.P82833W090PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090PCLIPSOUNYUUKI)
        self.inspection_thread.P82832W080PCLIPSOUNYUUKI_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080PCLIPSOUNYUUKI)

        self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        dailyTenken2go3go_01_widget = self.stackedWidget.widget(21)
        dailyTenken2go3go_02_widget = self.stackedWidget.widget(22)
        dailyTenken2go3go_03_widget = self.stackedWidget.widget(23)

        cameraCalibration1_widget = self.stackedWidget.widget(1)
        cameraCalibration2_widget = self.stackedWidget.widget(2)
        mergeCamera_widget = self.stackedWidget.widget(3)

        cameraCalibration1_button = main_widget.findChild(QPushButton, "camcalibrationbutton1")
        cameraCalibration2_button = main_widget.findChild(QPushButton, "camcalibrationbutton2")
        mergeCamera_button = main_widget.findChild(QPushButton, "cameraMerge")

        dailytenken01_button = main_widget.findChild(QPushButton, "dailytenkenbutton")
        dailytenken02_button = dailyTenken2go3go_01_widget.findChild(QPushButton, "nextButton")
        dailytenken03_button = dailyTenken2go3go_02_widget.findChild(QPushButton, "nextButton")
        dailytenken_kanryou_button = dailyTenken2go3go_03_widget.findChild(QPushButton, "finishButton")


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
            "P82833W050Pbutton": {"widget_index": 5, "inspection_param": 5},
            "P82832W040Pbutton": {"widget_index": 6, "inspection_param": 6},
            "P82833W090Pbutton": {"widget_index": 7, "inspection_param": 7},
            "P82832W080Pbutton": {"widget_index": 8, "inspection_param": 8},
            "P82833W050PKENGENbutton": {"widget_index": 9, "inspection_param": 9},
            "P82832W040PKENGENbutton": {"widget_index": 10, "inspection_param": 10},
            "P82833W090PKENGENbutton": {"widget_index": 11, "inspection_param": 11},
            "P82832W080PKENGENbutton": {"widget_index": 12, "inspection_param": 12},
            "P82833W050PCLIPSOUNYUUKIbutton": {"widget_index": 13, "inspection_param": 13},
            "P82832W040PCLIPSOUNYUUKIbutton": {"widget_index": 14, "inspection_param": 14},
            "P82833W090PCLIPSOUNYUUKIbutton": {"widget_index": 15, "inspection_param": 15},
            "P82832W080PCLIPSOUNYUUKIbutton": {"widget_index": 16, "inspection_param": 16},
        }


        for button_name, config in button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                # Connect each signal with the necessary parameters
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
                button.clicked.connect(self.calibration_thread.stop)

        dailytenken01_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        dailytenken01_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))
        dailytenken01_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken01_button.clicked.connect(self.calibration_thread.stop)

        dailytenken02_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        dailytenken02_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 22))
        dailytenken02_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken02_button.clicked.connect(self.calibration_thread.stop)

        dailytenken03_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(23))
        dailytenken03_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 23))
        dailytenken03_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken03_button.clicked.connect(self.calibration_thread.stop)


        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23]]

        self.siostatus_server = [self.stackedWidget.widget(i).findChild(QLabel, "status_sio") for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23]]

        self.inspection_widget_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        for i in self.inspection_widget_indices:
            self.Inspect_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectButton")
            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))

        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
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

            self.connect_line_edit_text_changed(widget_index=i, line_edit_name="kensain_name", inspection_param="kensainNumber")

        

        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 0))

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

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_inspection_params(self.inspection_thread, inspection_param, text))

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
    def _outputMeasurementText_P82833W050P(self, measurementValue, measurementResult):
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
    def _outputMeasurementText_P82832W040P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
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
    def _outputMeasurementText_P82833W090P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
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
    def _outputMeasurementText_P82832W080P(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
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
    def _outputMeasurementText_P82833W050PKENGEN(self, measurementValue, measurementResult):
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
    def _outputMeasurementText_P82832W040PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
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
    def _outputMeasurementText_P82833W090PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
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
    def _outputMeasurementText_P82832W080PKENGEN(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
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
    def _outputMeasurementText_P82833W050PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
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

#14
    def _outputMeasurementText_P82832W040PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label"]
        for widget_index in [14]:
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

#15
    def _outputMeasurementText_P82833W090PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [15]:
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

#16
    def _outputMeasurementText_P82832W080PCLIPSOUNYUUKI(self, measurementValue, measurementResult):
        label_names_part = ["P1label", "P2label", "P3label", "P4label", "P5label", "P6label", "P7label", "P8label", "P9label", "P10label", "P11label"]
        for widget_index in [16]:
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

    def _setPartFrame(self, image):
        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "framePart")
            label.setPixmap(QPixmap.fromImage(image))

    def _setFrameKatabuL(self, image):
        for i in [5, 6, 7, 8, 9, 10, 11, 12]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "frameKatabuL")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip1Frame(self, image):
        for i in [5, 6, 7, 8]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip1Frame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip2Frame(self, image):
        for i in [5, 6, 7, 8]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip2Frame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip3Frame(self, image):
        for i in [5, 6, 7, 8]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip3Frame")
            label.setPixmap(QPixmap.fromImage(image))
    
    def _setFrameKatabuR(self, image):
        for i in [5, 6, 7, 8, 9, 10, 11, 12]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "frameKatabuR")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip1Frame(self, image):
        for i in [5, 6, 7, 8]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip1Frame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setClip2Frame(self, image):
        for i in [5, 6, 7, 8]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip2Frame")
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

    def _setEthernetStatus(self, input):
        self.server_monitor_thread.server_config.eth_flag_0_4 = input

def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()