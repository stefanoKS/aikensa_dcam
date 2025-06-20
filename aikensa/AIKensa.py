import re
import cv2
import sys
from matplotlib.pylab import f
import yaml
import os
from enum import Enum
import time
import datetime
import imagingcontrol4 as ic4

from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QColor

from aikensa.thread.calibration_thread import CalibrationThread, CalibrationConfig
from aikensa.thread.inspection_thread import InspectionThread, InspectionConfig
from aikensa.thread.time_thread import TimeMonitorThread
from aikensa.thread.modbus_thread import ModbusServerThread
from aikensa.thread.camera_thread import CameraThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',         #index 0
    'aikensa/qtui/CALIBRATION/calibration_cam_left_1.ui', #index 1
    'aikensa/qtui/CALIBRATION/calibration_cam_left_2.ui', #index 2         
    'aikensa/qtui/CALIBRATION/camera_left_merge.ui',     #index 3
    'aikensa/qtui/CALIBRATION/calibration_cam_right_1.ui', #index 4
    'aikensa/qtui/CALIBRATION/calibration_cam_right_2.ui', #index 5         
    'aikensa/qtui/CALIBRATION/camera_right_merge.ui',     #index 6
    "aikensa/qtui/NISSAN/M_JC2D/P808397UA0A.ui",      #index 7
    "aikensa/qtui/NISSAN/M_JC2D/P808387UA0A.ui",      #index 8
    "aikensa/qtui/empty.ui", #empty 9
    "aikensa/qtui/empty.ui", #empty 10
    "aikensa/qtui/empty.ui", #empty 11
    "aikensa/qtui/empty.ui", #empty 12
    "aikensa/qtui/empty.ui", #empty 13
    "aikensa/qtui/empty.ui", #empty 14
    "aikensa/qtui/empty.ui", #empty 15
    "aikensa/qtui/empty.ui", #empty 16
    "aikensa/qtui/empty.ui", #empty 17
    "aikensa/qtui/empty.ui", #empty 18
    "aikensa/qtui/empty.ui", #empty 19
    "aikensa/qtui/empty.ui", #empty 20
    "aikensa/qtui/dailyTenken/dailyTenken_01.ui",  # index 21
    "aikensa/qtui/dailyTenken/dailyTenken_02.ui",  # index 22
    "aikensa/qtui/dailyTenken/dailyTenken_03.ui",  # index 23
]

class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.calibration_thread = CalibrationThread(CalibrationConfig())
        self.inspection_thread = InspectionThread(InspectionConfig()) 

        self._detect_screens()

        self.secondary = QMainWindow()
        self.secondary_stack = QStackedWidget()

        self._setup_ui()
        self._show_left_fullscreen()

        empty_right = self._load_ui(UI_FILES[9])  # ← empty.ui
        self.secondary_stack.addWidget(empty_right)

        right_page = self._load_ui(UI_FILES[8])   # ← P808387UA0A.ui
        self.secondary_stack.addWidget(right_page)

        self.secondary.setCentralWidget(self.secondary_stack)
        self.secondary.move(self.right_geo.topLeft())
        self.secondary.showFullScreen()

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.initial_colors = {}#store initial colors of the labels

        self.widget_dir_map = {
            7: "808397UA0A",
            8: "808387UA0A",
            21: "dailyTenken_01",
            22: "dailyTenken_02",
            23: "dailyTenken_03",
        }


    def timeUpdate(self, time):
        for label in self.timeLabel:
            if label:
                label.setText(time)

    def trigger_kensa(self):
        self.Inspect_button.click()

    def _setup_ui(self):

        self.calibration_thread.CalibCamStream.connect(self._setCalibFrame)

        self.calibration_thread.CamMerge1.connect(self._setMergeFrame1)
        self.calibration_thread.CamMerge2.connect(self._setMergeFrame2)
        self.calibration_thread.CamMergeAll.connect(self._setMergeFrameAll)

        # self.inspection_thread.partCam.connect(self._setPartFrame)
        # self.inspection_thread.partKatabuL.connect(self._setFrameKatabuL)
        # self.inspection_thread.partKatabuR.connect(self._setFrameKatabuR)

        # self.inspection_thread.clip1Signal.connect(self._setClip1Frame)
        # self.inspection_thread.clip2Signal.connect(self._setClip2Frame)
        # self.inspection_thread.clip3Signal.connect(self._setClip3Frame)

        # self.inspection_thread.ethernetStatus.connect(self._setEthernetStatus)

        # self.inspection_thread.P82833W050P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W050P)
        # self.inspection_thread.P82832W040P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W040P)
        # self.inspection_thread.P82833W090P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82833W090P)
        # self.inspection_thread.P82832W080P_InspectionResult_PitchMeasured.connect(self._outputMeasurementText_P82832W080P)

        # self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        # self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        dailyTenken01_widget = self.stackedWidget.widget(21)
        dailyTenken02_widget = self.stackedWidget.widget(22)
        dailyTenken03_widget = self.stackedWidget.widget(23)

        dailytenken01_button = main_widget.findChild(QPushButton, "dailytenken_button")
        dailytenken02_button = dailyTenken01_widget.findChild(QPushButton, "nextButton")
        dailytenken02_back_button = dailyTenken02_widget.findChild(QPushButton, "prevButton")
        dailytenken03_button = dailyTenken02_widget.findChild(QPushButton, "nextButton")
        dailytenken03_back_button = dailyTenken03_widget.findChild(QPushButton, "prevButton")
        dailytenken_kanryou_button = dailyTenken03_widget.findChild(QPushButton, "finishButton")

        camera_calibration_left_1_widget = self.stackedWidget.widget(1)
        camera_calibration_left_2_widget = self.stackedWidget.widget(2)
        camera_left_merge_widget = self.stackedWidget.widget(3)

        camera_calibration_right_1_widget = self.stackedWidget.widget(4)
        camera_calibration_right_2_widget = self.stackedWidget.widget(5)
        camera_right_merge_widget = self.stackedWidget.widget(6)

        calib_map = {
            1: "camcalibration_left_1_button",
            2: "camcalibration_left_2_button",
            3: "camera_left_merge_button",
            4: "camcalibration_right_1_button",
            5: "camcalibration_right_2_button",
            6: "camera_right_merge_button",
        }

        for idx, btn_name in calib_map.items():
            btn = main_widget.findChild(QPushButton, btn_name)
            if not btn:
                continue
            btn.clicked.connect(lambda _, i=idx: self.stackedWidget.setCurrentIndex(i))
            btn.clicked.connect(lambda _, i=idx: self._set_calib_params(self.calibration_thread, 'widget', i))
            btn.clicked.connect(self.calibration_thread.start)
            

        for i in [1, 2, 4, 5]:
            CalibrateSingleFrame = self.stackedWidget.widget(i).findChild(QPushButton, "calibSingleFrame")
            CalibrateSingleFrame.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateSingeFrameMatrix", True))

            CalibrateFinalCameraMatrix = self.stackedWidget.widget(i).findChild(QPushButton, "calibCam")
            CalibrateFinalCameraMatrix.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateCamMatrix", True))


        calcHomoCam1_left_button = camera_left_merge_widget.findChild(QPushButton, "calcH_cam1_button")
        calcHomoCam2_left_button = camera_left_merge_widget.findChild(QPushButton, "calcH_cam2_button")
        planarize_combined_camera_left = camera_left_merge_widget.findChild(QPushButton, "planarize_button")

        calcHomoCam1_left_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam1", True))
        calcHomoCam2_left_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam2", True))
        planarize_combined_camera_left.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize_left", True))

        calcHomoCam1_right_button = camera_right_merge_widget.findChild(QPushButton, "calcH_cam1_button")
        calcHomoCam2_right_button = camera_right_merge_widget.findChild(QPushButton, "calcH_cam2_button")
        planarize_combined_camera_right = camera_right_merge_widget.findChild(QPushButton, "planarize_button")

        calcHomoCam1_right_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam3", True))
        calcHomoCam2_right_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "calculateHomo_cam4", True))
        planarize_combined_camera_right.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, "savePlanarize_right", True))


        inspection_button_config = {
            "P808387UA1A_button": {"widget_index": 7, "inspection_param": 7},
        }

        for button_name, config in inspection_button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                button.clicked.connect(self.calibration_thread.stop)
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)

        dailytenken01_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        dailytenken01_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))
        dailytenken01_button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
        dailytenken01_button.clicked.connect(self.calibration_thread.stop)

        dailytenken02_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        dailytenken02_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 22))

        dailytenken02_back_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(21))
        dailytenken02_back_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 21))

        dailytenken03_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(23))
        dailytenken03_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 23))

        dailytenken03_back_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(22))
        dailytenken03_back_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 22))

        self.widget_indices_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.inspection_widget_indices = [7, 8, 21, 22, 23]
        self.inspection_widget_indices_without_dailytenken = [7, 8]

        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in self.widget_indices_list]


        for i in self.inspection_widget_indices:
            self.Inspect_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectButton")
            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))

        for i in self.inspection_widget_indices_without_dailytenken:
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

            #additional logic for ppms number
            if i in [13, 14, 15, 16, 17, 18]:
                self.connect_line_edit_text_changed(widget_index=i, line_edit_name="ppms_number", inspection_param="ppmsnumber")

        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 0))

        self.stackedWidget.currentChanged.connect(self._on_page_changed)
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
        self.calibration_thread.stop()
        self.inspection_thread.stop()
        self.calibration_thread.quit()
        self.inspection_thread.quit()
        self.calibration_thread.wait(1000)
        self.inspection_thread.wait(1000)
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
        for i in [1, 2, 4, 5]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame1(self, image):
        for i in [3, 6]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camMerge1")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame2(self, image):
        for i in [3, 6]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camMerge2")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrameAll(self, image):
        for i in [3, 6]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camMergeAll")
            label.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame(self, image):
        for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "framePart")
            label.setPixmap(QPixmap.fromImage(image))

    def _set_calib_params(self, thread, key, value):
        setattr(thread.calib_config, key, value)

    def _set_inspection_params(self, thread, key, value):
        setattr(thread.inspection_config, key, value)

    def _detect_screens(self):
        screens = QApplication.screens()
        if len(screens) < 2:
            raise RuntimeError("Two monitors required")
        self.left_geo  = screens[0].geometry()
        self.right_geo = screens[1].geometry()

    def _on_page_changed(self, idx):
        print("Left page changed to", idx)
        if idx == 7:
            print(" → showing RIGHT page")
            self.secondary_stack.setCurrentIndex(1)
        else:
            print(" → showing EMPTY page")
            self.secondary_stack.setCurrentIndex(0)


    def _show_left_fullscreen(self):
        self.move(self.left_geo.topLeft())
        self.showFullScreen()

def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()