import cv2
import sys
import yaml
import os
from enum import Enum
import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.cam_thread import CameraThread, CameraConfig

from aikensa.sio_thread import ServerMonitorThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',             # index 0
    'aikensa/qtui/calibration_multicam.ui', # index 1
    'aikensa/qtui/edgedetection.ui',        # index 2
    "aikensa/qtui/5755A491.ui",             # index 3
    "aikensa/qtui/5755A492.ui"              # index 4    
]


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cam_thread = CameraThread(CameraConfig())
        self._setup_ui()
        self.cam_thread.start()

        # Thread for SiO
        HOST = '192.168.0.100'  # Use the IP address from SiO settings
        PORT = 30001  # Use the port number from SiO settings

        self.server_monitor_thread = ServerMonitorThread(
            HOST, PORT, check_interval=0.1)
        self.server_monitor_thread.server_status_signal.connect(self.handle_server_status)
        self.server_monitor_thread.input_states_signal.connect(self.handle_input_states)
        self.server_monitor_thread.start()

    def handle_server_status(self, is_up):
        status_text = "ON" if is_up else "OFF"
        status_color = "green" if is_up else "red"

        #to show the label later. Implement later

        # for label in self.siostatus_cowltop:
        #     if label:  # Check if the label was found correctly
        #         label.setText(status_text)
        #         label.setStyleSheet(f"color: {status_color};")


    def handle_input_states(self, input_states):
        # check if input_stats is not empty
        if input_states:
            if input_states[0] == 1:
                self.trigger_kensa()
            if input_states[1] == 1:
                self.trigger_rekensa()
            else:
                pass

    def trigger_kensa(self):
        self.button_kensa5.click()
        self.button_kensa6.click()
        self.button_kensa7.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):
        self.cam_thread.camFrame1.connect(self._setFrameCam1)
        self.cam_thread.camFrame2.connect(self._setFrameCam2)
        self.cam_thread.mergeFrame.connect(self._setFrameMerge)
        
        self.cam_thread.kata1Frame.connect(self._setFrameKata1)
        self.cam_thread.kata2Frame.connect(self._setFrameKata2)
        
        self.cam_thread.clip1Frame.connect(self._setFrameClip1)
        self.cam_thread.clip2Frame.connect(self._setFrameClip2)
        self.cam_thread.clip3Frame.connect(self._setFrameClip3)

        # self.cam_thread.cowl_pitch_updated.connect(self._set_button_color)
        # self.cam_thread.cowl_numofPart_updated.connect(self._set_numlabel_text)
        
        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        button_calib = main_widget.findChild(QPushButton, "calibrationbutton")
        button_edgedetect = main_widget.findChild(QPushButton, "edgedetectbutton")

        button_P5755A491 = main_widget.findChild(QPushButton, "P5755A491button")
        button_P5755A492 = main_widget.findChild(QPushButton, "P5755A492button")

        self.siostatus = main_widget.findChild(QLabel, "status_sio")

        if button_calib:
            button_calib.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            button_calib.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 1))

        if button_edgedetect:
            button_edgedetect.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
            button_edgedetect.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 2))

        if button_P5755A491:
            button_P5755A491.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
            button_P5755A491.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 3))

        if button_P5755A492:
            button_P5755A492.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
            button_P5755A492.clicked.connect(lambda: self._set_cam_params(self.cam_thread, 'widget', 4))

        # add extra widgets here

        # Widget 0 -> <Main Page>

        
        #Widget 1 # Calibration and Sample
        captureButton1 = self.stackedWidget.widget(1).findChild(QPushButton, "takeImage1")
        captureButton2 = self.stackedWidget.widget(1).findChild(QPushButton, "takeImage2")
        captureButton1.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "captureCam1", True))
        captureButton2.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "captureCam2", True))

        cam1CalibrateButton = self.stackedWidget.widget(1).findChild(QPushButton, "calibCam1")
        cam1CalibrateButton.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "calculateCamMatrix1", True))

        cam2CalibrateButton = self.stackedWidget.widget(1).findChild(QPushButton, "calibCam2")
        cam2CalibrateButton.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "calculateCamMatrix2", True))

        delCamMatrix1 = self.stackedWidget.widget(1).findChild(QPushButton, "delCalib1")
        delCamMatrix2 = self.stackedWidget.widget(1).findChild(QPushButton, "delCalib2")
        delCamMatrix1.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "delCamMatrix1", True))
        delCamMatrix2.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "delCamMatrix2", True))

        checkUndistort1 = self.stackedWidget.widget(1).findChild(QPushButton, "checkUndistort1")
        checkUndistort2 = self.stackedWidget.widget(1).findChild(QPushButton, "checkUndistort2")
        checkUndistort1.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "checkUndistort1", True))
        checkUndistort2.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "checkUndistort2", True))

        calculateHomo = self.stackedWidget.widget(1).findChild(QPushButton, "calcH")
        calculateHomo.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "calculateHomo", True))

        deleteHomo = self.stackedWidget.widget(1).findChild(QPushButton, "delH")
        deleteHomo.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "deleteHomo", False))

        combineCam = self.stackedWidget.widget(1).findChild(QPushButton, "mergeCam")
        combineCam.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "mergeCam", True))

        saveImg = self.stackedWidget.widget(1).findChild(QPushButton, "saveImage")
        saveImg.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "saveImage", True))

        savePlanarize = self.stackedWidget.widget(1).findChild(QPushButton, "savePlanarize")
        savePlanarize.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "savePlanarize", True))

        delPlanarize = self.stackedWidget.widget(1).findChild(QPushButton, "delPlanarize")
        delPlanarize.clicked.connect(lambda: self._set_cam_params(self.cam_thread, "delPlanarize", True))

        # Widget 2
        button_saveparam = self.stackedWidget.widget(2).findChild(QPushButton, "saveparambutton")
        button_saveparam.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "savecannyparams", True))

        button_takecanny = self.stackedWidget.widget(2).findChild(QPushButton, "takeimagebutton")
        button_takecanny.pressed.connect(lambda: self._set_cam_params(self.cam_thread, "capture", True))

        button_readwarp = self.stackedWidget.widget(2).findChild(QPushButton, "button_readwarp")
        label_readwarp = self.stackedWidget.widget(2).findChild(QLabel, "label_readwarpcolor")
        button_readwarp.pressed.connect(lambda: self._toggle_param_and_update_label("cannyreadwarp", label_readwarp))

        # frame = process_for_edge_detection(frame, self.slider_value)
        slider_opacity = self.stackedWidget.widget(2).findChild(QSlider, "slider_opacity")
        slider_blur = self.stackedWidget.widget(2).findChild(QSlider, "slider_blur")
        slider_lowercanny = self.stackedWidget.widget(2).findChild(QSlider, "slider_lowercanny")
        slider_uppercanny = self.stackedWidget.widget(2).findChild(QSlider, "slider_uppercanny")
        slider_contrast = self.stackedWidget.widget(2).findChild(QSlider, "slider_contrast")
        slider_brightness = self.stackedWidget.widget(2).findChild(QSlider, "slider_brightness")

        slider_opacity.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'opacity', x/100))
        slider_blur.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'blur', x))
        slider_lowercanny.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'lower_canny', x))
        slider_uppercanny.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'upper_canny', x))
        slider_contrast.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'contrast', x/100))
        slider_brightness.valueChanged.connect(lambda x: self._set_cam_params(self.cam_thread, 'brightness', x/100))

        # Widget 3 and 4

        button_HDRes = self.connect_button_font_color_change(3, "button_HDResQT", "HDRes")


        # _____________________________________________________________________________________________________
       # Find and connect quit buttons and main menu buttons in all widgets
        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(
                    lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(
                    lambda: self._set_cam_params(self.cam_thread, 'widget', 0))
                # checkaruco.clicked.connect(lambda: set_params(self.cam_thread, 'check_aruco', False))

        self.stackedWidget.currentChanged.connect(self._on_widget_changed)

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()

    def connect_button_font_color_change(self, widget_index, qtbutton, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            button.setStyleSheet("color: black")

            # Method to toggle font color and cam_param value
            def toggle_font_color_and_param():
                current_value = getattr(self.cam_thread.cam_config, cam_param, False)
                new_value = not current_value
                setattr(self.cam_thread.cam_config, cam_param, new_value)
                self._set_cam_params(self.cam_thread, cam_param, new_value)

                # Update button font color
                new_color = "red" if new_value else "black"
                button.setStyleSheet(f"color: {new_color}")

                # Print statements for debugging
                print(f"Button pressed. {cam_param} changed to {new_value}. Font color changed to {new_color}.")

            # Connect the button's pressed signal to the toggle method
            button.pressed.connect(toggle_font_color_and_param)
            print(f"Button '{qtbutton}' connected to toggle method.")
        else:
            print(f"Button '{qtbutton}' not found.")




    def connect_line_edit_text_changed(self, widget_index, line_edit_name, cam_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_cam_params(self.cam_thread, cam_param, text))


    def connect_camparam_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_cam_params(self.cam_thread, cam_param, value))


    def simulateButtonKensaClicks(self):
        # Simulate clicking multiple buttons
        self.button_kensa5.click()
        self.button_kensa6.click()
        self.button_kensa7.click()

    def _on_widget_changed(self, idx: int):
        if idx == 5 or idx == 6 or idx == 7:
            #Change widget value to equal to index of stacked widget first
            self._set_cam_params(self.cam_thread, 'widget', idx)
            self.cam_thread.initialize_model()
            

    def _close_app(self):
        self.cam_thread.stop()
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
        # Toggle the parameter value
        new_value = not getattr(self.cam_thread.cam_config, param)
        self._set_cam_params(self.cam_thread, param, new_value)

        # Update the label color based on the new parameter value
        color = "green" if new_value else "red"
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    # def _cowltop_update_label(self, param, pitchvalue, labels):
    #     pitch = getattr(self.cam_thread.cam_config, pitchvalue)
    #     self._set_cam_params(self.cam_thread, param, True)
    #     colorOK = "green"
    #     colorNG = "red"
    #     # if the pitchvalue[i] is 1, then labels[i] is green, else red
    #     for i in range(len(pitch)):
    #         color = colorOK if pitch[i] else colorNG
    #         labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_button_color(self, pitch_data):
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")

    def _set_numlabel_text(self, numofPart):
        self.kanseihin_number_cowltop.setText(str(numofPart[0]))
        self.furyouhin_number_cowltop.setText(str(numofPart[1]))

    def _set_numlabel_text_rrside_LH(self, numofPart):
        self.kanseihin_number_rrside_lh.setText(str(numofPart[0]))
        self.furyouhin_number_rrside_lh.setText(str(numofPart[1]))

    def _set_numlabel_text_rrside_RH(self, numofPart):
        self.kanseihin_number_rrside_rh.setText(str(numofPart[0]))
        self.furyouhin_number_rrside_rh.setText(str(numofPart[1]))

    def _setFrameCam1(self, image):
        widget = self.stackedWidget.widget(1)
        label = widget.findChild(QLabel, "camFrame1")
        label.setPixmap(QPixmap.fromImage(image))

    def _setFrameCam2(self, image):
        widget = self.stackedWidget.widget(1)
        label = widget.findChild(QLabel, "camFrame2") 
        label.setPixmap(QPixmap.fromImage(image))

    def _setFrameMerge(self, image):
        for i in [1, 3]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "mergeFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setFrameKata1(self, image):
        for i in [1, 3, 4]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "kata1Frame") 
            label.setPixmap(QPixmap.fromImage(image))
        # widget = self.stackedWidget.widget(1)
        # label = widget.findChild(QLabel, "kata1Frame") 
        # label.setPixmap(QPixmap.fromImage(image))    

    def _setFrameKata2(self, image):
        for i in [1, 3, 4]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "kata2Frame") 
            label.setPixmap(QPixmap.fromImage(image))
        # widget = self.stackedWidget.widget(1)
        # label = widget.findChild(QLabel, "kata2Frame") 
        # label.setPixmap(QPixmap.fromImage(image))   

    def _setFrameClip1(self, image):
        for i in [3]: #modify this later
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip1Frame") 
            label.setPixmap(QPixmap.fromImage(image))

    def _setFrameClip2(self, image):
        for i in [3]: #modify this later
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip1Frame") 
            label.setPixmap(QPixmap.fromImage(image))

    def _setFrameClip3(self, image):
        for i in [3]: #modify this later
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "clip1Frame") 
            label.setPixmap(QPixmap.fromImage(image))

    # def _close_app(self):
    #     self.cam_thread.stop()
    #     self.close()

    def _set_cam_params(self, thread, key, value):
        setattr(thread.cam_config, key, value)


def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
