import imagingcontrol4 as ic4
from PyQt5.QtCore import QThread, pyqtSignal


class ICCameraThread(QThread):
    cam_frame  = pyqtSignal(np.ndarray)

    def __init__(self, device_info = None):
        if 