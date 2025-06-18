# livestream_with_yaml.py
import sys
import yaml
import imagingcontrol4 as ic4
import cv2

from PyQt5.QtCore import (
    QObject, QThread, pyqtSignal, pyqtSlot, Qt
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QLabel, QHBoxLayout, QWidget
)


class CameraThread(QThread):
    buffer_ready = pyqtSignal(object, int)  # (IC4 buffer, logical_cam_id)

    def __init__(self, logical_id, dev_info, parent=None):
        super().__init__(parent)
        self.logical_id = logical_id
        self.dev_info   = dev_info
        self._running   = True

    def run(self):
        with ic4.Library.init_context(
            api_log_level=ic4.LogLevel.INFO,
            log_targets=ic4.LogTarget.STDERR
        ):
            grabber = ic4.Grabber(self.dev_info)
            sink    = ic4.FrameQueueSink(max_buffers=4)
            grabber.stream_setup(sink)

            while self._running:
                entry = sink.get_next(timeout_ms=1000)
                if entry:
                    self.buffer_ready.emit(entry.buffer, self.logical_id)

            grabber.stream_stop()
            grabber.device_close()

    def stop(self):
        self._running = False
        self.wait()
