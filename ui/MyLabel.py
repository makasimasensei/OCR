from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel


class MyLabel(QLabel):
    fileDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 接受拖放
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_paths = []
        for url in event.mimeData().urls():
            file_paths.append(url.toLocalFile())
        self.fileDropped.emit(file_paths)
